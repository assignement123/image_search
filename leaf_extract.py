"""
leaf_extract.py  —  Trích xuất đặc trưng lá cây → lưu SQLite database
════════════════════════════════════════════════════════════════════════
CÁCH DÙNG:
  # Build database (tên file dạng 1027.jpg, 1023.jpg ...)
  python leaf_extract.py --data <thư_mục> --db leaf.db

  # Test 1 ảnh (in vector ra terminal)
  python leaf_extract.py --test 1027.jpg

  # Build lại từ đầu (xoá DB cũ)
  python leaf_extract.py --data leaves/ --db leaf.db --rebuild

  # Xem thống kê DB
  python leaf_extract.py --stats leaf.db

GIẢ ĐỊNH:
  • Nền ảnh: TRẮNG  (lá tối hơn nền)
  • Tất cả ảnh nằm chung 1 thư mục, tên file là số (1027.jpg, 1023.jpg ...)
  • Không cần label — filename là định danh duy nhất

SCHEMA SQLite (bảng "leaves"):
  id        INTEGER PRIMARY KEY AUTOINCREMENT
  filename  TEXT UNIQUE        (vd: "1027.jpg")
  path      TEXT               (đường dẫn đầy đủ)
  efd       BLOB               (float32 × 76 chiều)
  glcm      BLOB               (float32 × 20 chiều)
  color     BLOB               (float32 ×  9 chiều)
  vein      BLOB               (float32 ×  9 chiều)
  created   TEXT               (ISO timestamp)
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pyefd import elliptic_fourier_descriptors
from scipy.stats import circmean
from scipy.interpolate import interp1d
import sqlite3
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import psycopg2


def connect_db():
    conn = psycopg2.connect(
        host="localhost",
        port=5433,      # port docker của bạn
        dbname="leaf_db",
        user="admin",
        password="admin"
    )
    return conn

# ════════════════════════════════════════════════════════════════════
# THAM SỐ TRÍCH XUẤT
# ════════════════════════════════════════════════════════════════════
HARMONICS   = 20
N_RESAMPLE  = 600
GLCM_DIST   = [1, 3, 5, 7]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS = 32

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DIM_EFD   = (HARMONICS - 1) * 4   # 76
DIM_GLCM  = len(GLCM_DIST) * 5    # 20
DIM_COLOR = 9
DIM_VEIN  = 9


# ════════════════════════════════════════════════════════════════════
# PHẦN 1 — TIỆN ÍCH CONTOUR
# ════════════════════════════════════════════════════════════════════

def _shoelace_signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _remove_self_intersections(contour: np.ndarray) -> np.ndarray:
    pts = contour.astype(np.int32)
    x0  = pts[:, 0].min() - 5
    y0  = pts[:, 1].min() - 5
    w   = int(pts[:, 0].max() - x0) + 6
    h   = int(pts[:, 1].max() - y0) + 6
    loc = pts.copy(); loc[:, 0] -= x0; loc[:, 1] -= y0
    tmp = np.zeros((h, w), np.uint8)
    cv2.drawContours(tmp, [loc], -1, 255, -1)
    cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return contour
    clean = max(cnts, key=cv2.contourArea).squeeze()
    if clean.ndim == 1:
        clean = clean.reshape(-1, 2)
    clean = clean.astype(np.float64)
    clean[:, 0] += x0; clean[:, 1] += y0
    return clean


def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
    contour = np.asarray(contour, np.float64)
    contour = _remove_self_intersections(contour)
    if _shoelace_signed_area(contour) > 0:
        contour = contour[::-1].copy()
    idx = int(np.argmin(contour[:, 0]))
    return np.roll(contour, -idx, axis=0)


def _resample_contour(contour: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    contour = np.asarray(contour, np.float64)
    if len(contour) < 4:
        return contour
    diffs   = np.diff(contour, axis=0)
    cumdist = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total   = cumdist[-1]
    if total < 1e-6:
        return contour
    t  = np.linspace(0, total, n, endpoint=False)
    fx = interp1d(cumdist, contour[:, 0])
    fy = interp1d(cumdist, contour[:, 1])
    return np.stack([fx(t), fy(t)], axis=1)


# ════════════════════════════════════════════════════════════════════
# PHẦN 2 — TIỀN XỬ LÝ  (nền TRẮNG)
# ════════════════════════════════════════════════════════════════════

def preprocess_leaf(image_path) -> tuple:
    """
    Tiền xử lý ảnh lá trên nền TRẮNG.
    Trả về: (img, contour, gray, mask, leaf_area)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu + đảo ngược: lá tối hơn nền trắng → lá = trắng sau threshold
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("Không tìm thấy contour lá")

    raw     = max(cnts, key=cv2.contourArea).squeeze().reshape(-1, 2)
    contour = _fix_contour_orientation(raw)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    leaf_area = int((mask > 0).sum())
    if leaf_area < 500:
        raise ValueError(f"Vùng lá quá nhỏ ({leaf_area} px) — kiểm tra lại ảnh")

    return img, contour, gray, mask, leaf_area


# ════════════════════════════════════════════════════════════════════
# PHẦN 3 — TRÍCH XUẤT ĐẶC TRƯNG
# ════════════════════════════════════════════════════════════════════

def extract_efd(contour: np.ndarray) -> np.ndarray:
    """
    Elliptic Fourier Descriptors — hình dạng biên lá.
    Bất biến với translation và scale.
    Output: 76 chiều  [(HARMONICS-1) × 4]
    """
    if contour is None or len(contour) < 4:
        return np.zeros(DIM_EFD, np.float32)

    contour = np.asarray(contour, np.float64).reshape(-1, 2)
    contour = _resample_contour(contour, N_RESAMPLE)
    if len(contour) < 2 * HARMONICS:
        return np.zeros(DIM_EFD, np.float32)

    coeffs = elliptic_fourier_descriptors(
        contour, order=HARMONICS + 1, normalize=False)

    a1, b1, c1, d1 = coeffs[1]
    amp1 = np.sqrt(a1**2 + b1**2 + c1**2 + d1**2)
    if amp1 > 1e-10:
        coeffs /= amp1

    # Bỏ bậc 0 (DC) và bậc 1 (dùng để normalize) → lấy bậc 2 → HARMONICS
    return coeffs[2:HARMONICS + 1, :].flatten().astype(np.float32)


def extract_glcm(gray_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Gray Level Co-occurrence Matrix — texture bề mặt lá.
    Equalize histogram trước để phân biệt tốt hơn giữa các mẫu màu tương tự.
    Output: 20 chiều  [5 props × 4 distances]
    """
    masked  = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    eq      = cv2.equalizeHist(masked)
    resized = cv2.resize(eq, (256, 256))

    q = np.clip((resized // (256 // GLCM_LEVELS)).astype(np.uint8),
                0, GLCM_LEVELS - 1)

    glcm = graycomatrix(q,
                        distances=GLCM_DIST,
                        angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS,
                        symmetric=True, normed=True)

    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    return np.concatenate(
        [graycoprops(glcm, p).mean(axis=1) for p in props]
    ).astype(np.float32)


def extract_color_moments(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Color Moments trong không gian HSV — màu sắc lá.
    Output: 9 chiều  [mean, std, skew × H, S, V]
    """
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mhsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    moments = []
    for i, ch in enumerate(cv2.split(mhsv)):
        px = ch[mask > 0].astype(np.float64)
        if len(px) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue
        # Kênh H dùng circular mean để tránh sai khi H wrap qua 0/179
        mean = circmean(px, high=179, low=0) if i == 0 else float(np.mean(px))
        std  = float(np.std(px))
        skew = float(np.cbrt(((px - mean) ** 3).mean())) if std > 1e-6 else 0.0
        moments.extend([mean, std, skew])

    return np.array(moments, np.float32)


def extract_vein_features(img: np.ndarray,
                          mask: np.ndarray,
                          leaf_area: int) -> np.ndarray:
    """
    Đặc trưng gân lá — mật độ + phân bố hướng gân.
    Output: 9 chiều  [density + 8-bin angle histogram]
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smooth   = cv2.bilateralFilter(enhanced, 9, 75, 75)

    canny = cv2.Canny(smooth, 20, 80)
    sx    = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=5)
    sy    = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=5)
    smag  = cv2.normalize(np.hypot(sx, sy), None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.addWeighted(smag, 0.6, canny, 0.4, 0)
    vein  = cv2.bitwise_and(edges, edges, mask=mask)

    k3   = np.ones((3, 3), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_OPEN, k3, iterations=1)
    vein = cv2.dilate(vein, k3, iterations=1)

    density = float((vein > 0).sum() / leaf_area) if leaf_area > 0 else 0.0

    angle_map = np.arctan2(sy, sx) * 180.0 / np.pi % 180.0
    if (vein > 0).any():
        hist, _ = np.histogram(angle_map[vein > 0],
                               bins=8, range=(0, 180), density=True)
    else:
        hist = np.zeros(8)

    return np.concatenate([[density], hist]).astype(np.float32)


def extract_features(image_path) -> dict:
    """
    Trích xuất 4 nhóm đặc trưng từ 1 ảnh.
    Trả về dict {efd, glcm, color, vein} — giá trị là np.ndarray float32.
    Ném ValueError nếu ảnh không xử lý được.
    """
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path)
    return {
        "efd":   extract_efd(contour),
        "glcm":  extract_glcm(gray, mask),
        "color": extract_color_moments(img, mask),
        "vein":  extract_vein_features(img, mask, leaf_area),
    }


# ════════════════════════════════════════════════════════════════════
# PHẦN 4 — SQLITE DATABASE
# ════════════════════════════════════════════════════════════════════

def _arr_to_blob(arr: np.ndarray) -> bytes:
    """numpy float32 → bytes để lưu SQLite BLOB."""
    return arr.astype(np.float32).tobytes()


def _blob_to_arr(blob: bytes) -> np.ndarray:
    """bytes → numpy float32."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def init_db(db_path: str) -> sqlite3.Connection:
    """Tạo kết nối SQLite và khởi tạo schema nếu chưa có."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leaves (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT    UNIQUE NOT NULL,
            path      TEXT    NOT NULL,
            efd       BLOB    NOT NULL,
            glcm      BLOB    NOT NULL,
            color     BLOB    NOT NULL,
            vein      BLOB    NOT NULL,
            created   TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # Lưu tham số extract để truy vết
    for k, v in [
        ("harmonics",   str(HARMONICS)),
        ("n_resample",  str(N_RESAMPLE)),
        ("glcm_levels", str(GLCM_LEVELS)),
        ("dim_efd",     str(DIM_EFD)),
        ("dim_glcm",    str(DIM_GLCM)),
        ("dim_color",   str(DIM_COLOR)),
        ("dim_vein",    str(DIM_VEIN)),
    ]:
        conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)", (k, v))
    conn.commit()
    return conn


def _get_done_filenames(conn):
    cur = conn.cursor()
    cur.execute("SELECT filename FROM leaf_collection")
    rows = cur.fetchall()
    return {r[0] for r in rows}


def insert_entry(conn: sqlite3.Connection,
                 filename: str, path: str,
                 feats: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO leaves
            (filename, path, efd, glcm, color, vein, created)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, path,
        _arr_to_blob(feats["efd"]),
        _arr_to_blob(feats["glcm"]),
        _arr_to_blob(feats["color"]),
        _arr_to_blob(feats["vein"]),
        datetime.now().isoformat(timespec="seconds"),
    ))

def insert_pg(conn, filename, path, feats):

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO leaf_collection
        (filename, image_path, efd, glcm, color, vein)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (filename) DO UPDATE SET
            image_path = EXCLUDED.image_path,
            efd = EXCLUDED.efd,
            glcm = EXCLUDED.glcm,
            color = EXCLUDED.color,
            vein = EXCLUDED.vein
    """,
    (
        filename,
        path,
        feats["efd"].tolist(),
        feats["glcm"].tolist(),
        feats["color"].tolist(),
        feats["vein"].tolist()
    ))

    conn.commit()
def load_all_features(conn: sqlite3.Connection) -> list:
    """
    Load toàn bộ database vào RAM — dùng cho query / retrieval.
    Trả về list of dict: {id, filename, path, efd, glcm, color, vein}
    """
    rows = conn.execute(
        "SELECT id, filename, path, efd, glcm, color, vein FROM leaves"
    ).fetchall()
    entries = []
    for rid, fname, path, efd_b, glcm_b, color_b, vein_b in rows:
        entries.append({
            "id":       rid,
            "filename": fname,
            "path":     path,
            "efd":      _blob_to_arr(efd_b),
            "glcm":     _blob_to_arr(glcm_b),
            "color":    _blob_to_arr(color_b),
            "vein":     _blob_to_arr(vein_b),
        })
    return entries


def db_stats(conn):
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM leaf_collection")
    total = cur.fetchone()[0]

    cur.execute("SELECT filename FROM leaf_collection ORDER BY id LIMIT 1")
    first = cur.fetchone()

    cur.execute("SELECT filename FROM leaf_collection ORDER BY id DESC LIMIT 1")
    last = cur.fetchone()

    print(f"\n  Database  : {total} ảnh")
    if first and last:
        print(f"  File đầu  : {first[0]}")
        print(f"  File cuối : {last[0]}")


# ════════════════════════════════════════════════════════════════════
# PHẦN 5 — BUILD DATABASE (BATCH EXTRACT)
# ════════════════════════════════════════════════════════════════════

def build_database(data_dir: str,
                   db_path:  str  = "leaf.db",
                   rebuild:  bool = False) -> None:
    """
    Duyệt toàn bộ ảnh trong data_dir, extract đặc trưng, lưu vào SQLite.

    Tham số:
        data_dir : thư mục chứa ảnh (tên file: 1027.jpg, 1023.jpg ...)
        db_path  : file SQLite output
        rebuild  : True → xoá DB cũ và extract lại toàn bộ từ đầu
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục: {data_dir}")
        sys.exit(1)

    image_files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    ])
    if not image_files:
        print(f"[LỖI] Không tìm thấy ảnh trong: {data_dir}")
        sys.exit(1)

    if rebuild and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  → Đã xoá database cũ: {db_path}")

    print(f"\n{'═'*62}")
    print(f"  BUILD DATABASE")
    print(f"  Thư mục   : {data_dir.resolve()}")
    print(f"  Tổng ảnh  : {len(image_files)}")
    print(f"  Output DB : {os.path.abspath(db_path)}")
    print(f"{'═'*62}\n")

    conn = connect_db()
    done_names = _get_done_filenames(conn)
    if done_names and not rebuild:
        print(f"  → Resume: đã có {len(done_names)} ảnh, bỏ qua.\n")

    errors  = []
    skipped = 0
    added   = 0

    try:
        for fpath in tqdm(image_files, desc="Extracting", unit="img", ncols=72):
            fname = fpath.name

            # Bỏ qua nếu đã extract (resume mode)
            if fname in done_names:
                skipped += 1
                continue

            try:
                feats = extract_features(fpath)
                insert_pg(conn, fname, str(fpath), feats)
                added += 1

                # Checkpoint mỗi 50 ảnh
                if added % 50 == 0:
                    conn.commit()
                    tqdm.write(f"  [checkpoint] {added} ảnh mới đã lưu")

            except Exception as e:
                errors.append((fname, str(e)))

    finally:
        conn.commit()  # đảm bảo lưu kể cả khi bị Ctrl+C

    # ── Tóm tắt ──────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  KẾT QUẢ BUILD")
    print(f"  ✓ Đã extract    : {added:>6} ảnh mới")
    if skipped:
        print(f"  ○ Bỏ qua        : {skipped:>6} ảnh (đã có trong DB)")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM leaf_collection")
    total_db = cur.fetchone()[0]
    print(f"  ✓ Tổng trong DB : {total_db:>5} ảnh")
    if errors:
        print(f"  ✗ Lỗi           : {len(errors):>6} ảnh")
        err_path = db_path.replace(".db", "_errors.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            for fn, err in errors:
                f.write(f"{fn}\t{err}\n")
        print(f"    → Chi tiết lỗi : {err_path}")
    print(f"  → Database      : {os.path.abspath(db_path)}")
    print(f"{'═'*62}")

    db_stats(conn)
    conn.close()


# ════════════════════════════════════════════════════════════════════
# PHẦN 6 — CLI
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Trích xuất đặc trưng lá cây → SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python leaf_extract.py --data leaves/ --db leaf.db
  python leaf_extract.py --data leaves/ --db leaf.db --rebuild
  python leaf_extract.py --test 1027.jpg
  python leaf_extract.py --stats leaf.db
        """)

    ap.add_argument("--data",    help="Thư mục chứa ảnh lá (tên file: 1027.jpg ...)")
    ap.add_argument("--db",      default="leaf.db",
                    help="File SQLite output (mặc định: leaf.db)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Xoá DB cũ, extract lại từ đầu")
    ap.add_argument("--test",    metavar="IMAGE",
                    help="Extract 1 ảnh thử, in vector ra terminal")
    ap.add_argument("--stats",   metavar="DB",
                    help="In thống kê của file .db")

    args = ap.parse_args()

    # ── Test 1 ảnh ──────────────────────────────────────────────
    if args.test:
        print(f"\nTest extract: {args.test}")
        try:
            feats = extract_features(args.test)
            print(f"\n  {'─'*54}")
            total = 0
            for k, v in feats.items():
                print(f"  {k:<6} | {len(v):>3} chiều | "
                      f"min={v.min():.4f}  max={v.max():.4f}  "
                      f"mean={v.mean():.4f}")
                total += len(v)
            print(f"  {'─'*54}")
            print(f"  {'TỔNG':<6} | {total:>3} chiều\n")
        except Exception as e:
            print(f"  [LỖI] {e}")
        return

    # ── Stats DB ────────────────────────────────────────────────
    if args.stats:
        if not os.path.exists(args.stats):
            print(f"[LỖI] Không tìm thấy: {args.stats}")
        else:
            conn = sqlite3.connect(args.stats)
            db_stats(conn)
            conn.close()
        return

    # ── Build database ──────────────────────────────────────────
    if not args.data:
        ap.print_help()
        print("\n[LỖI] Cần truyền --data <thư_mục>  hoặc  --test <ảnh>")
        sys.exit(1)

    build_database(
        data_dir = args.data,
        db_path  = args.db,
        rebuild  = args.rebuild,
    )


if __name__ == "__main__":
    main()