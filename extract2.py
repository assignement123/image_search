"""
leaf_feature_extraction.py
==========================
Phiên bản TÁCH BIỆT từng nhóm đặc trưng — không dùng fused vector.

Ý tưởng:
  Mỗi ảnh lá được lưu vào DB dưới dạng 4 vector riêng biệt:
    - efd   : hình dạng biên (44–76 chiều tùy harmonics)
    - glcm  : texture bề mặt (8 hoặc 12 chiều)
    - color : màu sắc HSV (9 chiều)
    - vein  : kiểu gân lá (9 chiều)

  Khi tìm Top-5: tính cosine similarity từng nhóm riêng,
  rồi kết hợp bằng trọng số W_* LÚC QUERY (không phải lúc lưu).

Lợi ích so với fused vector:
  ✓ Thay đổi tiêu chí tìm kiếm (W_*) mà KHÔNG cần extract lại DB
  ✓ Có thể xem điểm tương đồng từng nhóm riêng để debug
  ✓ Thêm nhóm đặc trưng mới mà không làm hỏng vector cũ
  ✗ Tốn RAM hơn khi load DB (4 vector thay vì 1)
  ✗ Tính similarity chậm hơn một chút (4 lần dot product thay vì 1)
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pyefd import elliptic_fourier_descriptors
from scipy.stats import circmean
from scipy.interpolate import interp1d


# ════════════════════════════════════════════════════════════════════
# TRỌNG SỐ — chỉnh tại đây BẤT KỲ LÚC NÀO mà không cần extract lại
# ════════════════════════════════════════════════════════════════════
#
# Vì vector được lưu riêng, W_* chỉ dùng khi query → đổi tự do.
#
# ┌─────────────────────┬───────┬───────┬─────────┬────────┐
# │ Tiêu chí tìm kiếm   │ W_EFD │ W_GLCM│ W_COLOR │ W_VEIN │
# ├─────────────────────┼───────┼───────┼─────────┼────────┤
# │ Hình dạng biên      │  3.0  │  0.3  │   0.0   │  0.5   │
# │ Màu sắc             │  0.5  │  0.3  │   3.0   │  0.2   │
# │ Texture bề mặt      │  0.5  │  3.0  │   0.5   │  0.3   │
# │ Kiểu gân lá         │  0.5  │  0.5  │   0.2   │  3.0   │
# │ Màu + Texture       │  1.5  │  2.0  │   2.0   │  0.5   │
# │ Tổng hợp cân bằng   │  1.5  │  1.0  │   1.2   │  1.0   │
# └─────────────────────┴───────┴───────┴─────────┴────────┘

W_EFD   = 3.0
W_GLCM  = 0.3
W_COLOR = 0.0
W_VEIN  = 0.5


# ════════════════════════════════════════════════════════════════════
# BƯỚC 1 — TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════════════

def preprocess_leaf(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu: tự tìm ngưỡng tách lá (tối) khỏi nền trắng (sáng)
    # THRESH_BINARY_INV: nền trắng → 0, lá → 255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)  # xóa nhiễu nhỏ
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # lấp lỗ hổng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy contour lá")
    contour_raw = max(contours, key=cv2.contourArea).squeeze()
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour_raw.astype(int)], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    leaf_area = int(np.sum(mask > 0))
    if debug:
        cv2.imwrite("debug_mask.jpg", mask)
        cv2.imwrite("debug_contour.jpg",
                    cv2.drawContours(img.copy(), [contour_raw.astype(int)], -1, (0, 255, 0), 2))
        print(f"[Preprocess] contour_pts={len(contour_raw)}, leaf_area={leaf_area}")
    return img, contour_raw, gray, mask, leaf_area


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2A — EFD: hình dạng biên lá
# ════════════════════════════════════════════════════════════════════

def _resample_contour(contour, n_points=300):
    """
    Resample contour về n_points điểm cách đều theo arc-length.
    Đảm bảo EFD nhất quán bất kể contour thô có bao nhiêu điểm.

    n_points:
      200–300 : khuyến nghị — cân bằng chi tiết và tốc độ
      400–500 : bắt chi tiết mép lá mịn hơn, chậm hơn
      100–150 : chỉ giữ hình dạng tổng quát
    """
    if len(contour) < 4:
        return contour
    diffs   = np.diff(contour, axis=0)
    dists   = np.sqrt((diffs ** 2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    total   = cumdist[-1]
    if total < 1e-6:
        return contour
    t_uniform = np.linspace(0, total, n_points, endpoint=False)
    fx = interp1d(cumdist, contour[:, 0], kind='linear')
    fy = interp1d(cumdist, contour[:, 1], kind='linear')
    return np.stack([fx(t_uniform), fy(t_uniform)], axis=1)


def extract_efd(contour, harmonics=15, n_resample=300):
    """
    Trả về vector hình dạng biên lá.

    harmonics:
      8–10  : hình dạng tổng quát (oval/tròn/tim), tính nhanh
      12–15 : + chi tiết thùy và răng cưa lớn  ← khuyến nghị
      16–20 : + răng cưa nhỏ, chỉ dùng khi ảnh rất sắc nét
      >20   : bắt nhiễu ảnh chụp, không nên dùng

    Số chiều output: (harmonics - 1) × 4
      harmonics=12 → 44 chiều
      harmonics=15 → 56 chiều
      harmonics=20 → 76 chiều

    normalize=True: bất biến với xoay, tỉ lệ, điểm bắt đầu contour.
    Bắt buộc True — nếu False thì cùng lá chụp khác góc có similarity ~0.3.
    """
    contour = _resample_contour(contour, n_resample)
    if len(contour) < 20:
        return np.zeros(4 * harmonics, dtype=np.float32)
    coeffs  = elliptic_fourier_descriptors(contour, order=harmonics, normalize=True)
    efd_vec = coeffs[1:, :].flatten()  # bỏ hài bậc 0 (vị trí tâm, không phải hình dạng)
    return efd_vec.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2B — GLCM: texture bề mặt lá
# ════════════════════════════════════════════════════════════════════

def extract_glcm(gray_img, mask):
    """
    Trả về vector texture bề mặt lá.

    distances:
      [1,3,5] : đa tầng — d=1 (tế bào mịn), d=3 (gân phụ), d=5 (gân chính)
                dùng khi texture là tiêu chí chính → 12 chiều
      [3,5]   : bỏ d=1 khi texture chỉ phụ trợ, hoặc ảnh không đồng đều sharpness
                → 8 chiều

    levels:
      16 : GLCM đặc, tính nhanh, ít phân biệt
      32 : cân bằng ← khuyến nghị mặc định
      64 : phân biệt tốt hơn, cần ảnh sắc nét — nhiễu sẽ làm GLCM thưa

    4 thuộc tính trả về (mỗi cái là mean theo 4 hướng × n_distances):
      contrast    : texture thô (cao) vs mịn (thấp)
      homogeneity : đồng đều (cao ~1) vs biến thiên nhiều (thấp)
      energy      : lặp đều đặn (cao) vs ngẫu nhiên (thấp)
      correlation : có cấu trúc hướng rõ (cao ~1) vs hỗn loạn (thấp)

    Số chiều output: len(distances) × 4
      distances=[3,5]   → 8 chiều
      distances=[1,3,5] → 12 chiều
    """
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    masked_gray = cv2.resize(masked_gray, (256, 256))  # chuẩn hóa kích thước

    levels      = 32
    masked_gray = (masked_gray // (256 // levels)).astype(np.uint8)

    glcm = graycomatrix(
        masked_gray,
        distances=[3, 5],                               # đổi thành [1,3,5] để thêm texture mịn
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],       # 4 hướng — lấy mean → bất biến hướng
        levels=levels,
        symmetric=True,
        normed=True                                     # bắt buộc để so sánh được giữa ảnh
    )

    props    = ['contrast', 'homogeneity', 'energy', 'correlation']
    glcm_vec = np.array([
        graycoprops(glcm, prop).mean(axis=1)            # mean theo angles, giữ distances
        for prop in props
    ]).flatten()

    return glcm_vec.astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2C — Color Moments: màu sắc lá
# ════════════════════════════════════════════════════════════════════

def extract_color_moments(img, mask):
    """
    Trả về vector màu sắc lá (9 chiều cố định).

    Không gian màu:
      HSV : tách màu (H) khỏi độ sáng (V) — tốt cho ánh sáng thay đổi vừa
            Kênh H là GÓC → phải dùng circmean (xem bên dưới)
      Lab : bất biến ánh sáng tốt hơn — dùng khi điều kiện chụp thay đổi nhiều
            Đổi: cv2.COLOR_BGR2LAB, bỏ phần if i==0, dùng np.mean cho cả 3 kênh

    3 moments mỗi kênh:
      mean     : màu trung bình
      std      : độ phân tán màu (cao = lá loang lổ, thấp = màu đều)
      skewness : phân phối lệch về tối (âm) hay sáng (dương)

    Số chiều output: 3 kênh × 3 moments = 9 chiều (cố định, bất kể không gian màu)
    [H_mean, H_std, H_skew, S_mean, S_std, S_skew, V_mean, V_std, V_skew]
    """
    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

    moments = []
    for i, channel in enumerate(cv2.split(masked_hsv)):
        pixels = channel[mask > 0].astype(np.float64)
        if len(pixels) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue

        # Hue (i=0) là đại lượng GÓC — PHẢI dùng circular mean
        #   np.mean([5, 175]) = 90 (xanh lá) — SAI
        #   circmean([5, 175], high=179) ≈ 0 (đỏ) — đúng
        #   Bỏ nếu đổi sang Lab color space
        mean     = circmean(pixels, high=179, low=0) if i == 0 else np.mean(pixels)
        std      = np.std(pixels)
        skewness = float(np.cbrt(((pixels - mean) ** 3).mean())) if std > 0 else 0.0
        moments.extend([mean, std, skewness])

    return np.array(moments, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2D — Vein Features: kiểu gân lá
# ════════════════════════════════════════════════════════════════════

def extract_vein_features(img, mask, leaf_area):
    """
    Trả về vector đặc trưng gân lá (9 chiều mặc định).

    Cấu trúc output: [density, hist_bin0, ..., hist_bin{bins-1}]
      density  : tỉ lệ pixel gân / tổng pixel lá
                 0.05–0.15 = ít gân (lá sen, lá chuối)
                 0.15–0.30 = gân trung bình (lá xoài, lá ổi)
                 0.30–0.50 = nhiều gân (lá sả, lá tre)
      histogram: phân bố góc gân → phân biệt kiểu gân:
                 Tập trung ở 80°–100° → gân song song (lúa, sả)
                 Phân bố đều           → gân lưới (ổi, xoài)
                 Đỉnh ở 2–3 hướng      → gân lông chim (bưởi, chanh)

    Tham số quan trọng:
      CLAHE clipLimit  : 3–5 cho lá bình thường, 2–3 cho lá mịn ít gân
      Canny (20, 80)   : giảm threshold2 xuống 60 nếu bỏ sót gân mỏng
      bins histogram   : 8 (22.5°/bin, đủ phân biệt 3 kiểu gân chính)
                         16 nếu cần phân biệt góc gân tinh tế hơn → 17 chiều

    Số chiều output: 1 + bins = 9 chiều (mặc định bins=8)
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    canny_edges = cv2.Canny(smoothed, 20, 80)

    sobelx    = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    sobely    = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.hypot(sobelx, sobely)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges     = cv2.addWeighted(sobel_mag, 0.6, canny_edges, 0.4, 0)
    vein_mask = cv2.bitwise_and(edges, edges, mask=mask)

    kernel_small = np.ones((3, 3), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    vein_mask = cv2.dilate(vein_mask, kernel_small, iterations=1)

    vein_pixels = np.sum(vein_mask > 0)
    density     = float(vein_pixels / leaf_area) if leaf_area > 0 else 0.0

    angle_map = np.arctan2(sobely, sobelx) * 180.0 / np.pi % 180.0
    if (vein_mask > 0).any():
        angles_at_veins = angle_map[vein_mask > 0]
        hist, _ = np.histogram(angles_at_veins, bins=8, range=(0, 180), density=True)
        hist    = hist.astype(np.float32)
    else:
        hist = np.zeros(8, dtype=np.float32)

    return np.concatenate([[density], hist]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# HÀM CHÍNH — trả về dict 4 vector riêng biệt
# ════════════════════════════════════════════════════════════════════

def extract_features(image_path, harmonics=15, n_resample=300, debug=False):
    """
    Extract tất cả đặc trưng, trả về dict 4 vector riêng.
    Lưu dict này vào DB — KHÔNG fuse.

    Returns:
        {
          "efd"   : list[float]  — hình dạng biên, (harmonics-1)*4 chiều
          "glcm"  : list[float]  — texture,  8 hoặc 12 chiều
          "color" : list[float]  — màu sắc,  9 chiều
          "vein"  : list[float]  — gân lá,   9 chiều
        }
    """
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path, debug=debug)

    features = {
        "efd"   : extract_efd(contour, harmonics=harmonics, n_resample=n_resample).tolist(),
        "glcm"  : extract_glcm(gray, mask).tolist(),
        "color" : extract_color_moments(img, mask).tolist(),
        "vein"  : extract_vein_features(img, mask, leaf_area).tolist(),
    }

    if debug:
        print(f"\n{'='*45}")
        for k, v in features.items():
            print(f"{k:6s}: {len(v):2d} chieu | min={min(v):.4f} max={max(v):.4f}")
        print(f"{'='*45}\n")

    return features


# ════════════════════════════════════════════════════════════════════
# SIMILARITY — tính riêng từng nhóm
# ════════════════════════════════════════════════════════════════════

def _cosine(a, b):
    """Cosine similarity giữa 2 vector (= 1 nếu giống hệt, 0 nếu không liên quan)."""
    a, b  = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def compute_similarity(query_features, db_features):
    """
    Tính điểm tương đồng từng nhóm giữa query và 1 ảnh trong DB.

    Returns dict:
        {
          "efd"      : float  — similarity hình dạng [0, 1]
          "glcm"     : float  — similarity texture   [0, 1]
          "color"    : float  — similarity màu sắc   [0, 1]
          "vein"     : float  — similarity gân lá    [0, 1]
          "weighted" : float  — điểm tổng hợp theo W_* hiện tại
        }
    """
    sim_efd   = _cosine(query_features["efd"],   db_features["efd"])
    sim_glcm  = _cosine(query_features["glcm"],  db_features["glcm"])
    sim_color = _cosine(query_features["color"], db_features["color"])
    sim_vein  = _cosine(query_features["vein"],  db_features["vein"])
    total_w = W_EFD + W_GLCM + W_COLOR + W_VEIN
    weighted = (
        sim_efd   * W_EFD  +
        sim_glcm  * W_GLCM +
        sim_color * W_COLOR +
        sim_vein  * W_VEIN
    ) / total_w

    return {
        "efd"      : round(sim_efd,   4),
        "glcm"     : round(sim_glcm,  4),
        "color"    : round(sim_color, 4),
        "vein"     : round(sim_vein,  4),
        "weighted" : round(weighted,  4),
    }


# ════════════════════════════════════════════════════════════════════
# TÌM KIẾM TOP-5
# ════════════════════════════════════════════════════════════════════

def retrieve_top5(query_features, database, sort_by="weighted"):
    """
    Tìm Top-5 lá tương đồng nhất.

    Args:
        query_features : dict từ extract_features()
        database       : list of dict, mỗi phần tử có:
                           {"id": str, "label": str,
                            "efd": list, "glcm": list, "color": list, "vein": list}
        sort_by        : tiêu chí sắp xếp — một trong:
                           "weighted" : tổng hợp theo W_* (mặc định)
                           "efd"      : chỉ theo hình dạng
                           "glcm"     : chỉ theo texture
                           "color"    : chỉ theo màu sắc
                           "vein"     : chỉ theo gân lá

    Returns:
        list of dict — Top-5 kết quả, mỗi phần tử:
            {
              "id"       : str
              "label"    : str
              "scores"   : dict (efd, glcm, color, vein, weighted)
            }
        Sắp xếp giảm dần theo sort_by.

    Ví dụ đổi tiêu chí mà không extract lại:
        # Tìm theo hình dạng
        results = retrieve_top5(q, db, sort_by="efd")

        # Tìm theo màu sắc
        results = retrieve_top5(q, db, sort_by="color")

        # Tìm tổng hợp với trọng số mới — chỉ cần đổi W_* ở đầu file
        W_EFD, W_COLOR = 1.5, 2.0
        results = retrieve_top5(q, db, sort_by="weighted")
    """
    results = []
    for entry in database:
        db_feat = {k: entry[k] for k in ("efd", "glcm", "color", "vein")}
        scores  = compute_similarity(query_features, db_feat)
        results.append({
            "id"     : entry["id"],
            "label"  : entry.get("label", ""),
            "scores" : scores,
        })

    results.sort(key=lambda x: x["scores"][sort_by], reverse=True)
    return results[:5]


# ════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "1060.jpg"
    features = extract_features(path, harmonics=15, n_resample=300, debug=True)
    db_entry = {
        "id"    : "leaf_001",
        "label" : "Lá xoài",
        "efd"   : features["efd"],
        "glcm"  : features["glcm"],
        "color" : features["color"],
        "vein"  : features["vein"],
    }
    print("Cấu trúc DB entry:")
    print(json.dumps({k: (v[:3] if isinstance(v, list) else v)
                      for k, v in db_entry.items()}, ensure_ascii=False, indent=2))
    print("  (mỗi list hiển thị 3 phần tử đầu...)\n")
    fake_db = [db_entry]  
    print("Top-5 theo hình dạng:")
    top5 = retrieve_top5(features, fake_db, sort_by="efd")
    for r in top5:
        print(f"  {r['label']:15s} | shape={r['scores']['efd']:.4f} "
              f"texture={r['scores']['glcm']:.4f} "
              f"color={r['scores']['color']:.4f} "
              f"vein={r['scores']['vein']:.4f} "
              f"→ weighted={r['scores']['weighted']:.4f}")