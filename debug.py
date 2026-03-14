"""
leaf_feature_extraction.py
==========================
Phiên bản TÁCH BIỆT từng nhóm đặc trưng — không dùng fused vector.

Ý tưởng:
  Mỗi ảnh lá được lưu vào DB dưới dạng 4 vector riêng biệt:
    - efd   : hình dạng biên  — (harmonics-1)*4 chiều, mặc định 56 chiều
    - glcm  : texture bề mặt  — 8 chiều
    - color : màu sắc HSV     — 9 chiều
    - vein  : kiểu gân lá     — 9 chiều

  Khi tìm Top-5: tính cosine similarity từng nhóm riêng,
  rồi kết hợp bằng trọng số W_* LÚC QUERY (không phải lúc lưu).

Lợi ích so với fused vector:
  ✓ Thay đổi tiêu chí tìm kiếm (W_*) mà KHÔNG cần extract lại DB
  ✓ Có thể xem điểm tương đồng từng nhóm riêng để debug
  ✓ Thêm nhóm đặc trưng mới mà không làm hỏng vector cũ
  ✗ Tốn RAM hơn khi load DB (4 vector thay vì 1)
  ✗ Tính similarity chậm hơn một chút (4 lần dot product thay vì 1)

CHANGELOG:
  [BUG1] debug_efd — scale dùng trung bình 2 trục → méo hình
         FIX: tính sx, sy riêng biệt cho từng trục
  [BUG2] preprocess_leaf — squeeze() mất chiều nếu contour có 1 điểm
         FIX: thêm reshape(-1, 2) sau squeeze()
  [BUG3] extract_efd / debug_efd — không kiểm tra ndim trước khi dùng
         FIX: thêm guard ndim == 1 → reshape(-1, 2)
  [BUG4] _fix_contour_orientation — kiểm tra dấu diện tích ngược
         FIX: trong tọa độ ảnh (Y-down), CW → area > 0; pyefd cần CCW
              → đảo khi area > 0 (không phải < 0)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from pyefd import elliptic_fourier_descriptors
from scipy.stats import circmean
from scipy.interpolate import interp1d


# ════════════════════════════════════════════════════════════════════
# TRỌNG SỐ — chỉnh tại đây BẤT KỲ LÚC NÀO mà không cần extract lại
# ════════════════════════════════════════════════════════════════════
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
# HELPER — lưu ảnh / histogram
# ════════════════════════════════════════════════════════════════════

def _save_step(title, image, filename, cmap='gray', figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    if len(image.shape) == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  check {filename}")


def _save_hist(title, data_list, labels, colors, filename,
               xlabel='Gia tri pixel', figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for data, label, color in zip(data_list, labels, colors):
        ax.hist(data, bins=50, alpha=0.6, color=color, label=label)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('So luong pixel')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  check {filename}")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 1 — TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════════════

def _shoelace_signed_area(contour):
    """
    Diện tích có dấu (shoelace). Y-down: CW > 0, CCW < 0.
    """
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _remove_self_intersections(contour):
    """
    Loại bỏ self-intersection bằng cách smooth contour qua mask mới.
    
    Cách hoạt động:
      1. Vẽ contour lên mask tạm
      2. Tìm lại contour từ mask → tự động loại bỏ self-intersection
      3. Trả về contour sạch lớn nhất
    
    Đây là cách đáng tin cậy nhất vì dùng chính OpenCV fill+refind.
    """
    pts = contour.astype(np.int32)
    
    # Tìm bounding box để tạo mask nhỏ vừa đủ
    x_min, y_min = pts[:, 0].min() - 5, pts[:, 1].min() - 5
    x_max, y_max = pts[:, 0].max() + 5, pts[:, 1].max() + 5
    w = int(x_max - x_min) + 1
    h = int(y_max - y_min) + 1
    
    # Dịch contour về gốc tọa độ tạm
    pts_local = pts.copy()
    pts_local[:, 0] -= x_min
    pts_local[:, 1] -= y_min
    
    # Vẽ filled contour → loại bỏ mọi self-intersection
    mask_tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_tmp, [pts_local], -1, 255, -1)
    
    # Tìm lại contour sạch
    contours_new, _ = cv2.findContours(
        mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_new:
        return contour  # fallback
    
    clean = max(contours_new, key=cv2.contourArea).squeeze().reshape(-1, 2)
    
    # Dịch lại về tọa độ gốc
    clean = clean.astype(np.float64)
    clean[:, 0] += x_min
    clean[:, 1] += y_min
    
    return clean


def _fix_contour_orientation(contour):
    """
    Chuẩn hóa contour:
      1. Loại bỏ self-intersection (fill+refind)
      2. Đảm bảo chiều CCW (shoelace < 0)
      3. Dời điểm bắt đầu về x nhỏ nhất
    """
    contour = np.asarray(contour, dtype=np.float64)
    
    # Bước 1: loại bỏ self-intersection
    contour = _remove_self_intersections(contour)
    
    # Bước 2: sửa chiều
    signed_area = _shoelace_signed_area(contour)
    if signed_area > 0:
        contour = contour[::-1].copy()
    
    # Bước 3: dời điểm bắt đầu về x nhỏ nhất
    start_idx = int(np.argmin(contour[:, 0]))
    contour   = np.roll(contour, -start_idx, axis=0)
    
    return contour


def preprocess_leaf(image_path, debug=False):
    """
    Đọc ảnh, tách lá khỏi nền trắng.

    Returns:
      img         — ảnh gốc BGR
      contour_raw — contour đã chuẩn hóa, shape (N, 2) float64
      gray        — ảnh xám
      mask        — binary mask vùng lá
      leaf_area   — số pixel vùng lá
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # CHAIN_APPROX_NONE: giữ mọi điểm, tránh self-intersection
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Khong tim thay contour la")

    raw         = max(contours, key=cv2.contourArea).squeeze()
    raw         = raw.reshape(-1, 2)                   # [BUG2 FIX]
    contour_raw = _fix_contour_orientation(raw)        # [BUG4 FIX]

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour_raw.astype(np.int32)], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    leaf_area = int(np.sum(mask > 0))

    if debug:
        cv2.imwrite("debug_mask.jpg", mask)
        cv2.imwrite("debug_contour.jpg",
                    cv2.drawContours(img.copy(),
                                     [contour_raw.astype(np.int32)],
                                     -1, (0, 255, 0), 2))
        print(f"[Preprocess] contour_pts={len(contour_raw)}, "
              f"leaf_area={leaf_area}")

    return img, contour_raw, gray, mask, leaf_area


def debug_preprocess(image_path, out_prefix="step1"):
    """Lưu ảnh từng bước tiền xử lý — 7 ảnh."""
    print("\n[DEBUG] Buoc 1 — Tien xu ly")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    _save_step("Buoc 0: Anh goc", img, f"{out_prefix}_00_original.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_step("Buoc 1: Chuyen xam", gray, f"{out_prefix}_01_gray.png")

    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _save_step(
        f"Buoc 2: Otsu threshold (nguong={thresh_val:.0f})\n"
        "nen trang->0  la->255",
        binary, f"{out_prefix}_02_otsu_binary.png")

    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _save_step("Buoc 3: MORPH_OPEN — xoa nhieu nho",
               opened, f"{out_prefix}_03_morph_open.png")

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    _save_step("Buoc 4: MORPH_CLOSE — lap lo hong",
               closed, f"{out_prefix}_04_morph_close.png")

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw         = max(contours, key=cv2.contourArea).squeeze().reshape(-1, 2)
    contour_raw = _fix_contour_orientation(raw)

    img_c = cv2.drawContours(
        img.copy(), [contour_raw.astype(np.int32)], -1, (0, 255, 0), 2)
    _save_step(
        f"Buoc 5: Contour lon nhat ({len(contour_raw)} diem)\n"
        "vien xanh = duong bien la",
        img_c, f"{out_prefix}_05_contour.png")

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour_raw.astype(np.int32)], -1, 255, -1)
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    leaf_area = int(np.sum(mask > 0))
    _save_step(
        f"Buoc 6: Mask la (trang = vung la)\nleaf_area = {leaf_area:,} pixel",
        mask, f"{out_prefix}_06_mask.png")

    print(f"  -> 7 anh luu voi prefix '{out_prefix}_'")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2A — EFD: hình dạng biên lá
# ════════════════════════════════════════════════════════════════════

def _resample_contour(contour, n_points=300):
    """Resample contour về n_points điểm cách đều theo arc-length."""
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) < 4:
        return contour
    diffs   = np.diff(contour, axis=0)
    dists   = np.sqrt((diffs ** 2).sum(axis=1))
    cumdist = np.concatenate([[0.0], np.cumsum(dists)])
    total   = cumdist[-1]
    if total < 1e-6:
        return contour
    t_uni = np.linspace(0, total, n_points, endpoint=False)
    fx    = interp1d(cumdist, contour[:, 0], kind='linear')
    fy    = interp1d(cumdist, contour[:, 1], kind='linear')
    return np.stack([fx(t_uni), fy(t_uni)], axis=1)


def extract_efd(contour, harmonics=20, n_resample=600):
    """
    Trích xuất vector hình dạng biên lá bằng EFD.
    Returns: np.ndarray shape ((harmonics-1)*4,), dtype float32
    """
    out_dim = (harmonics - 1) * 4

    if contour is None or len(contour) < 4:
        return np.zeros(out_dim, dtype=np.float32)

    # [BUG3 FIX] đảm bảo shape (N, 2)
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return np.zeros(out_dim, dtype=np.float32)

    contour = _resample_contour(contour, n_resample)
    if len(contour) < 2 * harmonics:
        return np.zeros(out_dim, dtype=np.float32)

    # normalize=False: giữ nguyên hệ tọa độ, tự normalize bên dưới
    # Tránh pyefd normalize xoay sai pha khi lá có hình dạng phức tạp
    coeffs = elliptic_fourier_descriptors(
        contour, order=harmonics + 1, normalize=False)

    # Normalize thủ công: chia tất cả coefficients cho biên độ bậc 1
    # Biên độ bậc 1 = sqrt(a1^2 + b1^2 + c1^2 + d1^2)
    a1, b1, c1, d1 = coeffs[1]
    amp1 = np.sqrt(a1**2 + b1**2 + c1**2 + d1**2)
    if amp1 > 1e-10:
        coeffs = coeffs / amp1

    efd_vec = coeffs[1:harmonics + 1, :].flatten()
    return efd_vec.astype(np.float32)


def debug_efd(contour, harmonics=25, n_resample=600, out_prefix="step2a"):
    """Lưu ảnh từng bước EFD — 8 ảnh."""
    print("\n[DEBUG] Buoc 2A — EFD hinh dang bien")

    # [BUG3 FIX]
    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)

    # 00 — contour gốc
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=1.5)
    ax.plot(contour[0, 0], contour[0, 1], 'ro', markersize=8,
            label='diem bat dau')
    ax.set_title(f"Buoc 0: Contour goc ({len(contour)} diem)",
                 fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_00_contour_raw.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_00_contour_raw.png")

    # 01 — sau resample
    contour_rs = _resample_contour(contour, n_resample)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(contour_rs[:, 0], contour_rs[:, 1], 'g-', linewidth=1.5)
    ax.scatter(contour_rs[::30, 0], contour_rs[::30, 1],
               c='red', s=25, zorder=5, label='moi 30 diem')
    ax.set_title(f"Buoc 1: Sau resample ({n_resample} diem deu)",
                 fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_01_contour_resampled.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_01_contour_resampled.png")

    coeffs = elliptic_fourier_descriptors(
        contour_rs, order=harmonics + 1, normalize=False)
    # Manual normalize: chia cho biên độ bậc 1
    a1, b1, c1, d1 = coeffs[1]
    amp1 = np.sqrt(a1**2 + b1**2 + c1**2 + d1**2)
    if amp1 > 1e-10:
        coeffs = coeffs / amp1
    n_avail = len(coeffs) - 1

    T = 1.0
    t = np.linspace(0, T, n_resample, endpoint=False)

    def reconstruct(up_to_n):
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        for n in range(1, min(up_to_n + 1, len(coeffs))):
            an, bn, cn, dn = coeffs[n]
            x += an * np.cos(2*n*np.pi*t/T) + bn * np.sin(2*n*np.pi*t/T)
            y += cn * np.cos(2*n*np.pi*t/T) + dn * np.sin(2*n*np.pi*t/T)
        return x, y

    cx = contour_rs[:, 0]
    cy = contour_rs[:, 1]

    # 02–06 — tái tạo từng bậc
    for i, n in enumerate([1, 3, 5, 10, harmonics]):
        n_safe = min(n, n_avail)
        x, y   = reconstruct(n_safe)

        # [BUG1 FIX] scale từng trục riêng biệt
        sx       = (cx.max() - cx.min()) / (x.max() - x.min() + 1e-9)
        sy       = (cy.max() - cy.min()) / (y.max() - y.min() + 1e-9)
        cx_ctr   = (cx.max() + cx.min()) / 2
        cy_ctr   = (cy.max() + cy.min()) / 2
        x_scaled = (x - (x.max() + x.min()) / 2) * sx + cx_ctr
        y_scaled = (y - (y.max() + y.min()) / 2) * sy + cy_ctr

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(x, y, 'b-', linewidth=2)
        axes[0].set_title(f'Tai tao bac 1 -> {n_safe}\n(EFD normalize space)',
                          fontweight='bold')
        axes[0].set_aspect('equal'); axes[0].invert_yaxis()
        axes[0].grid(alpha=0.3)

        axes[1].plot(cx, cy, 'g--', linewidth=1.5, alpha=0.7,
                     label='contour goc (pixel)')
        axes[1].plot(x_scaled, y_scaled, 'b-', linewidth=2,
                     label=f'tai tao n={n_safe} (da scale)')
        axes[1].set_title('So sanh voi contour goc\n(cung pixel space)',
                          fontweight='bold')
        axes[1].set_aspect('equal'); axes[1].invert_yaxis()
        axes[1].legend(); axes[1].grid(alpha=0.3)

        fname = f"{out_prefix}_{i+2:02d}_reconstruct_n{n_safe}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  check {fname}")

    # 07 — biên độ
    amplitudes = [np.sqrt(sum(c**2 for c in coeffs[n]))
                  for n in range(1, len(coeffs))]
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_colors = ['#e74c3c' if i < 3 else ('#e67e22' if i < 8 else '#3498db')
                  for i in range(len(amplitudes))]
    ax.bar(range(1, len(amplitudes)+1), amplitudes,
           color=bar_colors, edgecolor='white', alpha=0.85)
    ax.set_xlabel('Bac hai (n)', fontsize=11)
    ax.set_ylabel('Bien do sqrt(a2+b2+c2+d2)', fontsize=11)
    ax.set_title('Bien do tung bac hai EFD\n'
                 'Do=hinh dang chinh  Cam=rang cua  Xanh=chi tiet nho',
                 fontweight='bold')
    ax.set_xticks(range(1, len(amplitudes)+1))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_07_amplitudes.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_07_amplitudes.png")
    print(f"  -> 8 anh luu voi prefix '{out_prefix}_'")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2B — GLCM: texture bề mặt lá
# ════════════════════════════════════════════════════════════════════

def extract_glcm(gray_img, mask):
    """
    Texture qua GLCM.
    Returns: np.ndarray shape (8,), dtype float32
    """
    masked  = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    masked  = cv2.resize(masked, (256, 256))
    levels  = 32
    masked  = (masked // (256 // levels)).astype(np.uint8)
    glcm    = graycomatrix(
        masked, distances=[3, 5],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels, symmetric=True, normed=True)
    vec = np.array([
        graycoprops(glcm, p).mean(axis=1)
        for p in ['contrast', 'homogeneity', 'energy', 'correlation']
    ]).flatten()
    return vec.astype(np.float32)


def debug_glcm(gray_img, mask, out_prefix="step2b"):
    """Lưu ảnh từng bước GLCM — 6 ảnh."""
    print("\n[DEBUG] Buoc 2B — GLCM texture")

    masked = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    _save_step("Buoc 0: Anh xam da cat vung la",
               masked, f"{out_prefix}_00_masked_gray.png")

    resized = cv2.resize(masked, (256, 256))
    _save_step("Buoc 1: Resize ve 256x256",
               resized, f"{out_prefix}_01_resized.png")

    levels    = 32
    quantized = (resized // (256 // levels)).astype(np.uint8)
    _save_step(f"Buoc 2: Giam xuong {levels} muc xam",
               (quantized * (255 // levels)).astype(np.uint8),
               f"{out_prefix}_02_quantized.png")

    px = quantized.flatten(); px = px[px > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(levels), np.bincount(px, minlength=levels),
           color='steelblue', edgecolor='white', width=0.8)
    ax.set_xlabel('Muc xam (0-31)'); ax.set_ylabel('So pixel')
    ax.set_title('Buoc 3: Histogram 32 muc xam', fontweight='bold')
    ax.set_xticks(range(0, levels, 4)); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_03_hist_quantized.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_03_hist_quantized.png")

    glcm = graycomatrix(quantized, distances=[3, 5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(glcm[:, :, 0, 0], cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_title('Buoc 4: Ma tran GLCM (d=3, angle=0)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_04_glcm_matrix.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_04_glcm_matrix.png")

    props      = ['contrast', 'homogeneity', 'energy', 'correlation']
    colors_bar = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for idx, (prop, color) in enumerate(zip(props, colors_bar)):
        vals = graycoprops(glcm, prop).mean(axis=1)
        axes[idx].bar(['d=3', 'd=5'], vals, color=color,
                      alpha=0.85, edgecolor='white')
        axes[idx].set_title(prop.capitalize(), fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        for j, v in enumerate(vals):
            axes[idx].text(j, v + abs(v)*0.02, f'{v:.4f}',
                           ha='center', va='bottom', fontsize=9)
    plt.suptitle('Buoc 5: 4 thuoc tinh GLCM',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_05_properties.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_05_properties.png")

    vec = extract_glcm(gray_img, mask)
    print(f"  -> GLCM vector ({len(vec)} chieu): {vec.round(4)}")
    print(f"  -> 6 anh luu voi prefix '{out_prefix}_'")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2C — Color Moments: màu sắc lá
# ════════════════════════════════════════════════════════════════════

def extract_color_moments(img, mask):
    """
    Color moments HSV.
    Returns: np.ndarray shape (9,) — [H_mean,H_std,H_skew, S_..., V_...]
    """
    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    moments    = []
    for i, ch in enumerate(cv2.split(masked_hsv)):
        px = ch[mask > 0].astype(np.float64)
        if len(px) == 0:
            moments.extend([0., 0., 0.]); continue
        mean = circmean(px, high=179, low=0) if i == 0 else np.mean(px)
        std  = np.std(px)
        skew = float(np.cbrt(((px - mean)**3).mean())) if std > 0 else 0.
        moments.extend([mean, std, skew])
    return np.array(moments, dtype=np.float32)


def debug_color_moments(img, mask, out_prefix="step2c"):
    """Lưu ảnh từng bước Color Moments — 8 ảnh."""
    print("\n[DEBUG] Buoc 2C — Color Moments HSV")

    _save_step("Buoc 0: Anh goc", img, f"{out_prefix}_00_original.png")

    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    h, s, v    = cv2.split(masked_hsv)

    _save_step("Buoc 1: Kenh H (Hue)",
               cv2.applyColorMap((h*(255//179)).astype(np.uint8),
                                  cv2.COLORMAP_HSV),
               f"{out_prefix}_01_hsv_hue.png")
    _save_step("Buoc 2: Kenh S (Saturation)",
               s, f"{out_prefix}_02_hsv_sat.png")
    _save_step("Buoc 3: Kenh V (Brightness)",
               v, f"{out_prefix}_03_hsv_val.png")

    px_h = h[mask > 0].astype(np.float64)
    px_s = s[mask > 0].astype(np.float64)
    px_v = v[mask > 0].astype(np.float64)

    mean_h = circmean(px_h, high=179, low=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(px_h, bins=36, range=(0, 179), color='coral',
            alpha=0.8, edgecolor='white')
    ax.axvline(mean_h, color='red', lw=2, ls='--',
               label=f'circmean={mean_h:.1f}')
    ax.axvline(np.mean(px_h), color='blue', lw=2, ls=':',
               label=f'mean thuong={np.mean(px_h):.1f}')
    ax.set_xlabel('H (0-179)'); ax.set_ylabel('So pixel')
    ax.set_title('Buoc 4: Histogram kenh H', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_04_hist_H.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_04_hist_H.png")

    for idx, (name, px, color) in enumerate([
        ('S', px_s, 'dodgerblue'), ('V', px_v, 'mediumseagreen')
    ], start=5):
        mean_v = np.mean(px)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(px, bins=32, range=(0, 255), color=color,
                alpha=0.8, edgecolor='white')
        ax.axvline(mean_v, color='red', lw=2, ls='--',
                   label=f'mean={mean_v:.1f}  std={np.std(px):.1f}')
        ax.set_xlabel(f'{name} (0-255)'); ax.set_ylabel('So pixel')
        ax.set_title(f'Buoc {idx}: Histogram kenh {name}',
                     fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_0{idx}_hist_{name}.png", dpi=120,
                    bbox_inches='tight')
        plt.close()
        print(f"  check {out_prefix}_0{idx}_hist_{name}.png")

    vec    = extract_color_moments(img, mask)
    labels = ['H_mean','H_std','H_skew',
              'S_mean','S_std','S_skew',
              'V_mean','V_std','V_skew']
    clrs   = ['#e74c3c']*3 + ['#3498db']*3 + ['#2ecc71']*3
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, vec, color=clrs, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vec):
        off = abs(val)*0.03 + 0.5
        ax.text(bar.get_x()+bar.get_width()/2,
                val+(off if val >= 0 else -off-2),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Buoc 7: Vector Color Moments (9 chieu)',
                 fontweight='bold')
    ax.set_ylabel('Gia tri'); ax.axhline(0, color='black', lw=0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_07_moments_summary.png", dpi=120,
                bbox_inches='tight')
    plt.close()
    print(f"  check {out_prefix}_07_moments_summary.png")
    print(f"  -> Color vector ({len(vec)} chieu): {vec.round(2)}")
    print(f"  -> 8 anh luu voi prefix '{out_prefix}_'")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2D — Vein Features: kiểu gân lá
# ════════════════════════════════════════════════════════════════════

def extract_vein_features(img, mask, leaf_area):
    """
    Dac trung gan la.
    Returns: np.ndarray shape (9,) — [density, angle_hist x8]
    """
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)

    canny_e   = cv2.Canny(smoothed, 20, 80)
    sobelx    = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    sobely    = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = cv2.normalize(
        np.hypot(sobelx, sobely), None, 0, 255,
        cv2.NORM_MINMAX).astype(np.uint8)

    edges     = cv2.addWeighted(sobel_mag, 0.6, canny_e, 0.4, 0)
    vein_mask = cv2.bitwise_and(edges, edges, mask=mask)

    k3        = np.ones((3, 3), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, k3, iterations=1)
    vein_mask = cv2.dilate(vein_mask, k3, iterations=1)

    density   = float(np.sum(vein_mask > 0) / leaf_area) if leaf_area > 0 else 0.

    angle_map = np.arctan2(sobely, sobelx) * 180. / np.pi % 180.
    if (vein_mask > 0).any():
        hist, _ = np.histogram(angle_map[vein_mask > 0],
                               bins=8, range=(0, 180), density=True)
    else:
        hist = np.zeros(8)

    return np.concatenate([[density], hist]).astype(np.float32)


def debug_vein_features(img, mask, leaf_area, out_prefix="step2d"):
    """Lưu ảnh từng bước Vein Features — 12 ảnh."""
    print("\n[DEBUG] Buoc 2D — Vein Features gan la")

    _save_step("Buoc 0: Anh goc", img, f"{out_prefix}_00_original.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save_step("Buoc 1: Chuyen xam", gray, f"{out_prefix}_01_gray.png")

    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _save_step("Buoc 2: CLAHE — tang tuong phan cuc bo",
               enhanced, f"{out_prefix}_02_clahe.png")

    _save_hist(
        "Buoc 3: Histogram truoc/sau CLAHE",
        [gray.flatten(), enhanced.flatten()],
        ['truoc', 'sau CLAHE'], ['steelblue', 'coral'],
        f"{out_prefix}_03_hist_clahe.png")

    smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _save_step("Buoc 4: Bilateral filter",
               smoothed, f"{out_prefix}_04_bilateral.png")

    _save_step("Buoc 5: Diff (CLAHE - bilateral) = nhieu bi loai",
               cv2.absdiff(enhanced, smoothed),
               f"{out_prefix}_05_diff_noise.png")

    canny_e = cv2.Canny(smoothed, 20, 80)
    _save_step("Buoc 6: Canny edges", canny_e, f"{out_prefix}_06_canny.png")

    sobelx    = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    sobely    = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = cv2.normalize(
        np.hypot(sobelx, sobely), None, 0, 255,
        cv2.NORM_MINMAX).astype(np.uint8)
    _save_step("Buoc 7: Sobel magnitude",
               sobel_mag, f"{out_prefix}_07_sobel_mag.png")

    edges = cv2.addWeighted(sobel_mag, 0.6, canny_e, 0.4, 0)
    _save_step("Buoc 8: Ket hop Sobel x0.6 + Canny x0.4",
               edges, f"{out_prefix}_08_combined_edges.png")

    vein_mask = cv2.bitwise_and(edges, edges, mask=mask)
    _save_step("Buoc 9: Cat vung la",
               vein_mask, f"{out_prefix}_09_vein_masked.png")

    k3         = np.ones((3, 3), np.uint8)
    vein_clean = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, k3, iterations=1)
    vein_clean = cv2.dilate(vein_clean, k3, iterations=1)
    vein_px    = int(np.sum(vein_clean > 0))
    density    = float(vein_px / leaf_area) if leaf_area > 0 else 0.
    _save_step(
        f"Buoc 10: Sau morphology\n"
        f"density={density:.4f}  ({vein_px:,}/{leaf_area:,} px)",
        vein_clean, f"{out_prefix}_10_vein_clean.png")

    angle_map = np.arctan2(sobely, sobelx) * 180. / np.pi % 180.
    if (vein_clean > 0).any():
        angles       = angle_map[vein_clean > 0]
        hist, edges2 = np.histogram(angles, bins=8, range=(0,180), density=True)
        bin_ctrs     = (edges2[:-1] + edges2[1:]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].bar(bin_ctrs, hist, width=20,
                    color='steelblue', edgecolor='white', alpha=0.85)
        peak = bin_ctrs[np.argmax(hist)]
        axes[0].axvline(peak, color='red', ls='--', lw=1.5,
                        label=f'dinh {peak:.0f} do')
        axes[0].set_xlabel('Goc (do)'); axes[0].set_ylabel('Mat do')
        axes[0].set_title('Phan bo goc gan la', fontweight='bold')
        axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

        angle_vis   = (angle_map / 180. * 255).astype(np.uint8)
        angle_color = cv2.applyColorMap(angle_vis, cv2.COLORMAP_HSV)
        vm3         = cv2.cvtColor(
            (vein_clean > 0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        axes[1].imshow(cv2.cvtColor(
            cv2.bitwise_and(angle_color, vm3), cv2.COLOR_BGR2RGB))
        axes[1].set_title('Mau = huong gan', fontweight='bold')
        axes[1].axis('off')

        plt.suptitle(f'Buoc 11: Goc gan  density={density:.4f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_11_angle_histogram.png", dpi=120,
                    bbox_inches='tight')
        plt.close()
        print(f"  check {out_prefix}_11_angle_histogram.png")

        vec = np.concatenate([[density], hist.astype(np.float32)])
        print(f"  -> Vein vector ({len(vec)} chieu): {vec.round(4)}")
    else:
        print("  Khong phat hien gan nao")

    print(f"  -> 12 anh luu voi prefix '{out_prefix}_'")


# ════════════════════════════════════════════════════════════════════
# HÀM CHÍNH
# ════════════════════════════════════════════════════════════════════

def extract_features(image_path, harmonics=15, n_resample=300, debug=False):
    """
    Trích xuất tất cả đặc trưng, trả về dict 4 vector.

    Returns:
        {
          "efd"   : list[float]  — (harmonics-1)*4 chiều
          "glcm"  : list[float]  — 8 chiều
          "color" : list[float]  — 9 chiều
          "vein"  : list[float]  — 9 chiều
        }
    """
    img, contour, gray, mask, leaf_area = preprocess_leaf(
        image_path, debug=debug)

    features = {
        "efd"  : extract_efd(contour, harmonics, n_resample).tolist(),
        "glcm" : extract_glcm(gray, mask).tolist(),
        "color": extract_color_moments(img, mask).tolist(),
        "vein" : extract_vein_features(img, mask, leaf_area).tolist(),
    }

    if debug:
        print(f"\n{'='*45}")
        for k, v in features.items():
            print(f"{k:6s}: {len(v):2d} chieu | "
                  f"min={min(v):.4f}  max={max(v):.4f}")
        print(f"{'='*45}\n")

    return features


# ════════════════════════════════════════════════════════════════════
# SIMILARITY & SEARCH
# ════════════════════════════════════════════════════════════════════

def _cosine(a, b):
    a    = np.asarray(a, dtype=np.float64)
    b    = np.asarray(b, dtype=np.float64)
    dnom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / dnom) if dnom > 1e-10 else 0.0


def compute_similarity(query_features, db_features):
    s_efd   = _cosine(query_features["efd"],   db_features["efd"])
    s_glcm  = _cosine(query_features["glcm"],  db_features["glcm"])
    s_color = _cosine(query_features["color"], db_features["color"])
    s_vein  = _cosine(query_features["vein"],  db_features["vein"])
    total_w = W_EFD + W_GLCM + W_COLOR + W_VEIN
    weighted = (s_efd*W_EFD + s_glcm*W_GLCM +
                s_color*W_COLOR + s_vein*W_VEIN) / total_w
    return {
        "efd"     : round(s_efd,   4),
        "glcm"    : round(s_glcm,  4),
        "color"   : round(s_color, 4),
        "vein"    : round(s_vein,  4),
        "weighted": round(weighted, 4),
    }


def retrieve_top5(query_features, database, sort_by="weighted"):
    results = []
    for entry in database:
        db_feat = {k: entry[k] for k in ("efd","glcm","color","vein")}
        scores  = compute_similarity(query_features, db_feat)
        results.append({"id": entry["id"],
                         "label": entry.get("label",""),
                         "scores": scores})
    results.sort(key=lambda x: x["scores"][sort_by], reverse=True)
    return results[:5]


# ════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else "1004.jpg"

    print("=" * 55)
    print("  DEBUG TOAN BO PIPELINE")
    print("=" * 55)

    img, contour, gray, mask, leaf_area = preprocess_leaf(path, debug=True)

    debug_preprocess(path,                    out_prefix="step1")
    debug_efd(contour, harmonics=15,          out_prefix="step2a")
    debug_glcm(gray, mask,                    out_prefix="step2b")
    debug_color_moments(img, mask,            out_prefix="step2c")
    debug_vein_features(img, mask, leaf_area, out_prefix="step2d")

    print("\n" + "=" * 55)
    print("  EXTRACT & DEMO QUERY")
    print("=" * 55)

    features = extract_features(path, harmonics=15, n_resample=300, debug=True)

    db_entry = {
        "id": "leaf_001", "label": "La xoai",
        **features,
    }

    print("DB entry (3 phan tu dau moi vector):")
    print(json.dumps(
        {k: (v[:3] if isinstance(v, list) else v)
         for k, v in db_entry.items()},
        ensure_ascii=False, indent=2))

    print("\nTop-5 theo hinh dang:")
    for r in retrieve_top5(features, [db_entry], sort_by="efd"):
        s = r["scores"]
        print(f"  {r['label']:15s} efd={s['efd']:.4f} "
              f"glcm={s['glcm']:.4f} color={s['color']:.4f} "
              f"vein={s['vein']:.4f} -> weighted={s['weighted']:.4f}")

    print("\nXong! Kiem tra:")
    print("  step1_*.png   — tien xu ly  (7 anh)")
    print("  step2a_*.png  — EFD          (8 anh)")
    print("  step2b_*.png  — GLCM         (6 anh)")
    print("  step2c_*.png  — Color        (8 anh)")
    print("  step2d_*.png  — Vein         (12 anh)")
    print("  Tong cong: 41 anh")