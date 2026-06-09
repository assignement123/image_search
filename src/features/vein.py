import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════
# GÂN LÁ — MẬT ĐỘ & PHÂN BỐ HƯỚNG GÂN (rotation-invariant)
# ════════════════════════════════════════════════════════════════════

N_VEIN_BINS = 16   # 16 bins × 11.25°/bin — đủ mịn để phân biệt venation patterns
                   # 8 bins (22.5°) quá thô: pinnate ≈ palmate ở nhiều loài
DIM_VEIN = 1 + N_VEIN_BINS  # 17 chiều: [density] + [histogram 16-bin]


def _leaf_main_axis_angle(mask: np.ndarray) -> float:
    """
    Góc trục chính của lá tính bằng image moments (PCA of blob).
    Trả về góc tính theo radian, trong [-π/2, π/2).

    Dùng để normalize vein angle histogram về frame của lá,
    bất biến với hướng đặt lá trên scanner.
    """
    M = cv2.moments(mask.astype(np.float64))
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']
    denom = mu20 - mu02
    if abs(denom) < 1e-6 and abs(mu11) < 1e-6:
        return 0.0  # gần tròn, không xác định hướng
    return 0.5 * np.arctan2(2.0 * mu11, denom)


def extract_vein_features(img, mask, leaf_area):
    """
    Trích xuất đặc trưng gân lá, rotation-invariant.

    Output: DIM_VEIN = 17 chiều float32, L2-normalized.
      [density | 16-bin angle histogram (relative to leaf main axis)]

    Rotation invariance:
      Góc Sobel được trừ đi góc trục chính lá (từ image moments) trước khi
      build histogram → cùng loài lá dù xoay vẫn cho histogram gần nhau.

    Pipeline:
      1. Erode mask
      2. CLAHE + GaussianBlur 9×9 σ=2
      3. Canny fixed (low=15, high=40)
      4. Sobel magnitude filter top-30%
      5. Connected components: loại nét ngắn
      6. MORPH_CLOSE 2×2
      7. density = #vein_px / inner_area
      8. Tính góc trục chính lá (image moments)
      9. angle histogram 16-bin, relative to main axis, sum-normalized
      10. L2-normalize
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 1. Erode mask ──────────────────────────────────────────────────
    k_erode    = np.ones((15, 15), np.uint8)
    mask_inner = cv2.erode(mask, k_erode, iterations=1)

    inner_area = float((mask_inner > 0).sum())
    if inner_area < 500:
        return np.zeros(DIM_VEIN, dtype=np.float32)

    # ── 2. CLAHE + Gaussian blur ───────────────────────────────────────
    clahe       = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced    = clahe.apply(gray)
    smooth_full = cv2.GaussianBlur(enhanced, (9, 9), 2.0)
    smooth      = cv2.bitwise_and(smooth_full, smooth_full, mask=mask_inner)

    # ── 3. Canny fixed threshold ───────────────────────────────────────
    CANNY_LOW, CANNY_HIGH = 15, 40
    canny = cv2.Canny(smooth, CANNY_LOW, CANNY_HIGH)
    vein  = cv2.bitwise_and(canny, canny, mask=mask_inner)

    # ── 4. Sobel magnitude filter: top 30% ────────────────────────────
    sx  = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)

    if (vein > 0).any():
        mag_threshold = np.percentile(mag[vein > 0], 70)
        strong_mask   = (mag >= mag_threshold).astype(np.uint8) * 255
        vein          = cv2.bitwise_and(vein, strong_mask)

    # ── 5. Connected components: loại nét ngắn (vectorized) ─────────
    min_vein_len = max(5, int(min(gray.shape) * 0.005))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vein, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]              # shape (n_labels-1,)
        valid = np.where(areas >= min_vein_len)[0] + 1   # label indices giữ lại
        vein  = np.isin(labels, valid).astype(np.uint8) * 255
    else:
        vein  = np.zeros_like(vein)

    # ── 6. MORPH_CLOSE 2×2 ────────────────────────────────────────────
    k2   = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, k2, iterations=1)

    # ── 7. Density ────────────────────────────────────────────────────
    density = float((vein > 0).sum()) / inner_area

    # ── 8. Góc trục chính lá (rotation normalization) ─────────────────
    #
    #  Vein angles tính theo frame ảnh bị ảnh hưởng bởi hướng đặt lá.
    #  Fix: trừ góc trục chính lá → histogram tính relative to leaf axis.
    #
    leaf_angle_rad = _leaf_main_axis_angle(mask_inner)
    leaf_angle_deg = leaf_angle_rad * 180.0 / np.pi  # trong [-90°, 90°)

    # ── 9. Angle histogram 16-bin, rotation-invariant ─────────────────
    mask_f    = (mask_inner > 0).astype(np.float64)
    angle_map = np.arctan2(sy * mask_f, sx * mask_f) * 180.0 / np.pi % 180.0

    # Normalize về frame của lá: trừ góc trục chính, wrap về [0, 180°)
    angle_map_norm = (angle_map - leaf_angle_deg) % 180.0

    if (vein > 0).any():
        hist, _ = np.histogram(
            angle_map_norm[vein > 0],
            bins=N_VEIN_BINS, range=(0, 180), density=False)
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist = hist / s   # probability distribution
    else:
        hist = np.zeros(N_VEIN_BINS, dtype=np.float32)

    # ── 10. Scale density → cùng magnitude với hist bins trước L2 ────
    #
    #  Vấn đề: density ≈ 0.2-0.4 >> mỗi hist bin ≈ 0.06 (average 1/16)
    #  → sau L2-norm, density chiếm ~75% vector direction
    #  → cosine distance bị dominated by density, histogram gần như vô nghĩa
    #
    #  Fix: scale density về cùng "unit" với mỗi bin histogram
    #  density_scaled = density * (1/N_VEIN_BINS) * N_VEIN_BINS = density giữ nguyên?
    #  Không — scale density xuống bằng average bin magnitude (1/N_VEIN_BINS):
    #  density_scaled = density / N_VEIN_BINS
    #  → density_scaled ≈ 0.013-0.025 ≈ mỗi bin histogram
    #  → histogram đóng góp tương đương với density trong cosine distance
    #
    density_scaled = density / N_VEIN_BINS  # ≈ 0.0125-0.025, cùng scale với hist bins
    vein_vec = np.concatenate([[density_scaled], hist]).astype(np.float32)

    # ── 11. L2-normalize ──────────────────────────────────────────────
    norm = np.linalg.norm(vein_vec)
    if norm > 1e-6:
        vein_vec = vein_vec / norm

    return vein_vec
