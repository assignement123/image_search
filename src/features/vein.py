import cv2
import numpy as np


# # ════════════════════════════════════════════════════════════════════
# # GÂN LÁ — MẬT ĐỘ & PHÂN BỐ HƯỚNG GÂN
# # ════════════════════════════════════════════════════════════════════
# def extract_vein_features(img, mask, leaf_area):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Erode mask để loại viền lá
#     k_erode = np.ones((15, 15), np.uint8)
#     mask_inner = cv2.erode(mask, k_erode, iterations=1)

#     clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
#     enhanced = clahe.apply(gray)

#     smooth_full = cv2.GaussianBlur(enhanced, (5, 5), 0)
#     smooth      = cv2.bitwise_and(smooth_full, smooth_full, mask=mask_inner)

#     # ✅ FIX: Otsu chỉ trên pixels trong mask, không phải toàn ảnh
#     pixels_in_mask = smooth[mask_inner > 0]
#     otsu_thresh, _ = cv2.threshold(pixels_in_mask, 0, 255,
#                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     low  = float(otsu_thresh) * 0.3
#     high = float(otsu_thresh) * 0.7
#     canny = cv2.Canny(smooth, low, high)

#     vein = cv2.bitwise_and(canny, canny, mask=mask_inner)

#     k2   = np.ones((2, 2), np.uint8)
#     vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, k2, iterations=1)

#     inner_area = float((mask_inner > 0).sum())
#     density = float((vein > 0).sum() / inner_area) if inner_area > 0 else 0.0

#     # Sobel chỉ để tính angle histogram
#     sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
#     sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

#     mask_f    = (mask_inner > 0).astype(np.float64)
#     angle_map = np.arctan2(sy * mask_f, sx * mask_f) * 180.0 / np.pi % 180.0

#     if (vein > 0).any():
#         hist, _ = np.histogram(angle_map[vein > 0],
#                                bins=8, range=(0, 180), density=True)
#     else:
#         hist = np.zeros(8)
#     # Thêm vào sau dòng tính otsu_thresh để verify
#     print(f"otsu_thresh = {otsu_thresh:.1f}  |  low={low:.1f}  high={high:.1f}")
#     print(f"canny pixels = {(canny > 0).sum()}  |  inner_area = {inner_area:.0f}")
#     print(f"density = {density:.4f}")

#     return np.concatenate([[density], hist]).astype(np.float32)



import cv2
import numpy as np


def extract_vein_features(img, mask, leaf_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k_erode    = np.ones((15, 15), np.uint8)
    mask_inner = cv2.erode(mask, k_erode, iterations=1)

    inner_area = float((mask_inner > 0).sum())
    if inner_area < 500:
        return np.zeros(9, dtype=np.float32)

    clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    smooth_full = cv2.GaussianBlur(enhanced, (9, 9), 2.0)
    smooth      = cv2.bitwise_and(smooth_full, smooth_full, mask=mask_inner)

    # ── NGƯỠNG CANNY CỐ ĐỊNH thay Otsu ──────────────────────────────
    #
    #  Tại sao Otsu thất bại với lá sáng / lá khô:
    #    - Pixel trong mask_inner của lá sáng tập trung ở 150-220
    #    - Otsu tìm ngưỡng tách 2 lớp → chọn ~100 (quá thấp)
    #    - low=30, high=70 → Canny bắt mọi gradient nhỏ = noise
    #
    #  Giải pháp: ngưỡng cố định trên gradient magnitude (Sobel)
    #  thay vì trên pixel intensity.
    #  Canny thực ra dùng gradient magnitude bên trong,
    #  nên đặt low/high trực tiếp trên scale gradient hợp lý hơn.
    #
    #  Với ảnh lá chuẩn hóa nền trắng (8-bit, scale 0-255):
    #    low  = 15  → bắt gân mảnh / lá non
    #    high = 40  → chỉ giữ gân có gradient rõ
    #
    #  Nếu vẫn noise: tăng lên low=20, high=50
    #  Nếu mất gân:   giảm xuống low=10, high=30
    #
    CANNY_LOW  = 15
    CANNY_HIGH = 40

    canny = cv2.Canny(smooth, CANNY_LOW, CANNY_HIGH)
    vein  = cv2.bitwise_and(canny, canny, mask=mask_inner)

    # ── Sobel magnitude filter ────────────────────────────────────────
    sx  = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)

    if (vein > 0).any():
        mag_threshold = np.percentile(mag[vein > 0], 70)
        strong_mask   = (mag >= mag_threshold).astype(np.uint8) * 255
        vein          = cv2.bitwise_and(vein, strong_mask)

    # ── Connected components: loại nét ngắn < min_vein_len px ────────
    min_vein_len = max(5, int(min(gray.shape) * 0.005))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vein, connectivity=8)
    clean = np.zeros_like(vein)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_vein_len:
            clean[labels == lbl] = 255
    vein = clean

    k2   = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, k2, iterations=1)

    density = float((vein > 0).sum()) / inner_area

    mask_f    = (mask_inner > 0).astype(np.float64)
    angle_map = np.arctan2(sy * mask_f, sx * mask_f) * 180.0 / np.pi % 180.0

    if (vein > 0).any():
        hist, _ = np.histogram(angle_map[vein > 0],
                               bins=8, range=(0, 180), density=True)
    else:
        hist = np.zeros(8)

    print(f"CANNY_LOW={CANNY_LOW}  CANNY_HIGH={CANNY_HIGH}  "
          f"(Otsu cũ: low≈{float(np.percentile(smooth[mask_inner>0],10)):.0f})")
    print(f"canny pixels = {(canny > 0).sum()}  |  inner_area = {inner_area:.0f}")
    print(f"density = {density:.4f}  |  min_vein_len = {min_vein_len}px")

    return np.concatenate([[density], hist]).astype(np.float32)