import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════
# GÂN LÁ — MẬT ĐỘ & PHÂN BỐ HƯỚNG GÂN
# ════════════════════════════════════════════════════════════════════
def extract_vein_features(img, mask, leaf_area):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Erode mask để loại viền lá
    k_erode = np.ones((15, 15), np.uint8)
    mask_inner = cv2.erode(mask, k_erode, iterations=1)

    clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    smooth_full = cv2.GaussianBlur(enhanced, (5, 5), 0)
    smooth      = cv2.bitwise_and(smooth_full, smooth_full, mask=mask_inner)

    # ✅ FIX: Otsu chỉ trên pixels trong mask, không phải toàn ảnh
    pixels_in_mask = smooth[mask_inner > 0]
    otsu_thresh, _ = cv2.threshold(pixels_in_mask, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    low  = float(otsu_thresh) * 0.3
    high = float(otsu_thresh) * 0.7
    canny = cv2.Canny(smooth, low, high)

    vein = cv2.bitwise_and(canny, canny, mask=mask_inner)

    k2   = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_CLOSE, k2, iterations=1)

    inner_area = float((mask_inner > 0).sum())
    density = float((vein > 0).sum() / inner_area) if inner_area > 0 else 0.0

    # Sobel chỉ để tính angle histogram
    sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

    mask_f    = (mask_inner > 0).astype(np.float64)
    angle_map = np.arctan2(sy * mask_f, sx * mask_f) * 180.0 / np.pi % 180.0

    if (vein > 0).any():
        hist, _ = np.histogram(angle_map[vein > 0],
                               bins=8, range=(0, 180), density=True)
    else:
        hist = np.zeros(8)
    # Thêm vào sau dòng tính otsu_thresh để verify
    print(f"otsu_thresh = {otsu_thresh:.1f}  |  low={low:.1f}  high={high:.1f}")
    print(f"canny pixels = {(canny > 0).sum()}  |  inner_area = {inner_area:.0f}")
    print(f"density = {density:.4f}")

    return np.concatenate([[density], hist]).astype(np.float32)