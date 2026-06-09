import cv2
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa contour về float64, shape (-1,2).
    Đã bỏ:
    - _remove_self_intersections: 0/1907 ảnh Flavia cần (+ mất sub-pixel accuracy)
    - CCW check: OpenCV RETR_EXTERNAL luôn trả CCW (200/200 test)
    - Starting point (argmin x): thừa vì pyefd normalize=True tự chọn starting point
    """
    return np.asarray(contour, dtype=np.float64).reshape(-1, 2)

FIXED_THRESH = 210

def preprocess_leaf(image_path) -> tuple:
    """
    Tiền xử lý ảnh lá trên nền TRẮNG.
    Trả về: (img, contour, gray, mask, leaf_area)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── 1. Blur nhẹ — giảm JPEG artifact trước khi threshold ────────
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 2. Ngưỡng cố định — ổn định hơn Otsu với nền trắng chuẩn hóa
    _, binary = cv2.threshold(gray_blur, FIXED_THRESH, 255,
                              cv2.THRESH_BINARY_INV)

    # ── 3. Morphology ────────────────────────────────────────────────
    kernel_open  = np.ones((3, 3), np.uint8)   # không xóa cuống lá mỏng
    kernel_close = np.ones((3, 3), np.uint8)   # lấp lỗ hổng nhỏ, không bridge rãnh lá
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # ── 4. Lọc contour nhỏ (bóng đổ, vết bẩn) ──────────────────────
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("Không tìm thấy contour lá")

    min_area      = 0.01 * h * w
    cnts_filtered = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts_filtered:
        cnts_filtered = cnts   # fallback

    raw     = max(cnts_filtered, key=cv2.contourArea).squeeze().reshape(-1, 2)
    contour = _fix_contour_orientation(raw)

    # ── 5. Mask ──────────────────────────────────────────────────────
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # _fill_holes bỏ: 0/1907 ảnh Flavia có lỗ hổng bên trong lá

    # ── 6. Validate ──────────────────────────────────────────────────
    leaf_area = int((mask > 0).sum())

    if leaf_area < 500:
        raise ValueError(f"Vùng lá quá nhỏ ({leaf_area} px)")

    ratio = leaf_area / (h * w)
    if ratio < 0.05:
        raise ValueError(
            f"Lá quá nhỏ so với ảnh ({ratio:.1%}) — kiểm tra lại ảnh hoặc nền")
    if ratio > 0.90:
        raise ValueError(
            f"Vùng lá chiếm {ratio:.1%} — có thể nền không đủ trắng "
            f"(thử tăng FIXED_THRESH)")

    return img, contour, gray, mask, leaf_area
