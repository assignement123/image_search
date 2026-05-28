import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH NỘI BỘ
# ════════════════════════════════════════════════════════════════════

def _shoelace_signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _remove_self_intersections(contour: np.ndarray) -> np.ndarray:
    """
    Vẽ contour lên ảnh tạm rồi re-extract → loại bỏ self-intersection.
    Cần thiết để shoelace area tính đúng dấu.
    """
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
    clean[:, 0] += x0
    clean[:, 1] += y0
    return clean


def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa contour:
    1. Loại bỏ self-intersection (cần thiết để tính shoelace đúng)
    2. Đảm bảo hướng CCW — trong OpenCV (y tăng xuống dưới),
       shoelace > 0 tương ứng CW → cần đảo
    3. Điểm bắt đầu: x nhỏ nhất (argmin đơn giản, tie-break tự nhiên theo index)
    """
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)

    # Bước 1: loại bỏ self-intersection
    contour = _remove_self_intersections(contour)

    # Bước 2: đảm bảo CCW — gốc dùng > 0 để đảo (OpenCV y-down: > 0 = CW)
    if _shoelace_signed_area(contour) > 0:
        contour = contour[::-1].copy()

    # Bước 3: điểm bắt đầu = x nhỏ nhất (argmin, khớp với DB đã build)
    idx = int(np.argmin(contour[:, 0]))
    return np.roll(contour, -idx, axis=0)


# ════════════════════════════════════════════════════════════════════
# TIỀN XỬ LÝ ẢNH LÁ (nền TRẮNG)
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