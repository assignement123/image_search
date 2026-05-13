import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
from scipy.interpolate import interp1d
from config import HARMONICS, N_RESAMPLE, DIM_EFD

def _shoelace_signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

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

def preprocess_leaf(image_path) -> tuple:
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

def extract_morphology(contour: np.ndarray) -> np.ndarray:
    contour = contour.astype(np.float32)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    return np.array([aspect_ratio, circularity, solidity], dtype=np.float32)
