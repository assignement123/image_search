import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
from scipy.interpolate import interp1d
from config import HARMONICS, N_RESAMPLE, DIM_EFD

# Import hàm chuẩn hóa từ module preprocess bạn vừa tách
from core.preprocess import _fix_contour_orientation 

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

def extract_efd(contour: np.ndarray) -> np.ndarray:
    """
    Elliptic Fourier Descriptors — hình dạng biên lá.
    Bất biến với translation và scale.
    Output: 76 chiều  [(HARMONICS-1) × 4]
    """
    if contour is None or len(contour) < 4:
        return np.zeros(DIM_EFD, np.float32)

    contour = np.asarray(contour, np.float64).reshape(-1, 2)
    # Vẫn gọi hàm này (đã import từ preprocess) để đảm bảo an toàn nếu module được dùng độc lập
    contour = _fix_contour_orientation(contour) 
    contour = _resample_contour(contour, N_RESAMPLE)
    
    if len(contour) < 2 * HARMONICS:
        return np.zeros(DIM_EFD, np.float32)

    coeffs = elliptic_fourier_descriptors(
        contour, order=HARMONICS + 1, normalize=False)

    a1, b1, c1, d1 = coeffs[1]
    amp1 = np.sqrt(a1**2 + b1**2 + c1**2 + d1**2)
    if amp1 > 1e-10:
        coeffs /= amp1
    return coeffs[2:HARMONICS + 1, :].flatten().astype(np.float32) 

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