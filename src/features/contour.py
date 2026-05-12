import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
from scipy.interpolate import interp1d
from config import HARMONICS, N_RESAMPLE, DIM_EFD

def _shoelace_signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if _shoelace_signed_area(contour) < 0:
        contour = contour[::-1].copy()
    xs = contour[:, 0]
    x_min = xs.min()
    candidates = np.where(xs <= x_min + 0.5)[0]
    idx = candidates[np.argmin(contour[candidates, 1])]
    return np.roll(contour, -idx, axis=0)

def _resample_contour(contour: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(contour) < 4: return contour
    if np.linalg.norm(contour[-1] - contour[0]) > 1e-6:
        contour = np.vstack([contour, contour[0]])
    diffs = np.diff(contour, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cumdist = np.concatenate([[0.0], np.cumsum(dists)])
    total_length = cumdist[-1]
    if total_length < 1e-8: return contour[:n] if len(contour) > n else contour
    
    t = np.linspace(0, total_length, n, endpoint=True)
    fx = interp1d(cumdist, contour[:, 0], kind='linear')
    fy = interp1d(cumdist, contour[:, 1], kind='linear')
    resampled = np.stack([fx(t), fy(t)], axis=1)
    resampled[-1] = resampled[0]
    return resampled

def preprocess_leaf(image_path):
    img = cv2.imread(str(image_path))
    if img is None: raise ValueError(f"Không đọc được ảnh: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: raise ValueError("Không tìm thấy contour lá")
    
    raw_contour = max(cnts, key=cv2.contourArea)
    
    contour_points = raw_contour.reshape(-1, 2)
    contour = _fix_contour_orientation(contour_points)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour.astype(np.int32).reshape((-1, 1, 2))], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    leaf_area = int((mask > 0).sum())
    
    return img, contour, gray, mask, leaf_area

def extract_efd(contour: np.ndarray) -> np.ndarray:
    contour = _fix_contour_orientation(np.asarray(contour, dtype=np.float64).reshape(-1, 2))
    contour = _resample_contour(contour, N_RESAMPLE)
    if len(contour) < 2 * HARMONICS: return np.zeros(DIM_EFD, dtype=np.float32)
    
    centroid = contour.mean(axis=0)
    contour = contour - centroid
    coeffs = elliptic_fourier_descriptors(contour, order=HARMONICS+1, normalize=False)
    
    a1, b1, c1, d1 = coeffs[1]
    mag1 = np.sqrt(a1**2 + b1**2 + c1**2 + d1**2)
    if mag1 < 1e-8: return np.zeros(DIM_EFD, dtype=np.float32)
    
    coeffs_norm = coeffs[1:] / mag1
    a1_n, b1_n, c1_n, d1_n = coeffs_norm[0]
    theta = 0.5 * np.arctan2(2 * (a1_n * b1_n + c1_n * d1_n), (a1_n**2 + c1_n**2 - b1_n**2 - d1_n**2))
    
    result = []
    for i, (a, b, c, d) in enumerate(coeffs_norm):
        harmonic_order = i + 1
        angle = harmonic_order * theta
        cos_t, sin_t = np.cos(angle), np.sin(angle)
        result.extend([
            a * cos_t + b * sin_t, -a * sin_t + b * cos_t,
            c * cos_t + d * sin_t, -c * sin_t + d * cos_t
        ])
    return np.array(result[4:], dtype=np.float32)

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