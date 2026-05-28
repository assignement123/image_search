import cv2
import numpy as np
from scipy.stats import circmean


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        return (vec / norm).astype(np.float32)
    return vec

def extract_color_moments(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mhsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    moments = []
    for i, ch in enumerate(cv2.split(mhsv)):
        px = ch[mask > 0].astype(np.float64)
        if len(px) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue
        mean = circmean(px, high=179, low=0) if i == 0 else float(np.mean(px))
        std = float(np.std(px))
        skew = float(np.cbrt(((px - mean) ** 3).mean())) if std > 1e-6 else 0.0
        moments.extend([mean, std, skew])
    return _l2_normalize(np.array(moments, dtype=np.float32))