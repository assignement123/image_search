import cv2
import numpy as np

def extract_vein_features(img: np.ndarray, mask: np.ndarray, leaf_area: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    canny = cv2.Canny(smooth, 20, 80)
    sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=5)
    smag = cv2.normalize(np.hypot(sx, sy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    edges = cv2.addWeighted(smag, 0.6, canny, 0.4, 0)
    vein = cv2.bitwise_and(edges, edges, mask=mask)
    
    k3 = np.ones((3, 3), np.uint8)
    vein = cv2.morphologyEx(vein, cv2.MORPH_OPEN, k3, iterations=1)
    vein = cv2.dilate(vein, k3, iterations=1)
    
    density = float((vein > 0).sum() / leaf_area) if leaf_area > 0 else 0.0
    angle_map = np.arctan2(sy, sx) * 180.0 / np.pi % 180.0
    if (vein > 0).any():
        hist, _ = np.histogram(angle_map[vein > 0], bins=8, range=(0, 180), density=True)
    else:
        hist = np.zeros(8)
        
    return np.concatenate([[density], hist]).astype(np.float32)