import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from src.config import LBP_P, LBP_R, GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS
import os, csv

def get_leaf_mask(gray_img: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray_img, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def extract_lbp(gray_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray_img, LBP_P, LBP_R, method="uniform")
    lbp_masked = lbp[mask > 0]
    hist, _ = np.histogram(lbp_masked, bins=np.arange(0, LBP_P + 3), range=(0, LBP_P + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_glcm(gray_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Trích xuất đặc trưng GLCM.
    - gray_masked: Ảnh xám đã bị ép đen phần nền từ vòng ngoài.
    - mask: Mặt nạ nhị phân của chiếc lá.
    """

    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return np.zeros(20, dtype=np.float32)
        
    cropped_gray = gray_masked[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(cropped_gray)
    
    eq = cv2.bitwise_and(eq, eq, mask=cropped_mask)
    
    q = np.clip((eq // (256 // GLCM_LEVELS)).astype(np.uint8), 0, GLCM_LEVELS - 1)
    
    glcm = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    
    glcm[0, 0, :, :] = 0 
    
    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    raw_features = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props]).astype(np.float32)
    
    # ── DEBUG: ghi raw_features ra CSV ──────────────────────────────────────────
    # CSV_PATH = "glcm_debug.csv"

    # # Tạo header động: contrast_0, contrast_1, ..., homogeneity_0, ...
    # n_cols = len(raw_features) // len(props)
    # header = [f"{p}_{i}" for p in props for i in range(n_cols)]

    # write_header = not os.path.exists(CSV_PATH)
    # with open(CSV_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     if write_header:
    #         writer.writerow(header)
    #     writer.writerow([round(float(v), 6) for v in raw_features])
    # ────────────────────────────────────────────────────────────────────────────

    # norm = np.linalg.norm(raw_features)
    # if norm > 1e-7:
    #     normalized_features = raw_features / norm
    # else:
    #     normalized_features = raw_features
        
    # return normalized_features

    CLIP_RANGES = np.array([
        200.0,
        1.0,
        0.18,
        2.5,
        14.0,
    ], dtype=np.float32)

    n = len(raw_features) // len(props)
    for i in range(n):
        raw_features[3 * n + i] += 1.0

    normalized = np.clip(raw_features / np.repeat(CLIP_RANGES, n), 0.0, 1.0)
    return normalized