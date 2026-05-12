import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from config import LBP_P, LBP_R, GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS

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

def extract_glcm(gray_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    eq = cv2.equalizeHist(masked)
    resized = cv2.resize(eq, (256, 256))
    q = np.clip((resized // (256 // GLCM_LEVELS)).astype(np.uint8), 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props]).astype(np.float32)