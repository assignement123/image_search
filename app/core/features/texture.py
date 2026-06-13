import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_texture_features(image):
    """
    Trích xuất vector 29 chiều:
    - 24 chiều LBP (mô tả vi cấu trúc bề mặt lá để tìm lá tương tự)
    - 5 chiều GLCM (mức độ tương phản/đồng nhất/năng lượng/tương quan/dissimilarity để phân biệt loài)
    """
    if image is None:
        return None
    
    # Chuyển về ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. Trích xuất LBP ---
    P, R = 24, 3
    lbp = local_binary_pattern(image, P, R, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lbp_hist_cleaned = lbp_hist[1:-1].astype("float")
    lbp_hist_cleaned /= (lbp_hist_cleaned.sum() + 1e-7)

    # --- 2. Trích xuất GLCM (64 levels) ---
    image_64 = (image // 4).astype(np.uint8) 
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_64, distances=distances, angles=angles, 
                        levels=64, symmetric=True, normed=True)
    
    glcm_features = np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'dissimilarity').mean()
    ])

    return np.hstack([lbp_hist_cleaned, glcm_features])
