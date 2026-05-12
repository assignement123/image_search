import cv2
from features.contour import preprocess_leaf, extract_efd, extract_morphology
from features.texture import extract_lbp, extract_glcm
from features.color import extract_color_moments
from features.vein import extract_vein_features

def process_single_image(image_path: str) -> dict:
    """
    Chạy toàn bộ pipeline trích xuất đặc trưng cho 1 ảnh.
    Trả về dictionary chứa các vector numpy.
    """
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path)
    
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    return {
        "efd":        extract_efd(contour),
        "morphology": extract_morphology(contour),
        "lbp":        extract_lbp(gray_masked, mask),
        "glcm":       extract_glcm(gray_masked, mask),
        "color":      extract_color_moments(img, mask),
        "vein":       extract_vein_features(img, mask, leaf_area),
    }