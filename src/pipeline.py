import cv2
from features.contour import preprocess_leaf, extract_efd, extract_morphology
from features.texture import extract_lbp, extract_glcm
from features.color import extract_color_moments
from features.vein import extract_vein_features

def process_single_image(image_path: str) -> dict:
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    return {
        "efd_coeffs":       extract_efd(contour),
        "morphology_stats":  extract_morphology(contour),
        "lbp_hist":         extract_lbp(gray_masked, mask),
        "glcm_stats":       extract_glcm(gray_masked, mask),
        "color_moments":    extract_color_moments(img, mask),
        "vein_features":    extract_vein_features(img, mask, leaf_area),
    }