import cv2
from src.core.preprocess import preprocess_leaf
from src.features.contour import extract_efd, extract_efd_flipped, extract_morphology
from src.features.texture import extract_lbp, extract_glcm
from src.features.color import extract_color_moments
from src.features.vein import extract_vein_features

def process_single_image(image_path: str) -> dict:
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    efd = extract_efd(contour)
    return {
        "efd_coeffs":         efd,
        "efd_coeffs_flipped": extract_efd_flipped(contour),  # reflection invariance
        "morphology_stats":   extract_morphology(contour),
        "lbp_hist":           extract_lbp(gray_masked, mask),
        "glcm_stats":         extract_glcm(gray_masked, mask),
        "color_moments":      extract_color_moments(img, mask),
        "vein_features":      extract_vein_features(img, mask, leaf_area),
    }