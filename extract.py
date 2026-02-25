import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pyefd import elliptic_fourier_descriptors
import os

def preprocess_leaf(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  
    upper_green = np.array([85, 255, 255])
    
    binary = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy contour lá")
    
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()
    mask = np.zeros_like(gray) if 'gray' in locals() else np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour.astype(int)], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    leaf_area = np.sum(mask > 0)
    cv2.imwrite("debug_binary.jpg", binary)
    cv2.imwrite("debug_mask.jpg", mask)
    cv2.imwrite("debug_contour.jpg", cv2.drawContours(img.copy(), [contour.astype(int)], -1, (0,255,0), 2))
    print(f"Contour points: {len(contour)}")
    print(f"Leaf area: {leaf_area}")
    return img, contour, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask, leaf_area

def extract_efd(contour, harmonics=12):
    """Extract EFD với pyefd (harmonics 10-15 khuyến nghị)"""
    if len(contour) < 20:
        print("Contour quá ngắn, trả vector zero")
        return np.zeros(4 * (harmonics - 1))
    coeffs = elliptic_fourier_descriptors(contour, order=harmonics, normalize=True)
    efd_vec = coeffs[1:, :].flatten()
    return efd_vec.astype(np.float32)

def extract_glcm(gray_img, mask):
    """Extract GLCM texture (4 features chính, trung bình 4 hướng)"""
    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    masked_gray = cv2.resize(masked_gray, (256, 256))
    levels = 32
    masked_gray = (masked_gray // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(masked_gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=levels, symmetric=True, normed=True)
    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    glcm_vec = np.array([graycoprops(glcm, prop).mean() for prop in props])
    return glcm_vec.astype(np.float32)

def extract_color_moments(img, mask):
    """Extract color moments trong HSV (mean, std, skewness cho 3 kênh)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    moments = []
    for channel in cv2.split(masked_hsv):
        pixels = channel[mask > 0] 
        if len(pixels) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue
        mean = np.mean(pixels)
        std = np.std(pixels)
        skewness = np.cbrt(((pixels - mean)**3).mean()) if std > 0 else 0.0
        moments.extend([mean, std, skewness])
    return np.array(moments, dtype=np.float32) 

def extract_vein_density(img, mask, leaf_area):
    """Extract vein density - Phiên bản cải tiến mạnh để detect gân mỏng"""
    print("   Đang extract vein density (cải tiến)...") 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite("debug_vein_enhanced.jpg", smoothed)
    print("   Đã lưu debug_vein_enhanced.jpg - ảnh sau khi tăng cường gân")
    sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny_edges = cv2.Canny(smoothed, 20, 80)  
    edges = cv2.addWeighted(sobel, 0.6, canny_edges, 0.4, 0)
    vein_mask = cv2.bitwise_and(edges, edges, mask=mask)
    kernel_small = np.ones((3,3), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    vein_mask = cv2.dilate(vein_mask, kernel_small, iterations=1)
    cv2.imwrite("debug_vein_mask.jpg", vein_mask)
    print("   Đã lưu debug_vein_mask.jpg - ảnh gân lá cuối cùng (đường trắng = gân)")
    
    vein_pixels = np.sum(vein_mask > 0)
    density = vein_pixels / leaf_area if leaf_area > 0 else 0.0
    
    print(f"   Vein pixels detected: {vein_pixels}")
    print(f"   Vein density: {density:.6f}")
    print(f"   → Nếu thấy nhiều đường trắng dọc lá → tốt. Nếu ít → giảm threshold Canny thêm")
    
    return np.array([density], dtype=np.float32)

def extract_fused_features(image_path, harmonics=12):
    """Hàm chính: extract tất cả và fuse + debug chi tiết từng bước"""
    print(f"\n=== BẮT ĐẦU EXTRACT CHO ẢNH: {image_path} ===")
    img, contour, gray, mask, leaf_area = preprocess_leaf(image_path)
    print(f"1. Preprocessing hoàn tất:")
    print(f"   - Số điểm contour: {len(contour)}")
    print(f"   - Diện tích lá (pixel): {leaf_area}")
    print(f"   - Kích thước ảnh gốc: {img.shape}")
    
    efd_vec = extract_efd(contour, harmonics)
    print(f"\n2. EFD (Elliptical Fourier Descriptors):")
    print(f"   - Số chiều: {len(efd_vec)}")
    print(f"   - 10 giá trị đầu: {efd_vec[:10]}")
    print(f"   - Min/Max/Mean: {np.min(efd_vec):.6f} / {np.max(efd_vec):.6f} / {np.mean(efd_vec):.6f}")
    print(f"   → Giá trị nhỏ, không toàn zero → biên lá được mô tả tốt")
    
    glcm_vec = extract_glcm(gray, mask)
    print(f"\n3. GLCM (Texture features):")
    print(f"   - Vector đầy đủ: {glcm_vec}")
    print(f"   - Contrast: {glcm_vec[0]:.4f} (cao → vân gân rõ)")
    print(f"   - Homogeneity: {glcm_vec[1]:.4f} (gần 1 → lá mịn)")
    print(f"   - Energy: {glcm_vec[2]:.4f} (thấp → texture phức tạp)")
    print(f"   - Correlation: {glcm_vec[3]:.4f} (gần 1 → cấu trúc có quy luật)")
    
    color_vec = extract_color_moments(img, mask)
    print(f"\n4. Color moments (HSV):")
    print(f"   - Vector đầy đủ: {color_vec}")
    print(f"   - Hue mean/std/skew: {color_vec[0]:.2f} / {color_vec[1]:.2f} / {color_vec[2]:.2f}")
    print(f"   - Saturation mean/std/skew: {color_vec[3]:.2f} / {color_vec[4]:.2f} / {color_vec[5]:.2f}")
    print(f"   - Value mean/std/skew: {color_vec[6]:.2f} / {color_vec[7]:.2f} / {color_vec[8]:.2f}")
    print(f"   → Hue ~60-120 là xanh lá → hợp lý cho lá cây")
    
    vein_vec = extract_vein_density(img, mask, leaf_area)
    print(f"\n5. Vein density:")
    print(f"   - Giá trị: {vein_vec[0]:.6f}")
    print(f"   → 0.05–0.25 là hợp lý cho lá có gân rõ (như lá sả/tre)")
    
    fused_vector = np.concatenate([efd_vec, glcm_vec, color_vec, vein_vec])
    
    print(f"\n6. Fused vector tổng:")
    print(f"   - Số chiều: {len(fused_vector)}")
    print(f"   - 10 giá trị đầu: {fused_vector[:10]}")
    print(f"   - Min/Max/Mean: {np.min(fused_vector):.6f} / {np.max(fused_vector):.6f} / {np.mean(fused_vector):.6f}")
    
    print("=== HOÀN THÀNH EXTRACT ===")
    
    return fused_vector, {
        "efd": efd_vec.tolist(),
        "glcm": glcm_vec.tolist(),
        "color": color_vec.tolist(),
        "vein": vein_vec.tolist(),
        "fused": fused_vector.tolist()
    }

if __name__ == "__main__":
    path = "1060.jpg" 
    fused, separate_features = extract_fused_features(path, harmonics=12)
    
    print("\n=== TÓM TẮT CÁC VECTOR RIÊNG LẺ (dạng list để lưu) ===")
    print('"efd":', separate_features["efd"][:10], "...")  
    print('"glcm":', separate_features["glcm"])
    print('"color":', separate_features["color"])
    print('"vein":', separate_features["vein"])
    print('"fused":', separate_features["fused"][:10], "...")