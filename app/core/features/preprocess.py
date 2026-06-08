import cv2
import numpy as np

def preprocess_leaf_image(image_path, target_size=512):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Tách nền bằng Otsu (Tự động tìm ngưỡng)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Tìm Contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    
    # 3. Tính góc xoay dựa trên Moments (Trục chính của lá)
    M = cv2.moments(cnt)
    # Tính góc phi (góc của trục chính)
    angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
    angle = np.degrees(angle) + 90  # Xoay để lá đứng dọc
    
    # 4. Thực hiện xoay ảnh (Dùng nền đen để không nhiễu GLCM)
    (h, w) = img.shape[:2]
    center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # 5. Đảm bảo chóp lá hướng lên trên (Flip nếu cần)
    # Tách mask của ảnh đã xoay để kiểm tra trọng tâm
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask_rot = cv2.threshold(gray_rot, 1, 255, cv2.THRESH_BINARY)
    
    # Cắt đôi mask theo chiều ngang và đếm pixel
    h_rot, w_rot = mask_rot.shape
    top_half = mask_rot[0:h_rot//2, :]
    bottom_half = mask_rot[h_rot//2:, :]
    
    # Thông thường chóp lá nhọn hơn nên diện tích (số pixel trắng) sẽ ít hơn phần cuống
    if np.sum(top_half) > np.sum(bottom_half):
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        mask_rot = cv2.rotate(mask_rot, cv2.ROTATE_180)

    # 6. Crop sát vật thể để loại bỏ vùng đen dư thừa
    coords = cv2.findNonZero(mask_rot)
    x_c, y_c, w_c, h_c = cv2.boundingRect(coords)
    cropped = rotated[y_c:y_c+h_c, x_c:x_c+w_c]
    
    # 7. Resize và Padding vào Canvas cố định
    h_p, w_p = cropped.shape[:2]
    ratio = target_size / max(h_p, w_p)
    new_size = (int(w_p * ratio), int(h_p * ratio))
    resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)
    
    # Tạo canvas đen (Nền 0 tốt cho GLCM/LBP)
    final_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_size[0]) // 2
    y_offset = (target_size - new_size[1]) // 2
    final_canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    
    # 8. Chuyển sang ảnh xám và tăng cường Texture cho LBP/GLCM
    gray_final = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray_final)
    
    return enhanced_img

# Test thử
result = preprocess_leaf_image('data/Leaves/1123.jpg')
if result is not None:
    cv2.imwrite('leaf1123.jpg', result)

result1 = preprocess_leaf_image('data/Leaves/1002.jpg')
if result1 is not None:
    cv2.imwrite('leaf1002.jpg', result1)

result2 = preprocess_leaf_image('data/Leaves/1076.jpg')
if result2 is not None:
    cv2.imwrite('leaf1076.jpg', result2)

result3 = preprocess_leaf_image('data/Leaves/1421.jpg')
if result3 is not None:
    cv2.imwrite('leaf1421.jpg', result3)