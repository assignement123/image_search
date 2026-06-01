import cv2
import numpy as np
import os
from pathlib import Path

def segment_leaf(img):
    """Tách lá khỏi nền"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu threshold
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Loại bỏ noise nhỏ
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Giữ contour lớn nhất (thân lá)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    
    return clean_mask, largest

def get_orientation_angle(mask, contour, method="pca"):
    """Tính góc xoay, trả về góc để chiều DÀI nằm NGANG"""
    
    if method == "pca":
        points = np.column_stack(np.where(mask > 0)).astype(np.float32)
        if len(points) < 10:
            return 0
        _, eigenvectors = cv2.PCACompute(points, mean=None)
        # arctan2 trả về góc của trục chính
        angle = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[0, 1]))
        
    elif method == "minrect":
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        w, h = rect[1]
        # Nếu chiều cao > chiều rộng → lá đang đứng → xoay thêm 90
        if h > w:
            angle += 90
            
    return angle


def rotate_to_horizontal(img, angle):
    """Xoay ảnh để chiều DÀI lá nằm NGANG"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Tính kích thước mới để không bị cắt góc
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    
    # Kiểm tra sau khi xoay: nếu vẫn cao hơn rộng → xoay thêm 90 độ
    rh, rw = rotated.shape[:2]
    if rh > rw:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
    
    return rotated


def normalize_leaf(img_path, method="pca"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠ Không đọc được: {img_path.name}")
        return None
    
    mask, contour = segment_leaf(img)
    if contour is None:
        print(f"  ⚠ Không tìm được lá: {img_path.name}")
        return None
    
    angle = get_orientation_angle(mask, contour, method=method)
    rotated = rotate_to_horizontal(img, angle)  # ← đổi tên hàm
    return rotated



def batch_normalize(input_dir, output_dir, method="pca", extensions=(".jpg", ".jpeg", ".png")):
    """
    Xử lý toàn bộ thư mục, giữ nguyên cấu trúc subfolder
    
    input_dir/
      ├── class_A/
      │     ├── leaf1.jpg
      │     └── leaf2.jpg
      └── class_B/
            └── leaf3.jpg
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Lấy tất cả ảnh (bao gồm subfolder)
    all_images = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in extensions
    ]
    
    print(f"Tìm thấy {len(all_images)} ảnh — method: {method}")
    
    success, failed = 0, 0
    
    for img_path in all_images:
        # Giữ nguyên cấu trúc thư mục
        relative = img_path.relative_to(input_dir)
        save_path = output_dir / relative
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  Xử lý: {relative}", end=" ... ")
        
        result = normalize_leaf(img_path, method=method)
        
        if result is not None:
            cv2.imwrite(str(save_path), result)
            print("✓")
            success += 1
        else:
            failed += 1
    
    print(f"\nHoàn tất: {success} thành công, {failed} thất bại")


# ============================================================
# CHẠY
# ============================================================
if __name__ == "__main__":
    batch_normalize(
        input_dir  = "leaves_data/",   # thư mục gốc
        output_dir = "dataset/normalized", # lưu ra đây
        method     = "pca",                # hoặc "minrect"
    )