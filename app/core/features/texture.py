import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def extract_texture_features(image):
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

# --- QUY TRÌNH SO SÁNH 3 ẢNH ---

paths = ['data/Leaves/1006.jpg', 'data/Leaves/1123.jpg', 'data/Leaves/1007.jpg']
raw_features = []

for path in paths:
    # Bỏ comment dòng dưới nếu bạn đã có file ảnh thật
    img = cv2.imread(path) 
    feat = extract_texture_features(img)
    raw_features.append(feat)
    pass

# Giả lập dữ liệu thô của 3 ảnh (mỗi ảnh 29 đặc trưng) để chạy demo
# raw_features = np.random.rand(3, 29) 

# 1. Chuẩn hóa đồng bộ cho cả 3 ảnh
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(raw_features)

# 2. Tính ma trận tương đồng (Cosine Similarity)
# Kết quả là ma trận 3x3 thể hiện độ giống nhau giữa từng cặp
sim_matrix = cosine_similarity(normalized_features)

print("--- KẾT QUẢ SO SÁNH TRỰC TIẾP ---")
print(f"1. Độ giống nhau giữa Ảnh 1 và Ảnh 2: {sim_matrix[0][1]*100:.2f}%")
print(f"2. Độ giống nhau giữa Ảnh 1 và Ảnh 3: {sim_matrix[0][2]*100:.2f}%")
print(f"3. Độ giống nhau giữa Ảnh 2 và Ảnh 3: {sim_matrix[1][2]*100:.2f}%")

# Tìm cặp giống nhau nhất (không tính đường chéo chính)
mask = np.eye(3, dtype=bool)
sim_matrix_no_diag = np.where(mask, 0, sim_matrix)
idx = np.unravel_index(sim_matrix_no_diag.argmax(), sim_matrix_no_diag.shape)

print(f"\n=> Cặp ảnh giống nhau nhất là: {paths[idx[0]]} và {paths[idx[1]]} ({sim_matrix[idx]*100:.2f}%)")