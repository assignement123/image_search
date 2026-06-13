# Leaf Image Search (CBIR)

## Khởi tạo môi trường

```bash
docker-compose up -d
```

## Đối chiếu theo yêu cầu bài toán

### 1) Dataset ≥ 500 ảnh lá, cùng kích thước và tỉ lệ
- Mã nguồn kiểm tra dataset ngay trước khi upload: `validate_dataset(...)` trong `main.py`.
- Hệ thống sẽ dừng nếu:
  - số lượng ảnh `< 500`,
  - kích thước ảnh không đồng nhất,
  - tỉ lệ khung hình không đồng nhất.
- Quy ước mỗi ảnh chứa 1 lá cần được đảm bảo từ khâu thu thập dữ liệu.

### 2) Bộ đặc trưng nhận dạng lá
- Đặc trưng dùng trong `app/core/features/texture.py`:
  - **LBP 24 chiều** (`P=24, R=3`): mô tả vi cấu trúc bề mặt lá, hữu ích để tìm ảnh tương tự.
  - **GLCM 5 chiều** (contrast, homogeneity, energy, correlation, dissimilarity): mô tả quan hệ mức xám để tăng khả năng phân biệt.
- Tổng vector đặc trưng: **29 chiều**.

### 3) Hệ CSDL metadata + truy hồi theo nội dung
- DB: PostgreSQL + pgvector (`docker-compose.yml`, `init.sql`).
- Bảng metadata: `leaf_metadata_test`.
- Dữ liệu lưu gồm: `id`, `species_name`, `scientific_name`, `image_path`, `detailed_features`, `fused_vector`.
- Đã sửa index vector đúng bảng trong `init.sql`.

### 4) Truy hồi ảnh lá top-5
- Hàm truy hồi: `find_similar_leaves(...)` trong `main.py`.
- Input: ảnh query mới.
- Output: đúng **5 ảnh tương tự nhất**, sắp xếp giảm dần theo `similarity`.

#### 4a) Block diagram và workflow
```text
Query Image
   ↓
Preprocess (segment, align, crop, resize 512x512, enhance)
   ↓
Feature Extraction (LBP 24 + GLCM 5 → vector 29D)
   ↓
Vector Similarity Search (pgvector cosine)
   ↓
Top-5 Similar Leaves (descending similarity)
```

#### 4b) Kết quả trung gian trong hệ thống
- Trung gian có thể quan sát ở các bước:
  - Ảnh sau tiền xử lý (`preprocess_leaf_image`),
  - vector 29 chiều (`extract_texture_features`),
  - điểm tương đồng `similarity` khi truy vấn DB.

### 5) Demo và đánh giá
- Chạy nạp dữ liệu:
  ```bash
  python main.py
  ```
- Chạy truy hồi:
  - gọi hàm `find_similar_leaves("duong_dan_anh_query.jpg")`
  - kiểm tra danh sách 5 kết quả và điểm similarity để đánh giá chất lượng.
