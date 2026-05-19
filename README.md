Dưới đây là nội dung file `README.md` được viết chuẩn theo định dạng Markdown, trình bày rõ ràng từ giới thiệu, cấu trúc thư mục cho đến các bước cài đặt và chạy dự án cho cả Windows và Linux.

Bạn chỉ cần copy toàn bộ nội dung bên dưới và lưu thành file `README.md` ở thư mục gốc của dự án (`IMAGE_SEARCH/`).

---

```markdown
# 🌿 LeafSearch - Hệ thống tìm kiếm ảnh lá cây

LeafSearch là một hệ thống web hỗ trợ tìm kiếm và nhận dạng các loại lá cây dựa trên mức độ tương đồng của hình ảnh. Hệ thống trích xuất 6 nhóm đặc trưng từ ảnh (Hình dáng, Hình thái học, Kết cấu LBP/GLCM, Màu sắc, Gân lá) và sử dụng cơ sở dữ liệu **PostgreSQL + pgvector** để lưu trữ và tìm kiếm vector (embeddings) với tốc độ cao.

## 📂 Cấu trúc dự án

Dự án được thiết kế theo chuẩn **Clean Architecture / Src Layout** của Flask:

```text
IMAGE_SEARCH/
 ├── app.py                 # File khởi chạy Flask Web Server (Entry point)
 ├── docker-compose.yml     # Cấu hình Docker chứa PostgreSQL + pgvector
 ├── requirements.txt       # Danh sách thư viện Python
 ├── .env                   # File cấu hình biến môi trường (DB, đường dẫn)
 ├── leaves_data/           # (Tự tạo) Thư mục chứa dataset ảnh lá để đưa vào DB
 ├── static/                # Chứa các tệp tĩnh Frontend (CSS, JS Modules)
 ├── templates/             # Chứa giao diện HTML (index.html)
 └── src/                   # Thư mục chứa Mã nguồn Lõi (Core Logic)
      ├── core/             # Tiền xử lý ảnh (cắt nền, tạo mask)
      ├── features/         # Thuật toán trích xuất (color, contour, texture, vein)
      ├── db/               # Module kết nối và thao tác với Database
      ├── routes/           # Các Flask Blueprints (chia nhỏ API endpoints)
      ├── config.py         # Cấu hình tham số trích xuất (Harmonics, GLCM levels...)
      ├── pipeline.py       # Luồng kết nối các bước trích xuất đặc trưng
      ├── etl_pipeline.py   # Script ETL nạp dữ liệu hàng loạt vào Database
      ├── main.py           # Công cụ test trích xuất một ảnh đơn lẻ & xuất CSV
      └── leaf_debug.py     # Script trực quan hóa chi tiết các bước xử lý ảnh

```

---

## 🚀 Hướng dẫn cài đặt và chạy dự án

### Bước 1: Khởi động Database (Bắt buộc)

Hệ thống sử dụng Docker để chạy PostgreSQL có tích hợp sẵn thư viện `pgvector`.
Mở Terminal/Command Prompt tại thư mục gốc của dự án và chạy:

```bash
# Khởi động container ngầm (chạy được cho cả Windows và Linux)
docker-compose up -d

# (Nếu dùng Docker phiên bản mới, bạn có thể dùng lệnh sau)
docker compose up -d

```

### Bước 2: Cài đặt môi trường Python

Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

**🖥️ Dành cho Windows:**

```bash
# Tạo môi trường ảo
python -m venv myenv

# Kích hoạt môi trường ảo
myenv\Scripts\activate

# Cài đặt các thư viện cần thiết và thư viện kết nối PostgreSQL
pip install -r requirements.txt
```

**🐧 Dành cho Linux / macOS:**

```bash
# Tạo môi trường ảo
python3 -m venv myenv

# Kích hoạt môi trường ảo
source myenv/bin/activate

# Cài đặt các thư viện cần thiết và thư viện kết nối PostgreSQL
pip install -r requirements.txt
```

*(Hãy đảm bảo bạn đã cấu hình file `.env` chứa thông tin kết nối DB giống với docker-compose)*.

---

### Bước 3: Nạp dữ liệu vào Database (ETL)

Thư mục chứa ảnh đưa vào Database (`leaves_data/`) thường không có cấu trúc cố định, chỉ cần là một folder chứa một list các ảnh (Ví dụ: `1004.jpg`, `1005.jpg`, `1006.jpg`...).

Để trích xuất đặc trưng của toàn bộ folder ảnh và lưu vào Database, chạy script sau:

```bash
python src/etl_pipeline.py

```

*(Hệ thống sẽ hiển thị thanh tiến trình xử lý từng ảnh)*.

---

### Bước 4: Khởi chạy Web Server

Sau khi đã có dữ liệu trong Database, bạn có thể khởi động ứng dụng Web bằng Flask:

```bash
python app.py

```

👉 Mở trình duyệt và truy cập: **http://localhost:5001** (hoặc port được báo trên terminal).

---

## 🛠️ Công cụ Debug và Testing

Dự án cung cấp các script để bạn dễ dàng kiểm tra thuật toán trích xuất trước khi đưa vào Database. Đứng ở thư mục gốc (`IMAGE_SEARCH/`), bạn chạy các lệnh sau:

**1. Trực quan hóa từng bước xử lý của một ảnh (Xuất ra ~47 ảnh debug):**

```bash
python src/leaf_debug.py leaves_data/1_Phyllostachys_edulis/1001.jpg

```

**2. Test trích xuất vector đơn lẻ và xuất ra file CSV:**

```bash
python src/main.py --test leaves_data/1_Phyllostachys_edulis/1001.jpg --export

```

---

## 🐘 Truy vấn Database mẫu (pgvector)

Dưới đây là câu lệnh SQL ví dụ để tìm kiếm 5 bức ảnh có **Hình dáng (EFD)** giống nhất với bức ảnh `1278.jpg` bằng cách tính khoảng cách L2 (Euclidean distance) `<->` trong `pgvector`:

```sql
SELECT filename
FROM leaf_collection
WHERE filename != '1278.jpg'
ORDER BY efd_coeffs <-> (
    SELECT efd_coeffs
    FROM leaf_collection
    WHERE filename = '1278.jpg'
)
LIMIT 5;

```

*(Ghi chú: Toán tử `<=>` được dùng cho Cosine Similarity, `<->` dùng cho Euclidean / L2 distance).*

```

```