# Image Search

Hệ thống tìm kiếm ảnh lá cây bằng Flask + PostgreSQL + pgvector.

## Tổng quan

Dự án này:
- trích xuất đặc trưng từ ảnh lá
- lưu vector vào PostgreSQL
- tìm ảnh tương tự bằng độ đo khoảng cách + gamma
- trả về danh sách kết quả qua API

## Kiến trúc ngắn gọn

- `app.py`: điểm khởi chạy Flask
- `src/pipeline.py`: gom các bước trích xuất đặc trưng
- `src/features/`: các bộ trích xuất `color`, `contour`, `texture`, `vein`
- `src/db/postgres_repo.py`: kết nối và ghi dữ liệu vào DB
- `src/etl_pipeline.py`: nạp toàn bộ dataset ảnh vào DB
- `src/compute_normalization.py`: tính normalization params và tự lưu `gamma` vào DB
- `src/routes/search.py`: API tìm kiếm ảnh
- `scripts/init_db.sql`: schema PostgreSQL + `pgvector` + bảng `gamma`

## Cần có

- Python 3.12+
- Docker / Docker Compose
- PostgreSQL chạy cùng extension `pgvector`

## Cài đặt nhanh

### 1) Clone repo

```bash
git clone <repo-url>
cd image_search
```

### 2) Tạo môi trường Python

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 3) Tạo file `.env`

Tạo file `.env` ở thư mục gốc:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=leaf_db
DB_USER=admin
DB_PASS=admin
DATA_DIR=/path/to/your/leaves_data
```

`DATA_DIR` nên trỏ tới thư mục ảnh bạn tải từ Google Drive về máy. Repo này không còn chứa sẵn dataset `leaves_data/`.

### 4) Tạo thư mục dữ liệu và tải ảnh từ Google Drive

Dataset được lưu ở folder Google Drive sau:

```text
https://drive.google.com/drive/u/0/folders/1nWr-Y5LjuzWuagGYNg5Hjeg9IsVq75b1
```

Bạn có thể dùng `gdown` để tải cả folder này về máy. Lệnh dưới đây sẽ tạo sẵn thư mục đích và tải dữ liệu vào đó.

#### Linux / macOS

```bash
mkdir -p ~/datasets/leaves_data
python3 -m pip install --upgrade gdown
cd ~/datasets/leaves_data
gdown --folder "https://drive.google.com/drive/folders/1nWr-Y5LjuzWuagGYNg5Hjeg9IsVq75b1"
```

Sau đó cập nhật `.env`:

```env
DATA_DIR=/home/<your-user>/datasets/leaves_data
```

#### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\datasets\leaves_data" | Out-Null
py -m pip install --upgrade gdown
Set-Location "$HOME\datasets\leaves_data"
gdown --folder "https://drive.google.com/drive/folders/1nWr-Y5LjuzWuagGYNg5Hjeg9IsVq75b1"
```

Sau đó cập nhật `.env`:

```env
DATA_DIR=C:\Users\<your-user>\datasets\leaves_data
```

Nếu `gdown` báo lỗi quyền truy cập, hãy kiểm tra folder Google Drive đã được chia sẻ công khai hoặc đăng nhập đúng tài khoản Google.

## Khởi động Database

Chạy PostgreSQL + `pgvector` bằng Docker:

```bash
docker compose up -d db
```

> Lưu ý: schema chính nằm trong `scripts/init_db.sql`. Nếu container chưa tự init schema, hãy chạy tay lệnh ở bước tiếp theo.

## Khởi tạo schema DB

### Cách 1: dùng `psql` trên máy

```bash
psql -h localhost -U admin -d leaf_db -f scripts/init_db.sql
```

### Cách 2: dùng trong container

```bash
docker exec -i leaf_vector_db psql -U admin -d leaf_db < scripts/init_db.sql
```

## Nạp dữ liệu vào DB

Sau khi DB có schema, chạy ETL để trích xuất vector từ ảnh và lưu vào PostgreSQL:

```bash
python -m src/etl_pipeline
```

## Tính normalization + gamma

Sau khi ETL xong, chạy script này để:
- tính tham số normalization
- ghi file `src/normalization_params.json`
- tự động upsert `gamma` vào bảng `search_feature_gamma`

```bash
python -m src.compute_normalization
```

## Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt:

```text
http://localhost:5001
```

## API chính

### `POST /api/search`

Upload 1 ảnh và nhận kết quả tương tự nhất.

Form fields thường dùng:
- `file`: ảnh cần tìm
- `top_k`: số kết quả trả về, mặc định `10`
- `w_efd`, `w_morphology`, `w_texture`, `w_color`, `w_vein`: trọng số từng nhóm đặc trưng

## Cách chạy end-to-end

```bash
# 1. start DB
docker compose up -d db

# 2. init schema
docker exec -i leaf_vector_db psql -U admin -d leaf_db < scripts/init_db.sql

# 3. cài dependencies
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# 4. nạp dữ liệu
python src/etl_pipeline.py

# 5. tính gamma
python src/compute_normalization.py

# 6. chạy web app
python app.py
```

## Ghi chú về search

- Các feature được chuẩn hoá ở tầng extractor.
- `gamma` được lưu trong DB để backend không phải hard-code.
- Weight vẫn do request/backend điều chỉnh linh hoạt.
- Search hiện tại ưu tiên correctness và dễ chỉnh hơn là tối ưu cực đại cho dữ liệu rất lớn.

## Cấu trúc thư mục

```text
image_search/
├── app.py
├── docker-compose.yml
├── requirements.txt
├── scripts/
│   └── init_db.sql
├── src/
│   ├── db/
│   ├── features/
│   ├── routes/
│   ├── compute_normalization.py
│   ├── etl_pipeline.py
│   └── pipeline.py
├── static/
└── templates/
```

## Nếu bị lỗi

- Kiểm tra DB đã chạy chưa: `docker ps`
- Kiểm tra `.env` đúng chưa
- Chạy ETL trước khi chạy `compute_normalization.py`
- Nếu search trả rỗng, kiểm tra bảng `leaf_collection` và `search_feature_gamma`

## Lưu ý

File schema trong repo hiện là `scripts/init_db.sql`. Nếu bạn muốn dùng `docker compose` tự init schema khi container lên, hãy đảm bảo mount đúng file schema này.
