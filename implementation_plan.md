# Leaf Image Search — Web UI Implementation Plan

Hiện tại dự án chỉ có các script Python CLI. Mục tiêu là xây dựng một **giao diện web đầy đủ** phục vụ tất cả chức năng của hệ thống: tìm kiếm ảnh lá tương đồng, duyệt loài, xem thống kê database, và quản lý dữ liệu.

## Proposed Changes

---

### Component 1: Flask Backend API

#### [NEW] [app.py](file:///c:/Users/Admin/image_search_new%20-%20Copy/app.py)

Server Flask đóng vai trò backend REST API, kết nối đến PostgreSQL/pgvector.  
Các endpoint:

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/search` | Upload 1 ảnh → extract features → trả về top-K ảnh tương đồng nhất (kèm score, species, filename) |
| `GET` | `/api/species` | Danh sách tất cả loài có trong DB (tên + số lượng ảnh) |
| `GET` | `/api/species/<name>` | Danh sách ảnh của 1 loài |
| `GET` | `/api/stats` | Thống kê DB: tổng số ảnh, số loài, tham số extract |
| `GET` | `/api/image/<filename>` | Serve ảnh lá từ `leaves_data/` folder |
| `POST` | `/api/build` | Kích hoạt build database từ thư mục `leaves_data` (chạy nền) |
| `GET` | `/api/build/status` | Trạng thái tiến trình build |

Tính năng tìm kiếm dùng **weighted cosine similarity** kết hợp 4 vector (efd, texture, color, vein) với trọng số có thể điều chỉnh.

#### [MODIFY] [requirements.txt](file:///c:/Users/Admin/image_search_new%20-%20Copy/requirements.txt)
Thêm: `flask`, `flask-cors`

---

### Component 2: Frontend (HTML/CSS/JS)

Tạo thư mục `static/` và `templates/` theo chuẩn Flask.

#### [NEW] `templates/index.html`
Một trang SPA với các tab:
- **🔍 Tìm kiếm** — Drag & drop / chọn ảnh → hiển thị kết quả tương đồng dạng grid với similarity score
- **🌿 Duyệt loài** — Grid các loài, click vào xem tất cả ảnh của loài đó
- **📊 Thống kê** — Biểu đồ số lượng ảnh theo loài, tổng số ảnh, info database
- **⚙️ Quản lý** — Nút Build Database, xem tiến trình

#### [NEW] `static/style.css`
Design premium: dark mode, glassmorphism cards, gradient accent, animations.

#### [NEW] `static/app.js`
Frontend logic: drag & drop upload, fetch API calls, render kết quả, biểu đồ (Chart.js từ CDN).

---

## Verification Plan

### Automated Tests
Chạy Flask server trước:
```bash
cd "c:\Users\Admin\image_search_new - Copy"
python app.py
```

### Manual Verification (Browser)
1. Mở `http://localhost:5000` trong trình duyệt
2. **Tab Tìm kiếm**: Kéo thả ảnh [1060.jpg](file:///c:/Users/Admin/image_search_new%20-%20Copy/1060.jpg) hoặc [1270.jpg](file:///c:/Users/Admin/image_search_new%20-%20Copy/1270.jpg) → kiểm tra kết quả hiện ra
3. **Tab Duyệt loài**: Xem danh sách 32 loài, click vào 1 loài → xem danh sách ảnh
4. **Tab Thống kê**: Kiểm tra số liệu database hiện đúng
5. **Slider trọng số**: Điều chỉnh trọng số EFD/Texture/Color/Vein → tìm kiếm lại để thấy kết quả thay đổi

> [!IMPORTANT]
> Docker container `leaf_vector_db` phải đang chạy trước khi test. Chạy `docker-compose up -d` nếu chưa chạy.

> [!NOTE]
> Nếu DB chưa có dữ liệu, tab Thống kê sẽ hiện 0 ảnh. Build DB bằng Tab Quản lý hoặc chạy `python leaf_extract.py --data leaves_data`.
