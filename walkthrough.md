# Thêm cột [species](file:///c:/Users/Admin/image_search_new/leaf_extract.py#406-419) — Tóm tắt thay đổi & Hướng dẫn chạy

## Những gì đã thay đổi

### 1. [init.sql](file:///c:/Users/Admin/image_search_new/init.sql) — Schema PostgreSQL
- Thêm cột **`species TEXT`** vào bảng `leaf_collection`
- Sửa index `idx_glcm_cosine` → `idx_texture_cosine` (trỏ đúng cột [texture](file:///c:/Users/Admin/image_search_new/leaf_extract.py#271-297))

### 2. [leaf_extract.py](file:///c:/Users/Admin/image_search_new/leaf_extract.py)
| Thay đổi | Mô tả |
|---|---|
| Hàm mới [get_species_from_path(image_path)](file:///c:/Users/Admin/image_search_new/leaf_extract.py#406-419) | Lấy tên folder cha, bỏ số đầu (`1_` → `Phyllostachys_edulis`) |
| [insert_pg(...)](file:///c:/Users/Admin/image_search_new/leaf_extract.py#421-448) | Nhận thêm tham số [species](file:///c:/Users/Admin/image_search_new/leaf_extract.py#406-419), ghi vào cột mới |
| [build_database(...)](file:///c:/Users/Admin/image_search_new/leaf_extract.py#489-582) | Dùng `rglob` thay `iterdir` → quét đệ quy thư mục con |

**Ví dụ tên loài được tự động suy ra:**

| Folder cha | [species](file:///c:/Users/Admin/image_search_new/leaf_extract.py#406-419) lưu vào DB |
|---|---|
| `1_Phyllostachys_edulis` | `Phyllostachys_edulis` |
| `2_Aesculus_chinensis` | `Aesculus_chinensis` |
| `32_Taxus_chinensis` | `Taxus_chinensis` |

---

## Hướng dẫn chạy

### Bước 1 — Đảm bảo Docker đang chạy
```powershell
cd C:\Users\Admin\image_search_new
docker compose up -d
```

### Bước 2 — (Nếu cần tạo lại DB từ đầu) Áp dụng schema mới
```powershell
# Reset & khởi tạo lại schema (XÓA data cũ)
docker exec -i leaf_vector_db psql -U admin -d leaf_db -f /docker-entrypoint-initdb.d/init.sql
```
> **Lưu ý:** Bỏ qua bước này nếu đã chạy `ALTER TABLE` (đã làm ở trên — cột [species](file:///c:/Users/Admin/image_search_new/leaf_extract.py#406-419) đã tồn tại trong DB).

### Bước 3 — Kích hoạt virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Bước 4 — Extract đặc trưng & build database
```powershell
# Trỏ --data vào thư mục chứa các folder loài
python leaf_extract.py --data leaves_data\

# Nếu muốn build lại từ đầu (xóa dữ liệu cũ):
python leaf_extract.py --data leaves_data\ --rebuild
```

Script sẽ tự động:
- Duyệt đệ quy qua `leaves_data\1_Phyllostachys_edulis\`, `leaves_data\2_Aesculus_chinensis\`, ...
- Với mỗi ảnh, lấy tên folder cha → suy ra `species`
- Lưu vào PostgreSQL với cột `species` đã điền đúng

### Bước 5 — Kiểm tra kết quả trong DB
```powershell
docker exec -i leaf_vector_db psql -U admin -d leaf_db -c "SELECT species, COUNT(*) FROM leaf_collection GROUP BY species ORDER BY species;"
```

Kết quả mong đợi:
```
        species         | count
------------------------+-------
 Aesculus_chinensis     |    63
 Phyllostachys_edulis   |    59
 ...
```

### Test 1 ảnh (không cần DB)
```powershell
python leaf_extract.py --test leaves_data\1_Phyllostachys_edulis\1001.jpg
```
