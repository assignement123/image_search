# PROMPT CHO AI AGENT — 3 NHIỆM VỤ

## CONTEXT DỰ ÁN

Đây là hệ thống CBIR (Content-Based Image Retrieval) cho lá cây, sử dụng Flask + PostgreSQL + pgvector. Cấu trúc thư mục:

```
src/
  config.py                  # HARMONICS=20, N_RESAMPLE=600, LBP_P=24, LBP_R=3, GLCM_LEVELS=64
  pipeline.py                # process_single_image() trả về dict 6 keys
  features/
    color.py                 # extract_color_moments() — đã có _l2_normalize ở cuối
    vein.py                  # extract_vein_features()
    texture.py               # extract_lbp(), extract_glcm()
    contour.py               # extract_efd(), extract_morphology()
  debug/
    color.py                 # RỖNG — cần implement hàm debug_color_moments()
    vein.py                  # RỖNG — cần implement hàm debug_vein_features()
    shape.py                 # ĐÃ CÓ — debug_shape(contour, out_dir)
    texture.py               # RỖNG
    preprocess.py            # RỖNG
    utils.py                 # save_img(), save_hist() — dùng matplotlib
    run_debug.py             # Orchestrator — import và gọi tất cả debug_*
  routes/
    stats.py                 # /api/stats — trả về extract_params hardcode
  core/
    preprocess.py            # preprocess_leaf()
static/
  js/
    stats.js                 # renderSpeciesChart(), renderParams() — cần sửa
    main.js
    api.js
templates/
  index.html                 # 4 tabs: search, browse, stats, manage
scripts/
  init_db.sql                # Schema mới với 6 vector columns + bảng meta
```

### Schema database hiện tại (init_db.sql):
```sql
CREATE TABLE leaf_collection (
    id                  SERIAL PRIMARY KEY,
    filename            VARCHAR(255) UNIQUE NOT NULL,
    image_path          TEXT NOT NULL,
    species             TEXT,
    efd_coeffs          vector(76) NOT NULL,
    morphology_stats    vector(3)  NOT NULL,
    lbp_hist            vector(26) NOT NULL,
    glcm_stats          vector(20) NOT NULL,
    color_moments       vector(9)  NOT NULL,
    vein_features       vector(9)  NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE meta (
    key   VARCHAR(50) PRIMARY KEY,
    value TEXT
);
-- meta keys: harmonics, n_resample, glcm_levels,
--            dim_efd, dim_morphology, dim_lbp, dim_glcm, dim_color, dim_vein
```

### extract_color_moments() hiện tại (features/color.py):
```python
def extract_color_moments(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mhsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    moments = []
    for i, ch in enumerate(cv2.split(mhsv)):
        px = ch[mask > 0].astype(np.float64)
        if len(px) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue
        mean = circmean(px, high=179, low=0) if i == 0 else float(np.mean(px))
        std  = float(np.std(px))
        skew = float(np.cbrt(((px - mean)**3).mean())) if std > 1e-6 else 0.0
        moments.extend([mean, std, skew])
    return _l2_normalize(np.array(moments, dtype=np.float32))  # → 9D, L2-normalized
```

Vector color_moments sau _l2_normalize: 9 chiều, sum of squares = 1.
Raw values trước normalize:
  [0]=H_mean(0-179), [1]=H_std, [2]=H_skew,
  [3]=S_mean(0-255), [4]=S_std, [5]=S_skew,
  [6]=V_mean(0-255), [7]=V_std, [8]=V_skew

### debug/shape.py (tham khảo pattern):
```python
def debug_shape(contour, out_dir):
    # Dùng matplotlib để vẽ và lưu PNG
    # Dùng plt.savefig(get_path(out_dir, "shape_01_xxx.png"), dpi=110)
    # Print "    → filename.png" sau mỗi file
    # plt.close() sau mỗi figure
```

### debug/utils.py (helper):
```python
def save_img(title, img, path, cmap='gray'): ...   # lưu ảnh OpenCV/numpy
def save_hist(title, data_list, labels, colors, path, xlabel): ...  # lưu histogram
```

### stats.js hiện tại (vấn đề):
```javascript
// BUG 1: Chỉ lấy top 20 loài — slice(0, 20) — bỏ sót 12 loài còn lại
const top = dist.slice(0, 20);

// BUG 2: extract_params hardcode sai key so với schema mới
// stats.py trả về: dim_efd, dim_texture, dim_color, dim_vein (key cũ)
// init_db.sql dùng: dim_efd, dim_morphology, dim_lbp, dim_glcm, dim_color, dim_vein (key mới)

// BUG 3: labelMap trong renderParams() dùng key cũ không khớp với meta table mới
const labelMap = {
    harmonics, n_resample, glcm_levels,
    dim_efd, dim_texture, dim_color, dim_vein, background, created_by
    // THIẾU: dim_morphology, dim_lbp, dim_glcm
};

// BUG 4: KPI kpi-dim tính sai: efd+tex+col+vin với tex=46 (cũ)
// Đúng: 76 + 3 + 26 + 20 + 9 + 9 = 143
```

---

## NHIỆM VỤ 1: Implement debug/color.py

**File cần tạo:** `src/debug/color.py`

**Hàm cần implement:** `debug_color_moments(img: np.ndarray, mask: np.ndarray, out_dir: str) -> None`

**Yêu cầu chi tiết:**

Hàm này nhận ảnh BGR gốc và mask nhị phân, tạo ra các file PNG debug sau:

**color_01_hsv_channels.png** — 3 subplot ngang (H, S, V channel) của vùng lá
- Mỗi subplot: hiển thị ảnh kênh tương ứng (masked) dùng colormap phù hợp
- H: dùng cmap='hsv', S: cmap='YlOrRd', V: cmap='gray'
- Title từng subplot: "Kênh H (Hue)", "Kênh S (Saturation)", "Kênh V (Value)"

**color_02_raw_moments.png** — Bar chart 9 chiều raw (trước L2 normalize)
- X-axis labels: ["H_mean","H_std","H_skew","S_mean","S_std","S_skew","V_mean","V_std","V_skew"]
- 3 nhóm màu: H=xanh lam (#4C9BE8), S=cam (#E88A4C), V=xanh lá (#4CE87A)
- Mỗi bar có annotation giá trị ở trên (2 chữ số thập phân)
- Title: "Color Moments — Giá trị thô (trước normalize)"
- Thêm đường kẻ ngang tại y=0 (màu đỏ nét đứt, lw=0.8)

**color_03_normalized_vector.png** — Bar chart 9 chiều sau L2 normalize (vector thực sự lưu vào DB)
- Cùng format với color_02 nhưng dùng giá trị đã normalize
- Title: "Vector color_moments (9D) lưu vào DB — L2 normalized"
- Thêm text annotation ở góc trên phải: f"‖v‖ = {norm:.4f}" (norm của raw vector trước normalize)

**color_04_hue_histogram.png** — Histogram phân bố Hue của vùng lá
- Vẽ histogram pixel Hue (bins=36, range=[0,179])
- Đánh dấu H_mean bằng đường đứt dọc màu đỏ, label: f"circmean={H_mean:.1f}°"
- So sánh với arithmetic mean bằng đường đứt dọc màu xanh lá, label: f"arith_mean={arith_mean:.1f}°"
- X-axis: "Hue (0–179)", Y-axis: "Số pixel", title: "Phân bố Hue — Circular vs Arithmetic Mean"
- Thêm legend để phân biệt 2 đường

**Yêu cầu kỹ thuật:**
- Import: cv2, numpy, matplotlib.pyplot, scipy.stats.circmean
- Import: `from src.debug.utils import save_img` (dùng cho channel images)
- Tái tính raw moments bên trong hàm (không gọi extract_color_moments để tránh double normalize)
- Sau mỗi savefig: print(f"    → color_0X_name.png")
- plt.close() sau mỗi figure, matplotlib.use('Agg') ở đầu file
- Tên file output: color_01_hsv_channels.png, color_02_raw_moments.png, color_03_normalized_vector.png, color_04_hue_histogram.png

---

## NHIỆM VỤ 2: Sửa tab Thống kê (stats.js + stats.py + index.html)

### 2A — Sửa biểu đồ phân bố loài (stats.js — renderSpeciesChart)

**Vấn đề:** `dist.slice(0, 20)` bỏ sót loài. Dataset có 32 loài, tất cả cần hiển thị.

**Yêu cầu:**
- Bỏ `slice(0, 20)` — render toàn bộ `dist` (32 loài)
- Thay chart type từ `"bar"` sang `"bar"` nằm ngang (`indexAxis: 'y'`) để label dài không bị cắt
- Tăng chiều cao canvas: set `canvas#species-chart` height động = `dist.length * 28 + 60` px (tính bằng JS trước khi tạo chart)
- `maintainAspectRatio: false` — giữ nguyên
- Font size label tăng lên 12
- maxRotation: bỏ (không cần vì chart nằm ngang)
- Thêm tooltip hiển thị: "Loài: {species}\nSố ảnh: {count}"
- Color: giữ nguyên `hsla(${i * 137.5 % 360}, 65%, 55%, 0.8)`

### 2B — Sửa renderParams (stats.js)

**Vấn đề:** labelMap dùng key cũ, thiếu key mới từ meta table.

**Yêu cầu — cập nhật labelMap thành:**
```javascript
const labelMap = {
    harmonics:      "Harmonics (EFD)",
    n_resample:     "N Resample (EFD)",
    glcm_levels:    "GLCM Levels",
    dim_efd:        "Chiều EFD (efd_coeffs)",
    dim_morphology: "Chiều Morphology (morphology_stats)",
    dim_lbp:        "Chiều LBP (lbp_hist)",
    dim_glcm:       "Chiều GLCM (glcm_stats)",
    dim_color:      "Chiều Color (color_moments)",
    dim_vein:       "Chiều Vein (vein_features)",
};
```

### 2C — Sửa KPI tổng chiều vector (stats.js — loadStats)

**Vấn đề:** Tính sai tổng chiều (dùng dim_texture=46 cũ).

**Yêu cầu:** Đọc từ extract_params trả về:
```javascript
const efd  = parseInt(data.extract_params?.dim_efd        || 76);
const morph= parseInt(data.extract_params?.dim_morphology || 3);
const lbp  = parseInt(data.extract_params?.dim_lbp        || 26);
const glcm = parseInt(data.extract_params?.dim_glcm       || 20);
const col  = parseInt(data.extract_params?.dim_color      || 9);
const vin  = parseInt(data.extract_params?.dim_vein       || 9);
document.getElementById("kpi-dim").textContent = efd + morph + lbp + glcm + col + vin; // = 143
```

### 2D — Sửa stats.py (backend)

**Vấn đề:** extract_params hardcode với key cũ, không đọc từ bảng meta.

**Yêu cầu:** Đọc dynamic từ bảng meta:
```python
cur.execute("SELECT key, value FROM meta")
meta_rows = cur.fetchall()
extract_params = {r["key"]: r["value"] for r in meta_rows}
# Trả về trong response thay vì hardcode
```

---

## NHIỆM VỤ 3: Thêm tab "Giới thiệu" vào index.html

**Yêu cầu:** Thêm tab mới "📖 Giới thiệu" vào navigation, đặt trước tab "🔍 Tìm kiếm" (đặt đầu tiên trong nav).

### Nội dung tab (section#tab-overview):

**Block 1 — Project header:**
- Tiêu đề: "Leaf Image Retrieval System"
- Subtitle: "Hệ thống lưu trữ và truy vấn ảnh lá cây dựa trên nội dung (CBIR)"
- Badge: "Flavia Leaf Dataset · 1,907 ảnh · 32 loài · 143 chiều đặc trưng"

**Block 2 — Pipeline tổng quan (dạng steps ngang hoặc dọc):**
```
[Ảnh lá đầu vào]
      ↓
[Tiền xử lý] — Grayscale → Otsu INV → Morph → Contour → Mask
      ↓
[Trích xuất đặc trưng] — 6 nhóm song song:
  EFD(76D) | Morphology(3D) | LBP(26D) | GLCM(20D) | Color(9D) | Vein(9D)
      ↓
[Lưu vào PostgreSQL + pgvector] — HNSW index, L2 distance
      ↓
[Truy vấn CBIR] — Weighted fusion / Cascaded filtering → Top-N kết quả
```

**Block 3 — Bảng đặc trưng (6 hàng):**
| Nhóm | Thuật toán | Chiều | Mô tả |
|---|---|---|---|
| Hình dáng | EFD (Elliptic Fourier Descriptors) | 76D | Biên lá — bất biến scale/rotation |
| Hình thái | Morphology Stats | 3D | Diện tích, chu vi, tỷ lệ tròn |
| Texture (cục bộ) | LBP (P=24, R=3, uniform) | 26D | Pattern vi cấu trúc bề mặt |
| Texture (không gian) | GLCM (5 props × 4 dist) | 20D | Thống kê đồng xuất hiện pixel |
| Màu sắc | Color Moments (HSV) | 9D | Mean/Std/Skew trên H, S, V |
| Gân lá | Vein Density + Angle Histogram | 9D | Mật độ và hướng gân lá |

**Block 4 — Công nghệ sử dụng (dạng badge/chip):**
- Python 3 | Flask | OpenCV | scikit-image | pyefd | scipy
- PostgreSQL 16 | pgvector | HNSW Index | Docker Compose

**Block 5 — 2 chế độ truy vấn:**
- **Weighted Fusion:** Tính điểm similarity có trọng số cho từng nhóm đặc trưng, kết hợp thành 1 điểm tổng. Phù hợp khi muốn điều chỉnh tầm quan trọng từng đặc trưng.
- **Cascaded Filtering:** Lọc dần từ 1907 ảnh → 200 → 100 → 50 → 30 → Top-N. Không cần normalize cross-metric, mỗi stage dùng distance metric riêng.

### Yêu cầu kỹ thuật HTML/CSS:
- Tab button: `<button class="nav-btn" data-tab="overview" id="nav-overview">📖 Giới thiệu</button>`
- Section: `<section class="tab-content" id="tab-overview">`
- Dùng class CSS đã có trong style.css: `glass`, `panel`, `panel-title`, `kpi-card`, `kpi-row`
- Không thêm CSS mới vào style.css — chỉ dùng class đã có
- Tab overview là tab DEFAULT active khi mở app (thay vì tab search)
  → Trong index.html: `<section class="tab-content active" id="tab-overview">`
  → nav-btn active ban đầu: `<button class="nav-btn active" data-tab="overview">`
  → Tab search: bỏ class `active`

### Sửa main.js để tab overview hoạt động:
- Thêm `"overview"` vào danh sách tabs nếu có
- Không cần lazy-load data cho tab overview (nội dung static)

---

## RÀNG BUỘC QUAN TRỌNG

1. **Không sửa** `src/features/color.py`, `src/features/vein.py`, `src/core/preprocess.py`, `src/pipeline.py`
2. **Không thêm CSS mới** vào style.css — dùng class đã có
3. **debug/color.py** phải import đúng: `from src.debug.utils import save_img`
4. **debug/color.py** phải dùng `matplotlib.use('Agg')` ở đầu file (server không có display)
5. **stats.py** khi đọc meta từ DB, cần fallback về hardcode nếu bảng meta rỗng:
   ```python
   if not extract_params:
       extract_params = {"dim_efd":"76","dim_morphology":"3","dim_lbp":"26",
                         "dim_glcm":"20","dim_color":"9","dim_vein":"9",
                         "harmonics":"20","n_resample":"600","glcm_levels":"64"}
   ```
6. **Canvas species-chart** trong index.html cần có style `height` ban đầu đủ lớn (min 400px), JS sẽ override động
7. **run_debug.py đã import** `from src.debug.color import debug_color_moments` — hàm phải có đúng signature này

## THỨ TỰ THỰC HIỆN ĐỀ XUẤT

1. `src/debug/color.py` — implement debug_color_moments()
2. `src/routes/stats.py` — đọc meta từ DB thay vì hardcode
3. `static/js/stats.js` — sửa renderSpeciesChart, renderParams, loadStats KPI
4. `templates/index.html` — thêm tab overview + sửa default active tab
5. `static/js/main.js` — đảm bảo tab overview được handle
