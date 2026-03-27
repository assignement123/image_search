from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import tempfile
import subprocess
from pathlib import Path
import threading
import time

# Import feature extraction
from leaf_extract import extract_features

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent

LEAVES_DATA_DIR = BASE_DIR / "leaves_data"
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_outputs"
DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)

DEBUG_STEP_ORDER = [
    ("Tiền xử lý",           "step1_"),
    ("EFD — Hình dạng biên", "step2a_"),
    ("Texture (LBP + GLCM)", "step2b_"),
    ("Color Moments",        "step2c_"),
    ("Gân lá",               "step2d_"),
    ("Tổng hợp",             "step3_"),
]

# Global state cho build process
build_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "added": 0,
    "skipped": 0,
    "errors": 0,
    "message": "Chưa bắt đầu"
}

# ── DB ─────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host="db",
        port=5432,
        dbname="leaf_db",
        user="admin",
        password="admin"
    )

# ── HOME ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── SEARCH ─────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = extract_features(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    finally:
        os.unlink(tmp_path)

    # Lấy trọng số từ request
    w_efd = float(request.form.get("w_efd", 0.4))
    w_texture = float(request.form.get("w_texture", 0.3))
    w_color = float(request.form.get("w_color", 0.2))
    w_vein = float(request.form.get("w_vein", 0.1))
    top_k = int(request.form.get("top_k", 10))

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = f"""
        SELECT filename, species,
        (
            {w_efd} * GREATEST(0, 1 - (efd     <=> %(efd)s::vector))   +
            {w_texture} * GREATEST(0, 1 - (texture <=> %(tex)s::vector))   +
            {w_color} * GREATEST(0, 1 - (color   <=> %(color)s::vector)) +
            {w_vein} * GREATEST(0, 1 - (vein    <=> %(vein)s::vector))
        ) AS similarity
        FROM leaf_collection
        ORDER BY similarity DESC
        LIMIT %(limit)s
    """

    cur.execute(query, {
        "efd": feats["efd"].tolist(),
        "tex": feats["texture"].tolist(),
        "color": feats["color"].tolist(),
        "vein": feats["vein"].tolist(),
        "limit": top_k
    })

    rows = cur.fetchall()
    conn.close()

    results = [{
        "filename": r["filename"],
        "species": r["species"],
        "similarity": round(float(r["similarity"]) * 100, 2),
        "image_url": f"/api/image/{r['filename']}"
    } for r in rows]

    return jsonify({"results": results, "count": len(results)})

# ── IMAGE ──────────────────────────────────────────
@app.route("/api/image/<filename>")
def serve_image(filename):
    safe = Path(filename).name

    for path in LEAVES_DATA_DIR.rglob(safe):
        return send_file(str(path))

    return jsonify({"error": "Not found"}), 404

# ── SPECIES LIST ───────────────────────────────────
@app.route("/api/species", methods=["GET"])
def get_species_list():
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT species, COUNT(*) as count
            FROM leaf_collection
            WHERE species IS NOT NULL
            GROUP BY species
            ORDER BY species
        """)
        
        rows = cur.fetchall()
        conn.close()
        
        return jsonify({"species": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── SPECIES IMAGES ─────────────────────────────────
@app.route("/api/species/<species_name>", methods=["GET"])
def get_species_images(species_name):
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        offset = (page - 1) * per_page
        
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Đếm tổng
        cur.execute(
            "SELECT COUNT(*) as total FROM leaf_collection WHERE species = %s",
            (species_name,)
        )
        total = cur.fetchone()["total"]
        
        # Lấy ảnh phân trang
        cur.execute("""
            SELECT filename
            FROM leaf_collection
            WHERE species = %s
            ORDER BY filename
            LIMIT %s OFFSET %s
        """, (species_name, per_page, offset))
        
        rows = cur.fetchall()
        conn.close()
        
        images = [{
            "filename": r["filename"],
            "image_url": f"/api/image/{r['filename']}"
        } for r in rows]
        
        return jsonify({
            "images": images,
            "total": total,
            "page": page,
            "per_page": per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── STATS ──────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Tổng số ảnh
        cur.execute("SELECT COUNT(*) as total FROM leaf_collection")
        total_images = cur.fetchone()["total"]
        
        # Tổng số loài
        cur.execute("SELECT COUNT(DISTINCT species) as total FROM leaf_collection")
        total_species = cur.fetchone()["total"]
        
        # Phân bố theo loài
        cur.execute("""
            SELECT species, COUNT(*) as count
            FROM leaf_collection
            WHERE species IS NOT NULL
            GROUP BY species
            ORDER BY count DESC
        """)
        species_dist = [dict(r) for r in cur.fetchall()]
        
        conn.close()
        
        return jsonify({
            "total_images": total_images,
            "total_species": total_species,
            "species_distribution": species_dist,
            "extract_params": {
                "harmonics": 20,
                "n_resample": 600,
                "glcm_levels": 64,
                "dim_efd": 76,
                "dim_texture": 46,
                "dim_color": 9,
                "dim_vein": 9,
                "background": "white",
                "created_by": "leaf_extract.py v2.0"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── BUILD DATABASE ─────────────────────────────────
@app.route("/api/build", methods=["POST"])
def build_database():
    global build_state
    
    if build_state["running"]:
        return jsonify({"error": "Đang có tiến trình build khác"}), 400
    
    data = request.get_json()
    rebuild = data.get("rebuild", False)
    
    # Reset state
    build_state.update({
        "running": True,
        "progress": 0,
        "total": 0,
        "added": 0,
        "skipped": 0,
        "errors": 0,
        "message": "Đang khởi tạo..."
    })
    
    # Chạy build trong thread riêng
    thread = threading.Thread(target=run_build_process, args=(rebuild,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

@app.route("/api/build/status", methods=["GET"])
def build_status():
    return jsonify(build_state)

def run_build_process(rebuild):
    """
    ✅ MỖI ẢNH MỘT TRANSACTION RIÊNG - tránh lỗi "transaction aborted"
    """
    global build_state
    
    try:
        # Nếu rebuild, xóa toàn bộ dữ liệu
        if rebuild:
            build_state["message"] = "Đang xóa dữ liệu cũ..."
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("TRUNCATE TABLE leaf_collection")
            conn.commit()
            conn.close()
        
        # Đếm tổng số file
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = []
        for ext in IMG_EXTS:
            all_images.extend(LEAVES_DATA_DIR.rglob(f"*{ext}"))
        
        build_state["total"] = len(all_images)
        build_state["message"] = f"Tìm thấy {len(all_images)} ảnh"
        
        # ═══════════════════════════════════════════════════════
        # XỬ LÝ TỪNG ẢNH - MỖI ẢNH MỘT CONNECTION RIÊNG
        # ═══════════════════════════════════════════════════════
        for idx, img_path in enumerate(all_images):
            build_state["progress"] = idx + 1
            build_state["message"] = f"Đang xử lý {img_path.name}..."
            
            filename = img_path.name
            species = img_path.parent.name
            
            # Bỏ prefix số_ nếu có (vd: "1_Phyllostachys_edulis" → "Phyllostachys_edulis")
            parts = species.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                species = parts[1]
            
            # ═══ MỞ CONNECTION MỚI CHO TỪNG ẢNH ═══
            conn = None
            try:
                conn = get_conn()
                cur = conn.cursor()
                
                # Check xem đã có chưa (nếu không rebuild)
                if not rebuild:
                    cur.execute("SELECT 1 FROM leaf_collection WHERE filename = %s", (filename,))
                    if cur.fetchone():
                        build_state["skipped"] += 1
                        continue
                
                # Extract features
                feats = extract_features(str(img_path))
                
                # ✅ Insert vào DB - THÊM image_path
                cur.execute("""
                    INSERT INTO leaf_collection (filename, image_path, species, efd, texture, color, vein)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename) DO UPDATE SET
                        image_path = EXCLUDED.image_path,
                        species = EXCLUDED.species,
                        efd = EXCLUDED.efd,
                        texture = EXCLUDED.texture,
                        color = EXCLUDED.color,
                        vein = EXCLUDED.vein
                """, (
                    filename,
                    str(img_path),           # ← ✅ THÊM image_path
                    species,
                    feats["efd"].tolist(),
                    feats["texture"].tolist(),
                    feats["color"].tolist(),
                    feats["vein"].tolist()
                ))
                
                conn.commit()
                build_state["added"] += 1
                
            except Exception as e:
                import traceback
                print(f"\n❌ Error processing {filename}:")
                print(f"   Species: {species}")
                print(f"   Path: {img_path}")
                print(f"   Error: {e}")
                traceback.print_exc()
                build_state["errors"] += 1
                if conn:
                    conn.rollback()
            
            finally:
                if conn:
                    conn.close()  # ✅ Đóng connection sau mỗi ảnh
        
        build_state["message"] = "✅ Hoàn thành!"
        
    except Exception as e:
        import traceback
        build_state["message"] = f"❌ Lỗi: {str(e)}"
        print(f"\n❌ Build error: {e}")
        traceback.print_exc()
    
    finally:
        build_state["running"] = False
# ── DEBUG API ──────────────────────────────────────
@app.route("/api/debug/<filename>", methods=["GET"])
def debug_image(filename):
    safe_name = Path(filename).name
    if safe_name != filename:
        return jsonify({"error": "Invalid filename"}), 400

    # tìm ảnh gốc
    img_path = None
    for path in LEAVES_DATA_DIR.rglob(safe_name):
        img_path = path
        break

    if not img_path:
        return jsonify({"error": "Không tìm thấy ảnh"}), 404

    stem = Path(filename).stem
    out_dir = DEBUG_OUTPUT_DIR / stem
    out_dir.mkdir(exist_ok=True)

    # check cache
    pngs = list(out_dir.glob("*.png"))
    if len(pngs) > 0:
        return jsonify({
            "status": "done",
            "cached": True,
            "images": build_debug_list(stem, pngs)
        })

    # chạy debug
    try:
        result = subprocess.run(
            ["python", "leaf_debug.py", str(img_path), "--out", str(out_dir)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return jsonify({
                "error": "Debug script failed",
                "stderr": result.stderr
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Debug timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    pngs = list(out_dir.glob("*.png"))

    if len(pngs) == 0:
        return jsonify({"error": "Không tạo được ảnh debug"}), 500

    return jsonify({
        "status": "done",
        "cached": False,
        "images": build_debug_list(stem, pngs)
    })

# ── SERVE DEBUG IMAGE ─────────────────────────────
@app.route("/api/debug-image/<stem>/<img>")
def serve_debug(stem, img):
    path = DEBUG_OUTPUT_DIR / stem / img
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(path))

# ── BUILD DEBUG LIST ──────────────────────────────
def build_debug_list(stem, pngs):
    result = []
    pngs = sorted(pngs)

    for group, prefix in DEBUG_STEP_ORDER:
        for p in pngs:
            if p.name.startswith(prefix):
                result.append({
                    "group": group,
                    "url": f"/api/debug-image/{stem}/{p.name}",
                    "title": p.name
                })

    return result

# ── DEBUG VIEW ────────────────────────────────────
@app.route("/debug-view/<filename>")
def debug_view(filename):
    safe_name = Path(filename).name

    return f"""
    <html>
    <head>
        <title>Debug Viewer</title>
        <style>
            body {{ font-family: Arial; padding: 20px; background: #1a1a1a; color: #fff; }}
            img {{ margin: 10px 0; border: 1px solid #444; max-width: 100%; }}
            h4 {{ color: #4CAF50; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h2>🌿 Debug ảnh: {safe_name}</h2>
        <div id="debug">⏳ Đang tải...</div>
        <script>
            fetch(`/api/debug/{safe_name}`)
                .then(res => res.json())
                .then(data => {{
                    if (data.error) {{
                        document.getElementById("debug").innerHTML = "❌ " + data.error;
                        return;
                    }}
                    let html = "";
                    data.images.forEach(img => {{
                        html += `<h4>${{img.group}}</h4><img src="${{img.url}}" width="600"/>`;
                    }});
                    document.getElementById("debug").innerHTML = html;
                }})
                .catch(err => {{
                    document.getElementById("debug").innerHTML = "❌ Lỗi: " + err;
                }});
        </script>
    </body>
    </html>
    """

# ── MAIN ──────────────────────────────────────────
if __name__ == "__main__":
    print("🌿 Leaf Search Server running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)