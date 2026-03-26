"""
app.py  —  Flask REST API for Leaf Image Search UI
══════════════════════════════════════════════════
Khởi động:
    python app.py
Server chạy tại: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import numpy as np
import os
import io
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
import tempfile

# Import feature extraction from existing module
from leaf_extract import extract_features, connect_db, insert_pg, get_species_from_path

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────────────────────
LEAVES_DATA_DIR = Path(__file__).parent / "leaves_data"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Build status tracking
build_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "added": 0,
    "skipped": 0,
    "errors": 0,
    "message": "Chưa chạy",
    "start_time": None,
    "end_time": None,
}


# ── DB helper ────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        dbname="leaf_db",
        user="admin",
        password="admin"
    )


# ════════════════════════════════════════════════════════════════════
# TRANG CHÍNH
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ════════════════════════════════════════════════════════════════════
# API: SEARCH
# ════════════════════════════════════════════════════════════════════

@app.route("/api/search", methods=["POST"])
def search():
    """
    Upload ảnh lá → extract features → tìm K ảnh tương đồng nhất.
    Form data:
        file     : ảnh upload
        top_k    : số kết quả (default 10)
        w_efd    : trọng số EFD   (0-1, default 0.4)
        w_texture: trọng số Texture (0-1, default 0.3)
        w_color  : trọng số Color (0-1, default 0.2)
        w_vein   : trọng số Vein  (0-1, default 0.1)
    """
    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file trống"}), 400

    top_k    = int(request.form.get("top_k", 10))
    w_efd    = float(request.form.get("w_efd",     0.4))
    w_texture = float(request.form.get("w_texture", 0.3))
    w_color  = float(request.form.get("w_color",   0.2))
    w_vein   = float(request.form.get("w_vein",    0.1))

    # Normalize weights
    total_w = w_efd + w_texture + w_color + w_vein
    if total_w < 1e-6:
        w_efd = w_texture = w_color = w_vein = 0.25
        total_w = 1.0
    w_efd /= total_w
    w_texture /= total_w
    w_color /= total_w
    w_vein /= total_w

    # Save uploaded file to temp
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = extract_features(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": f"Không extract được đặc trưng: {str(e)}"}), 422
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    efd_q    = feats["efd"].tolist()
    tex_q    = feats["texture"].tolist()
    color_q  = feats["color"].tolist()
    vein_q   = feats["vein"].tolist()

    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Weighted cosine similarity query using pgvector
        # We compute weighted sum of individual cosine similarities
        query = """
            SELECT
                filename,
                species,
                image_path,
                (
                    %(w_efd)s     * (1 - (efd     <=> %(efd)s::vector))   +
                    %(w_tex)s     * (1 - (texture  <=> %(tex)s::vector))   +
                    %(w_color)s   * (1 - (color    <=> %(color)s::vector)) +
                    %(w_vein)s    * (1 - (vein     <=> %(vein)s::vector))
                ) AS similarity
            FROM leaf_collection
            ORDER BY similarity DESC
            LIMIT %(top_k)s
        """
        cur.execute(query, {
            "efd":    efd_q,
            "tex":    tex_q,
            "color":  color_q,
            "vein":   vein_q,
            "w_efd":  w_efd,
            "w_tex":  w_texture,
            "w_color": w_color,
            "w_vein": w_vein,
            "top_k":  top_k,
        })
        rows = cur.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "filename":   r["filename"],
                "species":    r["species"] or "Unknown",
                "similarity": round(float(r["similarity"]) * 100, 2),
                "image_url":  f"/api/image/{r['filename']}",
            })

        return jsonify({
            "results": results,
            "count":   len(results),
            "weights": {
                "efd":     round(w_efd, 3),
                "texture": round(w_texture, 3),
                "color":   round(w_color, 3),
                "vein":    round(w_vein, 3),
            }
        })

    except Exception as e:
        return jsonify({"error": f"Lỗi database: {str(e)}"}), 500


# ════════════════════════════════════════════════════════════════════
# API: SPECIES
# ════════════════════════════════════════════════════════════════════

@app.route("/api/species", methods=["GET"])
def list_species():
    """Danh sách tất cả loài + số lượng ảnh mỗi loài."""
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT
                COALESCE(species, 'Unknown') AS species,
                COUNT(*) AS count
            FROM leaf_collection
            GROUP BY species
            ORDER BY species
        """)
        rows = cur.fetchall()
        conn.close()

        species_list = [{"species": r["species"], "count": int(r["count"])} for r in rows]
        return jsonify({"species": species_list, "total_species": len(species_list)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/species/<species_name>", methods=["GET"])
def get_species_images(species_name):
    """Danh sách ảnh của 1 loài cụ thể."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    offset = (page - 1) * per_page

    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT filename, species
            FROM leaf_collection
            WHERE species = %s
            ORDER BY filename
            LIMIT %s OFFSET %s
        """, (species_name, per_page, offset))
        rows = cur.fetchall()

        cur.execute("SELECT COUNT(*) AS cnt FROM leaf_collection WHERE species = %s", (species_name,))
        total = cur.fetchone()["cnt"]
        conn.close()

        images = [{"filename": r["filename"], "image_url": f"/api/image/{r['filename']}"} for r in rows]
        return jsonify({
            "species": species_name,
            "images": images,
            "total": int(total),
            "page": page,
            "per_page": per_page,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════
# API: STATS
# ════════════════════════════════════════════════════════════════════

@app.route("/api/stats", methods=["GET"])
def stats():
    """Thống kê tổng quan database."""
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT COUNT(*) AS total FROM leaf_collection")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(DISTINCT species) AS scount FROM leaf_collection")
        species_count = cur.fetchone()["scount"]

        cur.execute("""
            SELECT species, COUNT(*) AS cnt
            FROM leaf_collection
            GROUP BY species
            ORDER BY cnt DESC
        """)
        species_dist = [{"species": r["species"] or "Unknown", "count": int(r["cnt"])} for r in cur.fetchall()]

        cur.execute("SELECT key, value FROM meta ORDER BY key")
        meta = {r["key"]: r["value"] for r in cur.fetchall()}

        cur.execute("SELECT MIN(created_at) AS first, MAX(created_at) AS last FROM leaf_collection")
        times = cur.fetchone()
        conn.close()

        return jsonify({
            "total_images": int(total),
            "total_species": int(species_count),
            "species_distribution": species_dist,
            "extract_params": meta,
            "first_added": str(times["first"]) if times["first"] else None,
            "last_added": str(times["last"]) if times["last"] else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════
# API: IMAGE SERVING
# ════════════════════════════════════════════════════════════════════

@app.route("/api/image/<filename>", methods=["GET"])
def serve_image(filename):
    """Serve ảnh lá từ thư mục leaves_data."""
    # Security: no path traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    # Search recursively in leaves_data
    for ext in IMG_EXTS:
        for path in LEAVES_DATA_DIR.rglob(safe_name):
            if path.is_file():
                return send_file(str(path))

    # Also check root dir (for test images like 1060.jpg, 1270.jpg)
    root_path = Path(__file__).parent / safe_name
    if root_path.exists():
        return send_file(str(root_path))

    return jsonify({"error": f"Không tìm thấy ảnh: {filename}"}), 404


# ════════════════════════════════════════════════════════════════════
# API: BUILD DATABASE
# ════════════════════════════════════════════════════════════════════

def _run_build(rebuild: bool):
    """Background thread to build database."""
    global build_status
    build_status.update({
        "running": True,
        "progress": 0,
        "total": 0,
        "added": 0,
        "skipped": 0,
        "errors": 0,
        "message": "Đang khởi động...",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    })

    try:
        image_files = sorted([
            f for f in LEAVES_DATA_DIR.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ])
        build_status["total"] = len(image_files)
        build_status["message"] = f"Tìm thấy {len(image_files)} ảnh"

        conn = connect_db()
        cur = conn.cursor()

        if rebuild:
            cur.execute("DELETE FROM leaf_collection")
            conn.commit()
            done_names = set()
        else:
            cur.execute("SELECT filename FROM leaf_collection")
            done_names = {r[0] for r in cur.fetchall()}

        for i, fpath in enumerate(image_files):
            fname = fpath.name
            build_status["progress"] = i + 1
            build_status["message"] = f"Đang xử lý: {fname}"

            if fname in done_names and not rebuild:
                build_status["skipped"] += 1
                continue

            try:
                feats = extract_features(fpath)
                species = get_species_from_path(fpath)
                insert_pg(conn, fname, str(fpath), feats, species)
                build_status["added"] += 1
                if build_status["added"] % 50 == 0:
                    conn.commit()
            except Exception as e:
                build_status["errors"] += 1

        conn.commit()
        conn.close()

        build_status["message"] = (
            f"✓ Hoàn tất! Đã thêm {build_status['added']} ảnh, "
            f"bỏ qua {build_status['skipped']}, lỗi {build_status['errors']}"
        )

    except Exception as e:
        build_status["message"] = f"✗ Lỗi: {str(e)}"
        traceback.print_exc()
    finally:
        build_status["running"] = False
        build_status["end_time"] = datetime.now().isoformat()


@app.route("/api/build", methods=["POST"])
def build_database():
    """Kích hoạt build database trong background thread."""
    if build_status["running"]:
        return jsonify({"error": "Build đang chạy, vui lòng chờ"}), 409

    rebuild = request.json.get("rebuild", False) if request.is_json else False
    t = threading.Thread(target=_run_build, args=(rebuild,), daemon=True)
    t.start()

    return jsonify({"message": "Đã bắt đầu build", "rebuild": rebuild})


@app.route("/api/build/status", methods=["GET"])
def build_status_endpoint():
    """Trả về trạng thái tiến trình build."""
    return jsonify(build_status)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  🌿  Leaf Image Search UI — Flask Server")
    print(f"  URL: http://localhost:5000")
    print(f"  Thư mục ảnh: {LEAVES_DATA_DIR}")
    print("═" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
