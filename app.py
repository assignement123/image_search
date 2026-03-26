from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import tempfile
import subprocess
from pathlib import Path

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

    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
        SELECT filename, species,
        (
            0.0*(1 - (efd <=> %(efd)s::vector)) +
            0*(1 - (texture <=> %(tex)s::vector)) +
            0*(1 - (color <=> %(color)s::vector)) +
            1*(1 - (vein <=> %(vein)s::vector))
        ) AS similarity
        FROM leaf_collection
        ORDER BY similarity DESC
        LIMIT 40
    """

    cur.execute(query, {
        "efd": feats["efd"].tolist(),
        "tex": feats["texture"].tolist(),
        "color": feats["color"].tolist(),
        "vein": feats["vein"].tolist()
    })

    rows = cur.fetchall()
    conn.close()

    results = [{
        "filename": r["filename"],
        "species": r["species"],
        "similarity": round(float(r["similarity"]) * 100, 2),
        "image_url": f"/api/image/{r['filename']}"
    } for r in rows]

    return jsonify({"results": results})

# ── IMAGE ──────────────────────────────────────────
@app.route("/api/image/<filename>")
def serve_image(filename):
    safe = Path(filename).name

    for path in LEAVES_DATA_DIR.rglob(safe):
        return send_file(str(path))

    return jsonify({"error": "Not found"}), 404

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

# ── DEBUG VIEW (KHÔNG CẦN HTML FILE) ──────────────
@app.route("/debug-view/<filename>")
def debug_view(filename):
    safe_name = Path(filename).name

    return f"""
    <html>
    <head>
        <title>Debug Viewer</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            img {{ margin: 10px 0; border: 1px solid #ccc; }}
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
                        html += `
                            <h4>${{img.group}}</h4>
                            <img src="${{img.url}}" width="300"/>
                        `;
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