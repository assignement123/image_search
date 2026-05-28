# src/routes/debug.py
import os
import subprocess
from pathlib import Path
from flask import Blueprint, jsonify, send_file

debug_bp = Blueprint('debug', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LEAVES_DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "leaves_data")).resolve()
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

@debug_bp.route("/api/debug/<filename>", methods=["GET"])
def debug_image(filename):
    try:
        safe_name = Path(filename).name
        img_path = next(LEAVES_DATA_DIR.rglob(safe_name), None)

        if not img_path:
            return jsonify({"error": f"Không tìm thấy ảnh: {safe_name}"}), 404

        stem = Path(filename).stem
        out_dir = DEBUG_OUTPUT_DIR / stem
        out_dir.mkdir(exist_ok=True)

        # Kiểm tra cache
        pngs = list(out_dir.glob("*.png"))
        if len(pngs) > 0:
            return jsonify({
                "status": "done", 
                "cached": True, 
                "images": build_debug_list(stem, pngs)
            })

        # === Chạy debug script ===
        script_path = str(BASE_DIR / "src" / "debug" / "run_debug.py")
        if not os.path.exists(script_path):
            return jsonify({"error": f"Không tìm thấy debug script: {script_path}"}), 500

        print(f"[DEBUG] Bắt đầu chạy: {script_path} với ảnh {img_path}")  # Log

        result = subprocess.run(
            ["python", script_path, str(img_path), "--out", str(out_dir)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120
        )

        print(f"[DEBUG] Return code: {result.returncode}")

        if result.stdout:
            print("[DEBUG] STDOUT:\n", result.stdout)
        if result.stderr:
            print("[DEBUG] STDERR:\n", result.stderr)

        if result.returncode != 0:
            return jsonify({
                "error": "Debug script chạy thất bại",
                "stderr": result.stderr[:500],   # giới hạn độ dài
                "stdout": result.stdout[:300]
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Debug script chạy quá lâu (timeout)"}), 500
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("[DEBUG EXCEPTION]\n", error_detail)
        return jsonify({"error": str(e), "detail": error_detail}), 500

    # Lấy lại danh sách ảnh sau khi chạy
    pngs = list(out_dir.glob("*.png"))
    if not pngs:
        return jsonify({"error": "Không tạo được file debug nào"}), 500

    return jsonify({
        "status": "done", 
        "cached": False, 
        "images": build_debug_list(stem, pngs)
    })

@debug_bp.route("/api/debug-image/<stem>/<img>")
def serve_debug(stem, img):
    path = DEBUG_OUTPUT_DIR / stem / img
    return send_file(str(path)) if path.exists() else (jsonify({"error": "Not found"}), 404)

def build_debug_list(stem, pngs):
    result = []
    for group, prefix in DEBUG_STEP_ORDER:
        for p in sorted(pngs):
            if p.name.startswith(prefix):
                result.append({"group": group, "url": f"/api/debug-image/{stem}/{p.name}", "title": p.name})
    return result

@debug_bp.route("/debug-view/<filename>")
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
                    if (data.error) {{ document.getElementById("debug").innerHTML = "❌ " + data.error; return; }}
                    let html = "";
                    data.images.forEach(img => html += `<h4>${{img.group}}</h4><img src="${{img.url}}" width="600"/>`);
                    document.getElementById("debug").innerHTML = html;
                }})
        </script>
    </body>
    </html>
    """