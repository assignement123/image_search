# src/routes/debug.py
import os
import subprocess
from pathlib import Path
from flask import Blueprint, jsonify, render_template, send_file

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
    safe_name = Path(filename).name
    img_path = next(LEAVES_DATA_DIR.rglob(safe_name), None)

    if not img_path:
        return jsonify({"error": "Không tìm thấy ảnh"}), 404

    stem = Path(filename).stem
    out_dir = DEBUG_OUTPUT_DIR / stem
    out_dir.mkdir(exist_ok=True)

    pngs = list(out_dir.glob("*.png"))
    if len(pngs) > 0:
        return jsonify({"status": "done", "cached": True, "images": build_debug_list(stem, pngs)})

    try:
        script_path = str(BASE_DIR / "src" / "debug" / "run_debug.py")
        if not os.path.exists(script_path):
            script_path = str(BASE_DIR / "src" / "leaf_debug.py") # Fallback

        result = subprocess.run(
            ["python", script_path, str(img_path), "--out", str(out_dir)],
            cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({"error": "Debug script failed", "stderr": result.stderr}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Debug timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    pngs = list(out_dir.glob("*.png"))
    if len(pngs) == 0:
        return jsonify({"error": "Không tạo được ảnh debug"}), 500

    return jsonify({"status": "done", "cached": False, "images": build_debug_list(stem, pngs)})

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
    return render_template("debug_view.html", safe_name=safe_name)