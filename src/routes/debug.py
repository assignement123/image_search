# src/routes/debug.py
import os
import json
import tempfile
import subprocess
from pathlib import Path
from flask import Blueprint, jsonify, send_file, request

debug_bp = Blueprint('debug', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LEAVES_DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "leaves_data")).resolve()
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_outputs"
DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)

# Nhóm debug theo prefix file thực tế từ các debug module
DEBUG_GROUPS = [
    ("preprocess", "🔧 Tiền xử lý",       "step1_"),
    ("morphology", "🧬 Morphology",       "morph_"),
    ("shape",      "📐 Hình dạng (EFD)",   "shape_"),
    ("lbp",        "🔳 LBP",              "lbp_"),
    ("glcm",       "🧩 GLCM",             "glcm_"),
    ("color",      "🎨 Color Moments",     "color_"),
    ("vein",       "🌿 Gân lá",            "vein_"),
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

        # Kiểm tra cache — nếu đã có PNG thì trả luôn
        pngs = list(out_dir.glob("*.png"))
        if len(pngs) > 0:
            return jsonify({
                "status": "done",
                "cached": True,
                "filename": safe_name,
                "groups": build_debug_groups(stem, pngs),
                "morphology": load_morphology_summary(out_dir),
            })

        # === Chạy debug script ===
        script_path = str(BASE_DIR / "src" / "debug" / "run_debug.py")
        if not os.path.exists(script_path):
            return jsonify({"error": f"Không tìm thấy debug script: {script_path}"}), 500

        print(f"[DEBUG] Bắt đầu chạy: {script_path} với ảnh {img_path}")

        result = subprocess.run(
            ["python", script_path, str(img_path), "--out", str(out_dir)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=180
        )

        print(f"[DEBUG] Return code: {result.returncode}")
        if result.stdout:
            print("[DEBUG] STDOUT:\n", result.stdout)
        if result.stderr:
            print("[DEBUG] STDERR:\n", result.stderr)

        if result.returncode != 0:
            return jsonify({
                "error": "Debug script chạy thất bại",
                "stderr": result.stderr[:1000],
                "stdout": result.stdout[:500]
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Debug script chạy quá lâu (timeout 180s)"}), 500
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("[DEBUG EXCEPTION]\n", error_detail)
        return jsonify({"error": str(e), "detail": error_detail}), 500

    pngs = list(out_dir.glob("*.png"))
    if not pngs:
        return jsonify({"error": "Không tạo được file debug nào"}), 500

    return jsonify({
        "status": "done",
        "cached": False,
        "filename": safe_name,
        "groups": build_debug_groups(stem, pngs),
        "morphology": load_morphology_summary(out_dir),
    })


@debug_bp.route("/api/debug-upload", methods=["POST"])
def debug_upload():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    stem = Path(tmp_path).stem
    out_dir = DEBUG_OUTPUT_DIR / stem
    out_dir.mkdir(exist_ok=True)

    try:
        script_path = str(BASE_DIR / "src" / "debug" / "run_debug.py")
        if not os.path.exists(script_path):
            return jsonify({"error": f"Không tìm thấy debug script: {script_path}"}), 500

        result = subprocess.run(
            ["python", script_path, tmp_path, "--out", str(out_dir)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=180,
        )

        if result.returncode != 0:
            return jsonify({
                "error": "Debug script chạy thất bại",
                "stderr": result.stderr[:1000],
                "stdout": result.stdout[:500]
            }), 500

        pngs = list(out_dir.glob("*.png"))
        if not pngs:
            return jsonify({"error": "Không tạo được file debug nào"}), 500

        return jsonify({
            "status": "done",
            "cached": False,
            "filename": Path(file.filename).name or "input.jpg",
            "stem": stem,
            "groups": build_debug_groups(stem, pngs),
            "morphology": load_morphology_summary(out_dir),
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Debug script chạy quá lâu (timeout 180s)"}), 500
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("[DEBUG UPLOAD EXCEPTION]\n", error_detail)
        return jsonify({"error": str(e), "detail": error_detail}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@debug_bp.route("/api/debug-image/<stem>/<img>")
def serve_debug(stem, img):
    path = DEBUG_OUTPUT_DIR / stem / img
    if path.exists():
        return send_file(str(path))
    return jsonify({"error": "Not found"}), 404


@debug_bp.route("/api/debug-clear/<stem>", methods=["DELETE"])
def clear_debug_cache(stem):
    """Xoá cache debug để chạy lại."""
    import shutil
    out_dir = DEBUG_OUTPUT_DIR / Path(stem).stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    return jsonify({"status": "cleared"})


def build_debug_groups(stem: str, pngs: list) -> list:
    """
    Nhóm các PNG theo DEBUG_GROUPS. Trả về list dict:
    [{ "id": "color", "title": "...", "images": [{"url":..., "name":...}, ...] }]
    """
    sorted_pngs = sorted(pngs, key=lambda p: p.name)
    result = []
    used = set()

    for group_id, group_title, prefix in DEBUG_GROUPS:
        imgs = []
        for p in sorted_pngs:
            if p.name.startswith(prefix) and p.name not in used:
                used.add(p.name)
                imgs.append({
                    "url": f"/api/debug-image/{stem}/{p.name}",
                    "name": p.stem.replace("_", " ").title(),
                    "filename": p.name,
                })
        if imgs:
            result.append({
                "id": group_id,
                "title": group_title,
                "images": imgs,
            })

    # Nhóm "Khác" — các PNG chưa được match
    others = [p for p in sorted_pngs if p.name not in used]
    if others:
        result.append({
            "id": "other",
            "title": "📁 Khác",
            "images": [
                {
                    "url": f"/api/debug-image/{stem}/{p.name}",
                    "name": p.stem.replace("_", " ").title(),
                    "filename": p.name,
                }
                for p in others
            ]
        })

    return result


def load_morphology_summary(out_dir: Path) -> dict | None:
    summary_path = out_dir / "morphology.json"
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[DEBUG] Không đọc được morphology summary: {e}")
    return None