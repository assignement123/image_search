# src/routes/manage.py
import os
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify

from src.pipeline import process_single_image
from src.db.postgres_repo import connect_db, insert_pg

manage_bp = Blueprint('manage', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LEAVES_DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "leaves_data")).resolve()

build_state = {
    "running": False, "progress": 0, "total": 0, 
    "added": 0, "skipped": 0, "errors": 0, "message": "Chưa bắt đầu"
}

@manage_bp.route("/api/build", methods=["POST"])
def build_database():
    global build_state
    if build_state["running"]:
        return jsonify({"error": "Đang có tiến trình build khác"}), 400
    
    rebuild = request.get_json().get("rebuild", False)
    build_state.update({
        "running": True, "progress": 0, "total": 0, 
        "added": 0, "skipped": 0, "errors": 0, "message": "Đang khởi tạo..."
    })
    
    threading.Thread(target=run_build_process, args=(rebuild,), daemon=True).start()
    return jsonify({"status": "started"})

@manage_bp.route("/api/build/status", methods=["GET"])
def build_status():
    return jsonify(build_state)

def run_build_process(rebuild):
    global build_state
    try:
        if rebuild:
            build_state["message"] = "Đang xóa dữ liệu cũ..."
            conn = connect_db()
            cur = conn.cursor()
            cur.execute("TRUNCATE TABLE leaf_collection")
            conn.commit()
            conn.close()
        
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [p for p in LEAVES_DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS]
        
        build_state["total"] = len(all_images)
        
        for idx, img_path in enumerate(all_images):
            build_state["progress"] = idx + 1
            build_state["message"] = f"Đang xử lý {img_path.name}..."
            
            filename = img_path.name
            species = img_path.parent.name
            parts = species.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                species = parts[1]
            
            conn = None
            try:
                conn = connect_db()
                cur = conn.cursor()
                if not rebuild:
                    cur.execute("SELECT 1 FROM leaf_collection WHERE filename = %s", (filename,))
                    if cur.fetchone():
                        build_state["skipped"] += 1
                        continue
                
                feats = process_single_image(str(img_path))
                insert_pg(conn, filename, str(img_path), feats, species)
                build_state["added"] += 1
            except Exception as e:
                build_state["errors"] += 1
                if conn: conn.rollback()
            finally:
                if conn: conn.close()
                
        build_state["message"] = "✅ Hoàn thành!"
    except Exception as e:
        build_state["message"] = f"❌ Lỗi: {str(e)}"
    finally:
        build_state["running"] = False