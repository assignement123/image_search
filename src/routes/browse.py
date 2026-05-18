# src/routes/browse.py
import os
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
import psycopg2.extras

from src.db.postgres_repo import connect_db

browse_bp = Blueprint('browse', __name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LEAVES_DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "leaves_data")).resolve()

@browse_bp.route("/api/image/<filename>")
def serve_image(filename):
    safe = Path(filename).name
    for path in LEAVES_DATA_DIR.rglob(safe):
        return send_file(str(path))
    return jsonify({"error": "Not found"}), 404

@browse_bp.route("/api/species", methods=["GET"])
def get_species_list():
    try:
        conn = connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT species, COUNT(*) as count FROM leaf_collection
            WHERE species IS NOT NULL GROUP BY species ORDER BY species
        """)
        rows = cur.fetchall()
        return jsonify({"species": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()

@browse_bp.route("/api/species/<species_name>", methods=["GET"])
def get_species_images(species_name):
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        offset = (page - 1) * per_page
        
        conn = connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("SELECT COUNT(*) as total FROM leaf_collection WHERE species = %s", (species_name,))
        total = cur.fetchone()["total"]
        
        cur.execute("""
            SELECT filename FROM leaf_collection
            WHERE species = %s ORDER BY filename LIMIT %s OFFSET %s
        """, (species_name, per_page, offset))
        
        rows = cur.fetchall()
        images = [{"filename": r["filename"], "image_url": f"/api/image/{r['filename']}"} for r in rows]
        
        return jsonify({"images": images, "total": total, "page": page, "per_page": per_page})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()