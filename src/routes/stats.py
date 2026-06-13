# src/routes/stats.py
from flask import Blueprint, jsonify
import psycopg2.extras

from src.db.postgres_repo import connect_db

stats_bp = Blueprint('stats', __name__)

@stats_bp.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        conn = connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT COUNT(*) as total FROM leaf_collection")
        total_images = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(DISTINCT species) as total FROM leaf_collection")
        total_species = cur.fetchone()["total"]

        cur.execute("""
            SELECT species, COUNT(*) as count FROM leaf_collection
            WHERE species IS NOT NULL GROUP BY species ORDER BY count DESC
        """)
        species_dist = [dict(r) for r in cur.fetchall()]

        # Đọc dynamic từ bảng meta
        cur.execute("SELECT key, value FROM meta")
        meta_rows = cur.fetchall()
        extract_params = {r["key"]: r["value"] for r in meta_rows}

        # Fallback nếu bảng meta rỗng — dùng giá trị thực từ config.py
        if not extract_params:
            extract_params = {
                "dim_efd": "57",
                "dim_morphology": "3",
                "dim_lbp": "26",
                "dim_glcm": "20",
                "dim_color": "9",
                "dim_vein": "17",
                "harmonics": "15",
                "n_resample": "600",
                "glcm_levels": "64",
            }

        # Tính tổng chiều vector
        try:
            dim_total = (
                int(extract_params.get("dim_efd", 57)) +
                int(extract_params.get("dim_morphology", 3)) +
                int(extract_params.get("dim_lbp", 26)) +
                int(extract_params.get("dim_glcm", 20)) +
                int(extract_params.get("dim_color", 9)) +
                int(extract_params.get("dim_vein", 17))
            )
        except (ValueError, TypeError):
            dim_total = 132

        return jsonify({
            "total_images": total_images,
            "total_species": total_species,
            "species_distribution": species_dist,
            "extract_params": extract_params,
            "dim_total": dim_total,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn:
            conn.close()