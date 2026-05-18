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
        
        return jsonify({
            "total_images": total_images,
            "total_species": total_species,
            "species_distribution": species_dist,
            "extract_params": {
                "harmonics": 20, "n_resample": 600, "glcm_levels": 64,
                "dim_efd": 76, "dim_texture": 46, "dim_color": 9, "dim_vein": 9,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()