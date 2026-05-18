# src/routes/search.py
import os
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify
import psycopg2.extras

from src.pipeline import process_single_image
from src.db.postgres_repo import connect_db

search_bp = Blueprint('search', __name__)

@search_bp.route("/api/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = process_single_image(tmp_path)
    except Exception as e:
        print("\n❌ LỖI TẠI API SEARCH:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 422
    finally:
        os.unlink(tmp_path)

    w_efd = float(request.form.get("w_efd", 0.4))
    w_texture = float(request.form.get("w_texture", 0.3))
    w_color = float(request.form.get("w_color", 0.2))
    w_vein = float(request.form.get("w_vein", 0.1))
    top_k = int(request.form.get("top_k", 10))

    try:
        conn = connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = f"""
            SELECT filename, species,
            (
                {w_efd} * GREATEST(0, 1 - (efd_coeffs <=> %(efd)s::vector)) +
                ({w_texture} * 0.5) * GREATEST(0, 1 - (lbp_hist <=> %(lbp)s::vector)) +
                ({w_texture} * 0.5) * GREATEST(0, 1 - (glcm_stats <=> %(glcm)s::vector)) +
                {w_color} * GREATEST(0, 1 - (color_moments <=> %(color)s::vector)) +
                {w_vein} * GREATEST(0, 1 - (vein_features <=> %(vein)s::vector))
            ) AS similarity
            FROM leaf_collection
            ORDER BY similarity DESC
            LIMIT %(limit)s
        """

        cur.execute(query, {
            "efd": feats["efd_coeffs"].tolist(),
            "lbp": feats["lbp_hist"].tolist(),
            "glcm": feats["glcm_stats"].tolist(),
            "color": feats["color_moments"].tolist(),
            "vein": feats["vein_features"].tolist(),
            "limit": top_k
        })

        rows = cur.fetchall()
        
        results = [{
            "filename": r["filename"],
            "species": r["species"],
            "similarity": round(float(r["similarity"]) * 100, 2),
            "image_url": f"/api/image/{r['filename']}"
        } for r in rows]

        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()