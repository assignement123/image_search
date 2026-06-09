# src/routes/search.py
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import psycopg2.extras
from flask import Blueprint, request, jsonify

from src.db.postgres_repo import connect_db
from src.pipeline import process_single_image

search_bp = Blueprint('search', __name__)


def _to_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return value


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

    # Default weights tuned cho Flavia dataset
    w_efd        = float(request.form.get("w_efd",        0.35))
    w_morphology = float(request.form.get("w_morphology", 0.05))
    w_lbp        = float(request.form.get("w_lbp",        0.15))
    w_glcm       = float(request.form.get("w_glcm",       0.15))
    w_color      = float(request.form.get("w_color",      0.20))
    w_vein       = float(request.form.get("w_vein",       0.10))
    top_k = int(request.form.get("top_k", 10))

    try:
        conn = connect_db()   # register_vector đã được gọi trong connect_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query_vectors = {
            "efd_coeffs":         _to_vector(feats["efd_coeffs"]),
            "efd_coeffs_flipped": _to_vector(feats["efd_coeffs_flipped"]),
            "morphology_stats":   _to_vector(feats["morphology_stats"]),
            "lbp_hist":           _to_vector(feats["lbp_hist"]),
            "glcm_stats":         _to_vector(feats["glcm_stats"]),
            "color_moments":      _to_vector(feats["color_moments"]),
            "vein_features":      _to_vector(feats["vein_features"]),
        }

        # Cosine similarity (1 - cosine_distance) thay cho exp(-gamma * L2):
        #   - Không cần calibrate gamma
        #   - Tự normalize theo magnitude → ổn định hơn với vector nhiều chiều (EFD 76d, vein 9d)
        #   - GREATEST(0, ...) để tránh âm khi vector lệch phase
        # LBP giữ exp(-chi_square) vì là histogram — cosine kém hơn với phân phối xác suất
        sql = """
            WITH q AS (
                SELECT
                    %s::vector AS q_efd,
                    %s::vector AS q_efd_flip,
                    %s::vector AS q_morphology,
                    %s::vector AS q_lbp,
                    %s::vector AS q_glcm,
                    %s::vector AS q_color,
                    %s::vector AS q_vein
            )
            SELECT
                lc.filename,
                lc.species,
                (
                    -- EFD: lấy max giữa normal và flipped → reflection invariant
                    (%s * GREATEST(
                        GREATEST(0, 1 - (lc.efd_coeffs <=> q.q_efd)),
                        GREATEST(0, 1 - (lc.efd_coeffs <=> q.q_efd_flip))
                    )) +
                    (%s * GREATEST(0, 1 - (lc.morphology_stats <=> q.q_morphology))) +
                    (%s * exp(-chi_square_dist(lc.lbp_hist, q.q_lbp))) +
                    (%s * GREATEST(0, 1 - (lc.glcm_stats       <=> q.q_glcm))) +
                    (%s * GREATEST(0, 1 - (lc.color_moments    <=> q.q_color))) +
                    (%s * GREATEST(0, 1 - (lc.vein_features    <=> q.q_vein)))
                ) / NULLIF((%s + %s + %s + %s + %s + %s), 0) AS similarity
            FROM leaf_collection lc
            CROSS JOIN q
            ORDER BY similarity DESC
            LIMIT %s
        """

        params = [
            query_vectors["efd_coeffs"],
            query_vectors["efd_coeffs_flipped"],
            query_vectors["morphology_stats"],
            query_vectors["lbp_hist"],
            query_vectors["glcm_stats"],
            query_vectors["color_moments"],
            query_vectors["vein_features"],
            w_efd,
            w_morphology,
            w_lbp,
            w_glcm,
            w_color,
            w_vein,
            w_efd,
            w_morphology,
            w_lbp,
            w_glcm,
            w_color,
            w_vein,
            top_k,
        ]

        cur.execute(sql, params)
        rows = cur.fetchall()

        results = [{
            "filename": r["filename"],
            "species": r["species"],
            "similarity": round(float(r["similarity"]) * 100, 2) if r["similarity"] is not None else 0.0,
            "image_url": f"/api/image/{r['filename']}"
        } for r in rows]

        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cur'  in locals(): cur.close()
        if 'conn' in locals() and conn: conn.close()
