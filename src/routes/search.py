# src/routes/search.py
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import psycopg2.extras
from pgvector.psycopg2 import register_vector
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


def _load_gamma_from_db(cur) -> dict:
    cur.execute(
        """
        SELECT
            COALESCE(MAX(CASE WHEN feature_name = 'efd_coeffs' THEN gamma END), 1.0) AS efd_coeffs,
            COALESCE(MAX(CASE WHEN feature_name = 'morphology_stats' THEN gamma END), 1.0) AS morphology_stats,
            COALESCE(MAX(CASE WHEN feature_name = 'lbp_hist' THEN gamma END), 1.0) AS lbp_hist,
            COALESCE(MAX(CASE WHEN feature_name = 'glcm_stats' THEN gamma END), 1.0) AS glcm_stats,
            COALESCE(MAX(CASE WHEN feature_name = 'color_moments' THEN gamma END), 1.0) AS color_moments,
            COALESCE(MAX(CASE WHEN feature_name = 'vein_features' THEN gamma END), 1.0) AS vein_features
        FROM search_feature_gamma
        """
    )
    return cur.fetchone() or {}

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
    w_morphology = float(request.form.get("w_morphology", 0.0))
    w_texture = float(request.form.get("w_texture", 0.3))
    w_color = float(request.form.get("w_color", 0.2))
    w_vein = float(request.form.get("w_vein", 0.1))
    top_k = int(request.form.get("top_k", 10))

    try:
        conn = connect_db()
        register_vector(conn)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query_vectors = {
            "efd_coeffs": _to_vector(feats["efd_coeffs"]),
            "morphology_stats": _to_vector(feats["morphology_stats"]),
            "lbp_hist": _to_vector(feats["lbp_hist"]),
            "glcm_stats": _to_vector(feats["glcm_stats"]),
            "color_moments": _to_vector(feats["color_moments"]),
            "vein_features": _to_vector(feats["vein_features"]),
        }

        sql = """
            WITH q AS (
                SELECT
                    %s::vector AS q_efd,
                    %s::vector AS q_morphology,
                    %s::vector AS q_lbp,
                    %s::vector AS q_glcm,
                    %s::vector AS q_color,
                    %s::vector AS q_vein
            ), gamma AS (
                SELECT
                    COALESCE(MAX(CASE WHEN feature_name = 'efd_coeffs' THEN gamma END), 1.0) AS efd_gamma,
                    COALESCE(MAX(CASE WHEN feature_name = 'morphology_stats' THEN gamma END), 1.0) AS morphology_gamma,
                    COALESCE(MAX(CASE WHEN feature_name = 'lbp_hist' THEN gamma END), 1.0) AS lbp_gamma,
                    COALESCE(MAX(CASE WHEN feature_name = 'glcm_stats' THEN gamma END), 1.0) AS glcm_gamma,
                    COALESCE(MAX(CASE WHEN feature_name = 'color_moments' THEN gamma END), 1.0) AS color_gamma,
                    COALESCE(MAX(CASE WHEN feature_name = 'vein_features' THEN gamma END), 1.0) AS vein_gamma
                FROM search_feature_gamma
            )
            SELECT
                lc.filename,
                lc.species,
                (
                    (%s * exp(-g.efd_gamma * (lc.efd_coeffs <-> q.q_efd))) +
                    (%s * exp(-g.morphology_gamma * (lc.morphology_stats <-> q.q_morphology))) +
                    (%s * exp(-g.lbp_gamma * chi_square_dist(lc.lbp_hist, q.q_lbp))) +
                    (%s * exp(-g.glcm_gamma * (lc.glcm_stats <-> q.q_glcm))) +
                    (%s * exp(-g.color_gamma * (lc.color_moments <-> q.q_color))) +
                    (%s * exp(-g.vein_gamma * (lc.vein_features <-> q.q_vein)))
                ) / NULLIF((%s + %s + %s + %s + %s + %s), 0) AS similarity
            FROM leaf_collection lc
            CROSS JOIN q
            CROSS JOIN gamma g
            ORDER BY similarity DESC
            LIMIT %s
        """

        params = [
            query_vectors["efd_coeffs"],
            query_vectors["morphology_stats"],
            query_vectors["lbp_hist"],
            query_vectors["glcm_stats"],
            query_vectors["color_moments"],
            query_vectors["vein_features"],
            w_efd,
            w_morphology,
            w_texture * 0.5,
            w_texture * 0.5,
            w_color,
            w_vein,
            w_efd,
            w_morphology,
            w_texture * 0.5,
            w_texture * 0.5,
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
        if 'conn' in locals() and conn: conn.close()