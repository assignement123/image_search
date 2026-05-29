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


def _to_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return value


def _to_float_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).tolist()
    return np.asarray(value, dtype=np.float32).tolist()


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
    w_texture = request.form.get("w_texture")
    w_texture = float(w_texture) if w_texture is not None else 0.3
    w_lbp = request.form.get("w_lbp")
    w_lbp = float(w_lbp) if w_lbp is not None else w_texture * 0.5
    w_glcm = request.form.get("w_glcm")
    w_glcm = float(w_glcm) if w_glcm is not None else w_texture * 0.5
    w_color = float(request.form.get("w_color", 0.2))
    w_vein = float(request.form.get("w_vein", 0.1))
    explain = _to_bool(request.form.get("explain", "false"))
    top_k = int(request.form.get("top_k", 10))
    total_weight = w_efd + w_morphology + w_lbp + w_glcm + w_color + w_vein
    lbp_weight_ratio = (w_lbp / total_weight) if total_weight > 0 else 0.0
    glcm_weight_ratio = (w_glcm / total_weight) if total_weight > 0 else 0.0

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
            ), scored AS (
                SELECT
                    lc.filename,
                    lc.species,
                    lc.lbp_hist AS lbp_vec,
                    lc.glcm_stats AS glcm_vec,
                    (lc.efd_coeffs <-> q.q_efd) AS efd_dist,
                    (lc.morphology_stats <-> q.q_morphology) AS morphology_dist,
                    chi_square_dist(lc.lbp_hist, q.q_lbp) AS lbp_dist,
                    (lc.glcm_stats <-> q.q_glcm) AS glcm_dist,
                    (lc.color_moments <-> q.q_color) AS color_dist,
                    (lc.vein_features <-> q.q_vein) AS vein_dist,
                    g.lbp_gamma,
                    g.glcm_gamma,
                    exp(-g.efd_gamma * (lc.efd_coeffs <-> q.q_efd)) AS efd_score,
                    exp(-g.morphology_gamma * (lc.morphology_stats <-> q.q_morphology)) AS morphology_score,
                    exp(-g.lbp_gamma * chi_square_dist(lc.lbp_hist, q.q_lbp)) AS lbp_score,
                    exp(-g.glcm_gamma * (lc.glcm_stats <-> q.q_glcm)) AS glcm_score,
                    exp(-g.color_gamma * (lc.color_moments <-> q.q_color)) AS color_score,
                    exp(-g.vein_gamma * (lc.vein_features <-> q.q_vein)) AS vein_score
                FROM leaf_collection lc
                CROSS JOIN q
                CROSS JOIN gamma g
            )
            SELECT
                s.filename,
                s.species,
                (
                    (%s * s.efd_score) +
                    (%s * s.morphology_score) +
                    (%s * s.lbp_score) +
                    (%s * s.glcm_score) +
                    (%s * s.color_score) +
                    (%s * s.vein_score)
                ) / NULLIF((%s + %s + %s + %s + %s + %s), 0) AS similarity
                ,s.lbp_dist
                ,s.glcm_dist
                ,s.lbp_gamma
                ,s.glcm_gamma
                ,s.lbp_score
                ,s.glcm_score
                ,s.lbp_vec
                ,s.glcm_vec
                ,(%s * s.lbp_score) AS lbp_weighted
                ,(%s * s.glcm_score) AS glcm_weighted
            FROM scored s
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
            w_lbp,
            w_glcm,
            top_k,
        ]

        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            item = {
                "filename": r["filename"],
                "species": r["species"],
                "similarity": round(float(r["similarity"]) * 100, 2) if r["similarity"] is not None else 0.0,
                "image_url": f"/api/image/{r['filename']}"
            }
            if explain:
                item["explain"] = {
                    "texture": {
                        "lbp": {
                            "metric": "chi_square",
                            "distance": float(r["lbp_dist"]),
                            "gamma": float(r["lbp_gamma"]),
                            "score": float(r["lbp_score"]),
                            "weighted_contribution": float(r["lbp_weighted"]),
                            "weight": float(w_lbp),
                            "weight_ratio": float(lbp_weight_ratio),
                            "vector": _to_float_list(r["lbp_vec"]),
                        },
                        "glcm": {
                            "metric": "l2",
                            "distance": float(r["glcm_dist"]),
                            "gamma": float(r["glcm_gamma"]),
                            "score": float(r["glcm_score"]),
                            "weighted_contribution": float(r["glcm_weighted"]),
                            "weight": float(w_glcm),
                            "weight_ratio": float(glcm_weight_ratio),
                            "vector": _to_float_list(r["glcm_vec"]),
                        },
                    }
                }
            results.append(item)

        payload = {"results": results, "count": len(results)}
        if explain:
            payload["explain_meta"] = {
                "weights": {
                    "w_efd": float(w_efd),
                    "w_morphology": float(w_morphology),
                    "w_lbp": float(w_lbp),
                    "w_glcm": float(w_glcm),
                    "w_color": float(w_color),
                    "w_vein": float(w_vein),
                },
                "query_texture": {
                    "lbp": _to_float_list(query_vectors["lbp_hist"]),
                    "glcm": _to_float_list(query_vectors["glcm_stats"]),
                },
                "total_weight": float(total_weight),
            }

        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()
