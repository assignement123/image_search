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

SEARCH_FEATURES = [
    ("efd", "EFD", "w_efd"),
    ("morphology", "Morphology", "w_morphology"),
    ("lbp", "LBP", "w_lbp"),
    ("glcm", "GLCM", "w_glcm"),
    ("color", "Color", "w_color"),
    ("vein", "Vein", "w_vein"),
]


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


def _parse_search_weights(form) -> dict:
    w_efd = float(form.get("w_efd", 0.4))
    w_morphology = float(form.get("w_morphology", 0.0))
    w_texture = form.get("w_texture")
    w_texture = float(w_texture) if w_texture is not None else 0.3
    w_lbp = form.get("w_lbp")
    w_lbp = float(w_lbp) if w_lbp is not None else w_texture * 0.5
    w_glcm = form.get("w_glcm")
    w_glcm = float(w_glcm) if w_glcm is not None else w_texture * 0.5
    w_color = float(form.get("w_color", 0.2))
    w_vein = float(form.get("w_vein", 0.1))
    top_k = int(form.get("top_k", 10))

    return {
        "w_efd": w_efd,
        "w_morphology": w_morphology,
        "w_lbp": w_lbp,
        "w_glcm": w_glcm,
        "w_color": w_color,
        "w_vein": w_vein,
        "top_k": top_k,
        "sum_weights": w_efd + w_morphology + w_lbp + w_glcm + w_color + w_vein,
    }


def _build_query_vectors(feats: dict) -> dict:
    return {
        "efd_coeffs": _to_vector(feats["efd_coeffs"]),
        "morphology_stats": _to_vector(feats["morphology_stats"]),
        "lbp_hist": _to_vector(feats["lbp_hist"]),
        "glcm_stats": _to_vector(feats["glcm_stats"]),
        "color_moments": _to_vector(feats["color_moments"]),
        "vein_features": _to_vector(feats["vein_features"]),
    }


def _fetch_search_debug_rows(cur, query_vectors: dict, weights: dict) -> list:
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
            g.efd_gamma,
            g.morphology_gamma,
            g.lbp_gamma,
            g.glcm_gamma,
            g.color_gamma,
            g.vein_gamma,
            (lc.efd_coeffs <-> q.q_efd) AS efd_dist,
            (lc.morphology_stats <-> q.q_morphology) AS morphology_dist,
            chi_square_dist(lc.lbp_hist, q.q_lbp) AS lbp_dist,
            (lc.glcm_stats <-> q.q_glcm) AS glcm_dist,
            (lc.color_moments <-> q.q_color) AS color_dist,
            (lc.vein_features <-> q.q_vein) AS vein_dist,
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
        weights["w_efd"],
        weights["w_morphology"],
        weights["w_lbp"],
        weights["w_glcm"],
        weights["w_color"],
        weights["w_vein"],
        weights["w_efd"],
        weights["w_morphology"],
        weights["w_lbp"],
        weights["w_glcm"],
        weights["w_color"],
        weights["w_vein"],
        weights["top_k"],
    ]

    cur.execute(sql, params)
    return cur.fetchall()

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
        top_k = int(request.form.get("top_k", 10))

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
        print("\n❌ LỖI TẠI API SEARCH:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn: conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@search_bp.route("/api/search/debug", methods=["POST"])
def search_debug():
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    conn = None
    try:
        feats = process_single_image(tmp_path)
        weights = _parse_search_weights(request.form)
        query_vectors = _build_query_vectors(feats)

        conn = connect_db()
        register_vector(conn)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        rows = _fetch_search_debug_rows(cur, query_vectors, weights)
        results = []

        for row in rows:
            breakdown = []
            for feature_key, label, weight_key in SEARCH_FEATURES:
                weight = float(weights[weight_key])
                gamma = float(row[f"{feature_key}_gamma"] or 1.0)
                distance = float(row[f"{feature_key}_dist"] or 0.0)
                score = float(np.exp(-gamma * distance))
                contribution = (weight * score / weights["sum_weights"] * 100.0) if weights["sum_weights"] else 0.0
                breakdown.append({
                    "key": feature_key,
                    "label": label,
                    "weight": round(weight, 4),
                    "gamma": round(gamma, 6),
                    "distance": round(distance, 6),
                    "score": round(score, 6),
                    "contribution": round(contribution, 2),
                })

            results.append({
                "filename": row["filename"],
                "species": row["species"],
                "image_url": f"/api/image/{row['filename']}",
                "similarity": round(float(row["similarity"]) * 100, 2) if row["similarity"] is not None else 0.0,
                "breakdown": breakdown,
            })

        return jsonify({
            "count": len(results),
            "top_k": weights["top_k"],
            "meta": {
                "weights": {k: round(v, 4) for k, v in weights.items() if k.startswith("w_")},
                "sum_weights": round(weights["sum_weights"], 4),
            },
            "results": results,
        })
    except Exception as e:
        print("\n❌ LỖI TẠI API SEARCH DEBUG:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
