# src/routes/search.py
import ast
import json
import math
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

PARAMS_PATH = Path(__file__).resolve().parents[1] / "normalization_params.json"


def _load_params() -> dict:
    if not PARAMS_PATH.exists():
        return {}
    try:
        with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("params", {})
    except Exception:
        return {}


def _to_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except Exception:
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return None
        if isinstance(parsed, (list, tuple)):
            return np.asarray(parsed, dtype=np.float32)
    return None


def _euclidean_distance(v1, v2) -> float:
    return float(np.linalg.norm(v1 - v2))


def _chi_square_distance(v1, v2) -> float:
    return float(0.5 * np.sum(((v1 - v2) ** 2) / (v1 + v2 + 1e-10)))


def _score_from_distance(distance: float, gamma: float | None) -> float:
    gamma = 1.0 if gamma is None else float(gamma)
    return float(math.exp(-gamma * max(distance, 0.0)))

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
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT filename, species, efd_coeffs, morphology_stats, lbp_hist, glcm_stats, color_moments, vein_features
            FROM leaf_collection
        """)

        rows = cur.fetchall()
        params = _load_params()

        gamma_map = {
            "efd_coeffs": params.get("efd_coeffs", {}).get("gamma"),
            "morphology_stats": params.get("morphology_stats", {}).get("gamma"),
            "lbp_hist": params.get("lbp_hist", {}).get("gamma"),
            "glcm_stats": params.get("glcm_stats", {}).get("gamma"),
            "color_moments": params.get("color_moments", {}).get("gamma"),
            "vein_features": params.get("vein_features", {}).get("gamma"),
        }

        query_vectors = {
            "efd_coeffs": _to_vector(feats["efd_coeffs"]),
            "morphology_stats": _to_vector(feats["morphology_stats"]),
            "lbp_hist": _to_vector(feats["lbp_hist"]),
            "glcm_stats": _to_vector(feats["glcm_stats"]),
            "color_moments": _to_vector(feats["color_moments"]),
            "vein_features": _to_vector(feats["vein_features"]),
        }

        scored_rows = []
        for row in rows:
            row_vectors = {
                "efd_coeffs": _to_vector(row.get("efd_coeffs")),
                "morphology_stats": _to_vector(row.get("morphology_stats")),
                "lbp_hist": _to_vector(row.get("lbp_hist")),
                "glcm_stats": _to_vector(row.get("glcm_stats")),
                "color_moments": _to_vector(row.get("color_moments")),
                "vein_features": _to_vector(row.get("vein_features")),
            }

            feature_scores = {}
            if row_vectors["efd_coeffs"] is not None and query_vectors["efd_coeffs"] is not None:
                d = _euclidean_distance(row_vectors["efd_coeffs"], query_vectors["efd_coeffs"])
                feature_scores["efd"] = _score_from_distance(d, gamma_map["efd_coeffs"])

            if row_vectors["morphology_stats"] is not None and query_vectors["morphology_stats"] is not None:
                d = _euclidean_distance(row_vectors["morphology_stats"], query_vectors["morphology_stats"])
                feature_scores["morphology"] = _score_from_distance(d, gamma_map["morphology_stats"])

            if row_vectors["lbp_hist"] is not None and query_vectors["lbp_hist"] is not None:
                d = _chi_square_distance(row_vectors["lbp_hist"], query_vectors["lbp_hist"])
                feature_scores["lbp"] = _score_from_distance(d, gamma_map["lbp_hist"])

            if row_vectors["glcm_stats"] is not None and query_vectors["glcm_stats"] is not None:
                d = _euclidean_distance(row_vectors["glcm_stats"], query_vectors["glcm_stats"])
                feature_scores["glcm"] = _score_from_distance(d, gamma_map["glcm_stats"])

            if row_vectors["color_moments"] is not None and query_vectors["color_moments"] is not None:
                d = _euclidean_distance(row_vectors["color_moments"], query_vectors["color_moments"])
                feature_scores["color"] = _score_from_distance(d, gamma_map["color_moments"])

            if row_vectors["vein_features"] is not None and query_vectors["vein_features"] is not None:
                d = _euclidean_distance(row_vectors["vein_features"], query_vectors["vein_features"])
                feature_scores["vein"] = _score_from_distance(d, gamma_map["vein_features"])

            total_similarity = 0.0
            total_weight = 0.0

            if "efd" in feature_scores:
                total_similarity += w_efd * feature_scores["efd"]
                total_weight += w_efd

            if "morphology" in feature_scores:
                total_similarity += w_morphology * feature_scores["morphology"]
                total_weight += w_morphology

            if "lbp" in feature_scores:
                total_similarity += (w_texture * 0.5) * feature_scores["lbp"]
                total_weight += (w_texture * 0.5)

            if "glcm" in feature_scores:
                total_similarity += (w_texture * 0.5) * feature_scores["glcm"]
                total_weight += (w_texture * 0.5)

            if "color" in feature_scores:
                total_similarity += w_color * feature_scores["color"]
                total_weight += w_color

            if "vein" in feature_scores:
                total_similarity += w_vein * feature_scores["vein"]
                total_weight += w_vein

            if total_weight > 0:
                similarity = total_similarity / total_weight
            else:
                similarity = 0.0

            scored_rows.append({
                "filename": row["filename"],
                "species": row["species"],
                "similarity": similarity,
            })

        scored_rows.sort(key=lambda item: item["similarity"], reverse=True)
        rows = scored_rows[:top_k]

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