# src/routes/search_cascaded.py
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import psycopg2.extras
from flask import Blueprint, request, jsonify

from src.db.postgres_repo import connect_db
from src.pipeline import process_single_image

search_cascaded_bp = Blueprint('search_cascaded', __name__)


def _to_vector(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    return value


CASCADED_SQL = """
WITH
s1 AS (
    -- Tầng 1: EFD hình dạng, có reflection invariance
    -- LEAST → lấy distance nhỏ nhất giữa normal và flipped query
    SELECT filename, species,
           morphology_stats, lbp_hist, glcm_stats, vein_features, color_moments,
           LEAST(
               efd_coeffs <=> %(q_efd)s::vector,
               efd_coeffs <=> %(q_efd_flip)s::vector
           ) AS d_efd
    FROM   leaf_collection
    ORDER  BY d_efd
    LIMIT  %(top1)s
),
s2 AS (
    -- Tầng 2: Morphology — cosine (vector raw: aspect_ratio, circularity, solidity)
    SELECT *, morphology_stats <=> %(q_morph)s::vector AS d_morph
    FROM   s1
    ORDER  BY d_morph
    LIMIT  %(top2)s
),
s3 AS (
    -- Tầng 3: LBP — chi-square (metric tối ưu cho probability histogram)
    SELECT *, chi_square_dist(lbp_hist, %(q_lbp)s::vector) AS d_lbp
    FROM   s2
    ORDER  BY d_lbp
    LIMIT  %(top3)s
),
s4 AS (
    -- Tầng 4: GLCM texture — cosine trên vector L2-normalized
    SELECT *, glcm_stats <=> %(q_glcm)s::vector AS d_glcm
    FROM   s3
    ORDER  BY d_glcm
    LIMIT  %(top4)s
),
s5 AS (
    -- Tầng 5: Vein gân lá — cosine trên vector L2-normalized
    SELECT *, vein_features <=> %(q_vein)s::vector AS d_vein
    FROM   s4
    ORDER  BY d_vein
    LIMIT  %(top5)s
),
s6 AS (
    -- Tầng 6: Color moments — cosine trên vector L2-normalized
    SELECT *, color_moments <=> %(q_color)s::vector AS d_color
    FROM   s5
    ORDER  BY d_color
    LIMIT  %(top6)s
)
SELECT
    filename,
    species,
    '/api/image/' || filename AS image_url,
    d_efd, d_morph, d_lbp, d_glcm, d_vein, d_color
FROM s6
ORDER BY d_color
"""


@search_cascaded_bp.route("/api/search/cascaded", methods=["POST"])
def search_cascaded():
    """
    Cascaded / funnel search – 6 features, mỗi tầng 1 feature:
      Stage 1 – top {top1} by EFD (hình dạng)
      Stage 2 – top {top2} by Morphology (hình thái)
      Stage 3 – top {top3} by LBP (texture cục bộ)
      Stage 4 – top {top4} by GLCM (texture thống kê)
      Stage 5 – top {top5} by Vein (gân lá)
      Stage 6 – top {top6} by Color moments (màu sắc)
    """
    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh"}), 400

    file   = request.files["file"]
    suffix = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = process_single_image(tmp_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 422
    finally:
        os.unlink(tmp_path)

    # ── Funnel sizes ────────────────────────────────────────────────
    top1 = int(request.form.get("top1", 200))
    top2 = int(request.form.get("top2", 100))
    top3 = int(request.form.get("top3",  50))
    top4 = int(request.form.get("top4",  30))
    top5 = int(request.form.get("top5",  20))
    top6 = int(request.form.get("top6",  10))

    q = {k: _to_vector(v) for k, v in feats.items()}

    params = dict(
        q_efd      = q["efd_coeffs"],
        q_efd_flip = q["efd_coeffs_flipped"],   # reflection invariance cho EFD
        q_morph    = q["morphology_stats"],
        q_lbp      = q["lbp_hist"],
        q_glcm     = q["glcm_stats"],
        q_vein     = q["vein_features"],
        q_color    = q["color_moments"],
        top1=top1, top2=top2, top3=top3,
        top4=top4, top5=top5, top6=top6,
    )

    try:
        conn = connect_db()   # register_vector đã có trong connect_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(CASCADED_SQL, params)
        rows = cur.fetchall()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cur'  in locals(): cur.close()
        if 'conn' in locals() and conn: conn.close()

    results = [
        {
            "filename":  r["filename"],
            "species":   r["species"],
            "image_url": r["image_url"],
            "debug": {
                "d_efd":   round(float(r["d_efd"]),   4) if r.get("d_efd")   else None,
                "d_morph": round(float(r["d_morph"]), 4) if r.get("d_morph") else None,
                "d_lbp":   round(float(r["d_lbp"]),   4) if r.get("d_lbp")   else None,
                "d_glcm":  round(float(r["d_glcm"]),  4) if r.get("d_glcm")  else None,
                "d_vein":  round(float(r["d_vein"]),  4) if r.get("d_vein")  else None,
                "d_color": round(float(r["d_color"]), 4) if r.get("d_color") else None,
            }
        }
        for r in rows
    ]

    return jsonify({
        "results": results,
        "count":   len(results),
        "funnel":  {
            "top1": top1, "top2": top2, "top3": top3,
            "top4": top4, "top5": top5, "top6": top6,
        },
    })