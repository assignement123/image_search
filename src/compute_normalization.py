import os
import json
import math
import random
import psycopg2.extras
import numpy as np
from collections import defaultdict
from src.db.postgres_repo import connect_db

# Cấu hình
TARGET_SCORE = float(os.getenv("TARGET_SCORE", 0.2))
NUM_SAMPLE_PAIRS = int(os.getenv("NUM_SAMPLE_PAIRS", 3000))
OUT_FILE = os.path.join(os.path.dirname(__file__), "normalization_params.json")


def chi_square_distance(v1, v2):
    v1, v2 = np.array(v1, dtype=np.float64), np.array(v2, dtype=np.float64)
    return 0.5 * np.nansum(((v1 - v2) ** 2) / (v1 + v2 + 1e-10))


def euclidean_distance(v1, v2):
    return float(np.linalg.norm(np.array(v1, dtype=np.float64) - np.array(v2, dtype=np.float64)))


def safe_parse(val):
    # Chuẩn hoá đầu vào: nếu DB trả string, cố parse JSON/AST
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, (bytes, bytearray)):
        try:
            s = val.decode("utf-8")
        except Exception:
            return None
        val = s
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("["):
            try:
                return json.loads(val)
            except Exception:
                import ast
                try:
                    return list(ast.literal_eval(val))
                except Exception:
                    return None
        else:
            # not an array-like string
            return None
    return None


def gather_features(rows):
    features = defaultdict(list)
    species_dict = defaultdict(list)

    vector_cols = [
        "efd_coeffs", "morphology_stats", "lbp_hist",
        "glcm_stats", "color_moments", "vein_features"
    ]

    for row in rows:
        species = row.get("species") or "__unknown__"
        parsed = {}
        for col in vector_cols:
            parsed[col] = None
            if col in row:
                v = row[col]
                pv = safe_parse(v)
                # if safe_parse returned None but v is list already, accept
                if pv is None and isinstance(v, (list, tuple)):
                    pv = list(v)
                parsed[col] = pv
                if pv is not None:
                    features[col].append(pv)
        species_dict[species].append(parsed)

    return features, species_dict


def compute_mean_std(arrs):
    # arrs: list of list (same dim)
    if not arrs:
        return None, None
    mat = np.vstack([np.array(a, dtype=np.float64) for a in arrs])
    mean = np.mean(mat, axis=0).tolist()
    std = np.std(mat, axis=0).tolist()
    # avoid zeros in std
    std = [s if s > 1e-6 else 1.0 for s in std]
    return mean, std


def estimate_gammas(species_dict):
    species_list = list(species_dict.keys())
    if len(species_list) < 2:
        raise RuntimeError("Cần ít nhất 2 loài trong DB để estimate gamma")

    distances = {"efd": [], "morph": [], "lbp": [], "glcm": [], "color": [], "vein": []}

    pairs = min(NUM_SAMPLE_PAIRS, 100000)
    for _ in range(pairs):
        sp1, sp2 = random.sample(species_list, 2)
        leaf1 = random.choice(species_dict[sp1])
        leaf2 = random.choice(species_dict[sp2])

        try:
            if leaf1["efd_coeffs"] is not None and leaf2["efd_coeffs"] is not None:
                distances["efd"].append(euclidean_distance(leaf1["efd_coeffs"], leaf2["efd_coeffs"]))
            if leaf1["morphology_stats"] is not None and leaf2["morphology_stats"] is not None:
                distances["morph"].append(euclidean_distance(leaf1["morphology_stats"], leaf2["morphology_stats"]))
            if leaf1["glcm_stats"] is not None and leaf2["glcm_stats"] is not None:
                distances["glcm"].append(euclidean_distance(leaf1["glcm_stats"], leaf2["glcm_stats"]))
            if leaf1["color_moments"] is not None and leaf2["color_moments"] is not None:
                distances["color"].append(euclidean_distance(leaf1["color_moments"], leaf2["color_moments"]))
            if leaf1["vein_features"] is not None and leaf2["vein_features"] is not None:
                distances["vein"].append(euclidean_distance(leaf1["vein_features"], leaf2["vein_features"]))
            if leaf1["lbp_hist"] is not None and leaf2["lbp_hist"] is not None:
                distances["lbp"].append(chi_square_distance(leaf1["lbp_hist"], leaf2["lbp_hist"]))
        except Exception:
            continue

    gammas = {}
    for k, vals in distances.items():
        if not vals:
            gammas[k] = None
            continue
        avg_d = float(np.mean(vals))
        if avg_d <= 1e-9:
            gammas[k] = 1.0
        else:
            gammas[k] = float(-math.log(TARGET_SCORE) / avg_d)
    return gammas


def main():
    print("⏳ Kết nối DB và tải dữ liệu để tính normalization parameters...")
    conn = connect_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT species, efd_coeffs, morphology_stats, lbp_hist, glcm_stats, color_moments, vein_features FROM leaf_collection")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("❌ Database rỗng. Chạy ETL trước khi tính normalization parameters.")
        return

    features, species_dict = gather_features(rows)

    params = {}

    # morphology
    mean_morph, std_morph = compute_mean_std(features.get("morphology_stats", []))
    params["morphology_stats"] = {
        "method": "z_l2",
        "mean": mean_morph,
        "std": std_morph,
        "gamma": None
    }

    # color
    mean_color, std_color = compute_mean_std(features.get("color_moments", []))
    params["color_moments"] = {
        "method": "z_l2",
        "mean": mean_color,
        "std": std_color,
        "gamma": None
    }

    # efd (suggest l2)
    params["efd_coeffs"] = {"method": "l2", "gamma": None}

    # glcm already l2 in extractor
    params["glcm_stats"] = {"method": "l2", "gamma": None}

    # lbp histogram (l1)
    params["lbp_hist"] = {"method": "l1", "gamma": None}

    # vein: convert hist to l1 + then l2 normalize whole vector
    params["vein_features"] = {"method": "l1_then_l2", "gamma": None}

    # estimate gamma values
    print("⏳ Đang ước lượng gamma từ dữ liệu...")
    gammas = estimate_gammas(species_dict)

    params_map = {
        "efd_coeffs": "efd",
        "morphology_stats": "morph",
        "lbp_hist": "lbp",
        "glcm_stats": "glcm",
        "color_moments": "color",
        "vein_features": "vein",
    }

    for k, short in params_map.items():
        params[k]["gamma"] = gammas.get(short)

    # save
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"target_score": TARGET_SCORE, "params": params}, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã xuất normalization params → {OUT_FILE}")


if __name__ == "__main__":
    main()
