import psycopg2

def connect_db():
    return psycopg2.connect(host="db", port=5432, dbname="leaf_db", user="admin", password="admin")

def insert_pg(conn, filename, path, feats, species=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO leaf_collection
        (filename, image_path, species, efd_coeffs, morphology_stats, lbp_hist, glcm_stats, color_moments, vein_features)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE SET
            image_path       = EXCLUDED.image_path,
            species          = EXCLUDED.species,
            efd_coeffs       = EXCLUDED.efd_coeffs,
            morphology_stats = EXCLUDED.morphology_stats,
            lbp_hist         = EXCLUDED.lbp_hist,
            glcm_stats       = EXCLUDED.glcm_stats,
            color_moments    = EXCLUDED.color_moments,
            vein_features    = EXCLUDED.vein_features
    """,
    (
        filename, path, species,
        feats["efd_coeffs"].tolist(),
        feats["morphology_stats"].tolist(),
        feats["lbp_hist"].tolist(),
        feats["glcm_stats"].tolist(),
        feats["color_moments"].tolist(),
        feats["vein_features"].tolist()
    ))
    conn.commit()