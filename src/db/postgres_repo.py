import psycopg2

def connect_db():
    return psycopg2.connect(host="db", port=5432, dbname="leaf_db", user="admin", password="admin")

def insert_pg(conn, filename, path, feats, species=None):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO leaf_collection
        (filename, image_path, species, efd, morphology, lbp, glcm, color, vein)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE SET
            image_path = EXCLUDED.image_path,
            species    = EXCLUDED.species,
            efd        = EXCLUDED.efd,
            morphology = EXCLUDED.morphology,
            lbp        = EXCLUDED.lbp,
            glcm       = EXCLUDED.glcm,
            color      = EXCLUDED.color,
            vein       = EXCLUDED.vein
    """,
    (
        filename,
        path,
        species,
        feats["efd"].tolist(),
        feats["morphology"].tolist(),
        feats["lbp"].tolist(),
        feats["glcm"].tolist(),
        feats["color"].tolist(),
        feats["vein"].tolist()
    ))

    conn.commit()