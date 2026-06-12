CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS leaf_collection;

CREATE TABLE leaf_collection (
    id                  SERIAL PRIMARY KEY,
    filename            VARCHAR(255) UNIQUE NOT NULL,
    image_path          TEXT NOT NULL,
    species             TEXT,
    
    efd_coeffs          vector(57) NOT NULL,
    morphology_stats    vector(4)  NOT NULL,
    
    lbp_hist            vector(26) NOT NULL,
    glcm_stats          vector(20) NOT NULL,
    
    color_moments       vector(9)  NOT NULL,
    vein_features       vector(17) NOT NULL,
    
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR(50) PRIMARY KEY,
    value TEXT
);

INSERT INTO meta (key, value) VALUES
    ('harmonics',      '15'),
    ('n_resample',     '600'),
    ('glcm_levels',    '64'),
    ('dim_efd',        '57'),
    ('dim_morphology', '4'),
    ('dim_lbp',        '26'),
    ('dim_glcm',       '20'),
    ('dim_color',      '9'),
    ('dim_vein',       '17')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

-- Dùng vector_cosine_ops để khớp với operator <=> (cosine distance) trong search.py
CREATE INDEX idx_efd_hnsw   ON leaf_collection USING hnsw (efd_coeffs       vector_cosine_ops);
CREATE INDEX idx_morph_hnsw ON leaf_collection USING hnsw (morphology_stats vector_l2_ops);
CREATE INDEX idx_glcm_hnsw  ON leaf_collection USING hnsw (glcm_stats       vector_l2_ops);
CREATE INDEX idx_vein_hnsw  ON leaf_collection USING hnsw (vein_features     vector_cosine_ops);
CREATE INDEX idx_lbp_hnsw   ON leaf_collection USING hnsw (lbp_hist          vector_cosine_ops);
CREATE INDEX idx_color_hnsw ON leaf_collection USING hnsw (color_moments     vector_cosine_ops);

CREATE OR REPLACE FUNCTION chi_square_dist(vec1 vector, vec2 vector)
RETURNS float8 AS $$
DECLARE
    s1 text := regexp_replace(vec1::text, '[\[\]\s]', '', 'g');
    s2 text := regexp_replace(vec2::text, '[\[\]\s]', '', 'g');
    arr1 float8[] := CASE WHEN s1 = '' THEN ARRAY[]::float8[] ELSE string_to_array(s1, ',')::float8[] END;
    arr2 float8[] := CASE WHEN s2 = '' THEN ARRAY[]::float8[] ELSE string_to_array(s2, ',')::float8[] END;
    dim int := array_length(arr1, 1);
    distance float8 := 0.0;
    diff float8;
    sum_val float8;
BEGIN
    IF dim IS NULL OR dim != array_length(arr2, 1) THEN
        RAISE EXCEPTION 'Số chiều của hai vector không khớp nhau!';
    END IF;

    FOR i IN 1..dim LOOP
        sum_val := arr1[i] + arr2[i];
        IF sum_val > 1e-9 THEN
            diff := arr1[i] - arr2[i];
            distance := distance + ((diff * diff) / sum_val);
        END IF;
    END LOOP;

    RETURN distance;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;
