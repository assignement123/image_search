CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS leaf_collection;

CREATE TABLE leaf_collection (
    id                  SERIAL PRIMARY KEY,
    filename            VARCHAR(255) UNIQUE NOT NULL,
    image_path          TEXT NOT NULL,
    species             TEXT,
    
    efd_coeffs          vector(76) NOT NULL,
    morphology_stats    vector(3)  NOT NULL,
    
    lbp_hist            vector(26) NOT NULL,
    glcm_stats          vector(20) NOT NULL,
    
    color_moments       vector(9)  NOT NULL,
    vein_features       vector(9)  NOT NULL,
    
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR(50) PRIMARY KEY,
    value TEXT
);

INSERT INTO meta (key, value) VALUES
    ('harmonics',      '20'),
    ('n_resample',     '600'),
    ('glcm_levels',    '64'),
    ('dim_efd',        '76'),
    ('dim_morphology', '3'),
    ('dim_lbp',        '26'),
    ('dim_glcm',       '20'),
    ('dim_color',      '9'),
    ('dim_vein',       '9')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

CREATE INDEX idx_efd_hnsw ON leaf_collection USING hnsw (efd_coeffs vector_l2_ops);
CREATE INDEX idx_morph_hnsw ON leaf_collection USING hnsw (morphology_stats vector_l2_ops);
CREATE INDEX idx_glcm_hnsw ON leaf_collection USING hnsw (glcm_stats vector_l2_ops);
CREATE INDEX idx_vein_hnsw ON leaf_collection USING hnsw (vein_features vector_l2_ops);

CREATE INDEX idx_lbp_hnsw ON leaf_collection USING hnsw (lbp_hist vector_l2_ops);
CREATE INDEX idx_color_hnsw ON leaf_collection USING hnsw (color_moments vector_l2_ops);

CREATE OR REPLACE FUNCTION chi_square_dist(vec1 vector, vec2 vector)
RETURNS float8 AS $$
DECLARE
    arr1 float8[] := vector_to_float8_array(vec1);
    arr2 float8[] := vector_to_float8_array(vec2);
    dim int := array_length(arr1, 1);
    distance float8 := 0.0;
    diff float8;
    sum_val float8;
BEGIN
    IF dim != array_length(arr2, 1) THEN
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
