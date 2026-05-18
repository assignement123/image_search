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


