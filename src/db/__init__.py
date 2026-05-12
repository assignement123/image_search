DROP TABLE IF EXISTS leaf_collection;

CREATE TABLE IF NOT EXISTS leaf_collection (
    id          SERIAL PRIMARY KEY,
    filename    VARCHAR(255) UNIQUE NOT NULL,
    image_path  TEXT         NOT NULL,
    species     TEXT,
    efd         vector(76)   NOT NULL,
    morphology  vector(3)    NOT NULL,
    lbp         vector(26)   NOT NULL,
    glcm        vector(20)   NOT NULL,
    color       vector(9)    NOT NULL,
    vein        vector(9)    NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# -- Index Euclidean (<->) cho EFD, Morphology, GLCM, Vein
CREATE INDEX ON leaf_collection USING hnsw (efd vector_l2_ops);
CREATE INDEX ON leaf_collection USING hnsw (morphology vector_l2_ops);
CREATE INDEX ON leaf_collection USING hnsw (glcm vector_l2_ops);
CREATE INDEX ON leaf_collection USING hnsw (vein vector_l2_ops);

# -- Index Euclidean tạm cho LBP, Color (Vì pgvector không hỗ trợ index Chi-Square native)
CREATE INDEX ON leaf_collection USING hnsw (lbp vector_l2_ops);
CREATE INDEX ON leaf_collection USING hnsw (color vector_l2_ops);