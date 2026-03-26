
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS leaf_collection (
    id          SERIAL PRIMARY KEY,
    filename    VARCHAR(255) UNIQUE NOT NULL,   -- "1027.jpg"
    image_path  TEXT         NOT NULL,           -- đường dẫn đầy đủ
    species     TEXT,                            -- tên loài từ folder cha (vd: "Phyllostachys_edulis")
    efd         vector(76)   NOT NULL,
    texture        vector(46)   NOT NULL,
    color       vector(9)    NOT NULL,
    vein        vector(9)    NOT NULL,
    

    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index để tăng tốc tìm kiếm vector (HNSW — tốt hơn IVFFlat cho dataset nhỏ/vừa)
-- Dùng cosine distance vì đặc trưng không cùng đơn vị
CREATE INDEX IF NOT EXISTS idx_efd_cosine
    ON leaf_collection USING hnsw (efd   vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_texture_cosine
    ON leaf_collection USING hnsw (texture vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_color_cosine
    ON leaf_collection USING hnsw (color vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_vein_cosine
    ON leaf_collection USING hnsw (vein  vector_cosine_ops);

-- Bảng meta: lưu tham số extract để truy vết
CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR(50) PRIMARY KEY,
    value TEXT
);

INSERT INTO meta (key, value) VALUES
    ('harmonics',   '20'),
    ('n_resample',  '600'),
    ('glcm_levels', '32'),
    ('dim_efd',     '76'),
    ('dim_texture',    '46'),
    ('dim_color',   '9'),
    ('dim_vein',    '9'),
    ('background',  'white'),
    ('created_by',  'leaf_extract.py')
ON CONFLICT (key) DO NOTHING;