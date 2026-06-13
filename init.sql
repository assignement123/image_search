CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE leaf_metadata_test (
    id UUID PRIMARY KEY,
    species_name TEXT,
    scientific_name TEXT,
    image_path TEXT,    
    detailed_features JSONB, 
    fused_vector vector(29) 
);

CREATE INDEX ON leaf_metadata_test USING hnsw (fused_vector vector_cosine_ops);
