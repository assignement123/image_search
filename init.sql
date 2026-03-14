CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE leaf_collection (
    id SERIAL PRIMARY KEY,
    species_name VARCHAR(100),       
    image_path TEXT,                 
    embedding vector(2048),          
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
