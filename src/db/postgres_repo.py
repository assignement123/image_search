import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

def connect_db():
    """Tạo kết nối đến PostgreSQL"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "leaf_db"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASS", "admin")
    )

def check_connection():
    """Hàm kiểm tra DB đã kết nối thành công chưa"""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"✅ [THÀNH CÔNG] Đã kết nối đến Database!")
        print(f"📦 Version: {db_version[0]}")
        
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("🚀 [OK] Extension 'pgvector' đã được cài đặt.")
        else:
            print("⚠️ [CẢNH BÁO] Chưa cài đặt extension 'pgvector'!")
            
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ [LỖI] Không thể kết nối Database. Chi tiết lỗi:")
        print(e)
        return False

def insert_pg(conn, filename, path, feats, species=None):
    """Hàm insert dữ liệu vào DB (giữ nguyên logic cũ của bạn)"""
    register_vector(conn)
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