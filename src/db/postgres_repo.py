import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()


def connect_db():
    """Tạo kết nối đến PostgreSQL và đăng ký kiểu vector."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )
    register_vector(conn)   # đăng ký 1 lần/connection, không gọi lại trong insert_pg
    return conn


def check_connection():
    """Kiểm tra kết nối DB và pgvector extension."""
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
        print(f"❌ [LỖI] Không thể kết nối Database: {e}")
        return False


def insert_pg(conn, filename, path, feats, species=None):
    """
    Insert hoặc update một ảnh lá vào leaf_collection.
    - conn: connection đã có register_vector (từ connect_db())
    - feats: dict từ process_single_image(), giá trị là numpy array
    - efd_coeffs_flipped không cần lưu: reflection invariance xử lý ở query time
    """
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO leaf_collection
                (filename, image_path, species,
                 efd_coeffs, morphology_stats, lbp_hist,
                 glcm_stats, color_moments, vein_features)
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
                filename,
                path,
                species,
                feats["efd_coeffs"],        # numpy array — pgvector xử lý trực tiếp
                feats["morphology_stats"],
                feats["lbp_hist"],
                feats["glcm_stats"],
                feats["color_moments"],
                feats["vein_features"],
            )
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
