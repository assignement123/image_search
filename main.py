import os
import cv2
import glob
import uuid
import json
import psycopg2
import numpy as np
from tqdm import tqdm
from app.core.features.preprocess import preprocess_leaf_image
from app.core.features.texture import extract_texture_features

# Cấu hình Database PostgreSQL (dựa theo config bạn đang dùng)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "leaf_db"
DB_USER = "admin"
DB_PASS = "admin"

def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def run_upload():
    DATA_DIR = "data/Leaves/"
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục {DATA_DIR}")
        return

    # Lấy danh sách toàn bộ ảnh .jpg
    image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not image_paths:
        print(f"❌ Không tìm thấy ảnh nào trong {DATA_DIR}")
        return

    print(f"🚀 Bắt đầu trích xuất và upload {len(image_paths)} ảnh vào Database...")
    print("-" * 50)

    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        success_count = 0
        error_count = 0

        # Dùng tqdm để hiển thị thanh tiến trình
        for img_path in tqdm(image_paths, desc="Processing Leaves", unit="img"):
            try:
                # 1. Tiền xử lý ảnh (Cắt form, xoay chuẩn)
                processed_img = preprocess_leaf_image(img_path)
                if processed_img is None:
                    error_count += 1
                    continue
                
                # 2. Trích xuất đặc trưng LBP + GLCM => vector 29 chiều
                vector = extract_texture_features(processed_img)
                if vector is None:
                    error_count += 1
                    continue

                # 3. Ép kiểu về List Float chuẩn để nạp vào pgvector
                vector_list = vector.tolist()
                
                # 4. Định dạng dữ liệu meta
                item_id = str(uuid.uuid4())
                file_name = os.path.basename(img_path)
                
                # Vì tập data đang không có nhãn loài, tạm để Unknown
                features_meta = json.dumps({"source": "extract_texture_features", "dimension": len(vector_list)})

                # 5. Lưu vào Table leaf_metadata_test (dựa trên init.sql)
                cursor.execute("""
                    INSERT INTO leaf_metadata_test 
                    (id, species_name, scientific_name, image_path, detailed_features, fused_vector)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                """, (
                    item_id,
                    "Unknown",          # species_name
                    file_name,          # Lưu tạm tên file vào scientific_name
                    img_path,           # image_path
                    features_meta,      # detailed_features JSON
                    vector_list         # fused_vector
                ))
                
                success_count += 1
                
                # Checkpoint commit mỗi 50 ảnh
                if success_count % 50 == 0:
                    conn.commit()

            except Exception as e:
                error_count += 1
                tqdm.write(f"💥 Lỗi tại ảnh {img_path}: {e}")
                continue

        # Commit những dữ liệu cuối cùng chưa được đẩy
        conn.commit()

        print("-" * 50)
        print(f"✅ Hoàn tất quá trình Upload!")
        print(f"   - Thành công: {success_count} ảnh")
        print(f"   - Bỏ qua/Lỗi: {error_count} ảnh")

    except Exception as db_err:
        print(f"💥 Lỗi kết nối Database: {db_err}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def find_similar_leaves(img_path: str, limit: int = 10):
    # 1. Tiền xử lý ảnh query
    processed_img = preprocess_leaf_image(img_path)
    if processed_img is None:
        print(f"❌ Không thể xử lý ảnh: {img_path}")
        return []
    
    # 2. Trích xuất đặc trưng
    vector = extract_texture_features(processed_img)
    if vector is None:
        print(f"❌ Không thể trích xuất đặc trưng ảnh: {img_path}")
        return []
        
    vector_list = vector.tolist()

    try:
        conn = connect_db()
        cursor = conn.cursor()
        
        # 3. Truy vấn tìm kiếm Vector (So sánh L2 Distance / Cosine)
        # <=> là toán tử Cosine distance trong pgvector
        query = """
            SELECT id, species_name, scientific_name, image_path, 
                   1 - (fused_vector <=> %s::vector) AS similarity 
            FROM leaf_metadata_test 
            ORDER BY fused_vector <=> %s::vector 
            LIMIT %s
        """
        cursor.execute(query, (vector_list, vector_list, limit))
        
        similar_leaves = cursor.fetchall()
        
        results = []
        for row in similar_leaves:
            results.append({
                "id": str(row[0]),
                "species_name": row[1],
                "scientific_name": row[2],
                "image_path": row[3],
                "similarity": float(row[4])
            })
            
        return results
        
    except Exception as e:
        print(f"💥 Loi truy van CSDL: {e}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_upload()
