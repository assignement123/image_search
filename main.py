import os
import glob
import uuid
import json
import argparse

# Cấu hình Database PostgreSQL (dựa theo config bạn đang dùng)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "leaf_db"
DB_USER = "admin"
DB_PASS = "admin"


def _load_leave_data(data_path: str):
    import pandas as pd

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {data_path}")

    _, ext = os.path.splitext(data_path.lower())
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(data_path)
    return pd.read_csv(data_path)


def analyze_leave_data(data_path: str = "leave_data.csv"):
    try:
        df = _load_leave_data(data_path)
    except Exception as err:
        print(f"❌ Không thể đọc dataset '{data_path}': {err}")
        return {"is_ready": False, "errors": [str(err)]}

    total_rows, total_cols = df.shape
    missing_ratio = (df.isna().mean() * 100).round(2)
    duplicated_rows = int(df.duplicated().sum())
    duplicate_ratio = round((duplicated_rows / total_rows) * 100, 2) if total_rows else 0.0

    all_missing_cols = [col for col, pct in missing_ratio.items() if pct >= 100]
    high_missing_cols = [col for col, pct in missing_ratio.items() if 30 <= pct < 100]
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    usable_cols = [col for col in df.columns if col not in all_missing_cols and col not in constant_cols]

    critical_issues = []
    warnings = []

    if total_rows < 20:
        critical_issues.append(f"Số dòng quá ít ({total_rows}) để phân tích thuộc tính ổn định.")
    if total_cols < 2:
        critical_issues.append(f"Số cột quá ít ({total_cols}) để phân tích đa thuộc tính.")
    if len(usable_cols) < 2:
        critical_issues.append("Số cột có dữ liệu hữu ích < 2.")
    if all_missing_cols:
        critical_issues.append(f"Cột trống hoàn toàn: {all_missing_cols}")
    if not numeric_cols:
        warnings.append("Không có cột số; phân tích thống kê định lượng sẽ bị hạn chế.")
    if high_missing_cols:
        warnings.append(f"Cột thiếu dữ liệu cao (>=30%): {high_missing_cols}")
    if constant_cols:
        warnings.append(f"Cột không biến thiên (<=1 giá trị duy nhất): {constant_cols}")
    if duplicate_ratio > 20:
        warnings.append(f"Tỷ lệ dòng trùng lặp cao: {duplicate_ratio}%")

    is_ready = len(critical_issues) == 0

    result = {
        "is_ready": is_ready,
        "rows": total_rows,
        "columns": total_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "duplicate_rows": duplicated_rows,
        "duplicate_ratio_percent": duplicate_ratio,
        "missing_ratio_percent": missing_ratio.to_dict(),
        "critical_issues": critical_issues,
        "warnings": warnings,
    }

    print("\n📊 ĐÁNH GIÁ DATASET LEAVE_DATA")
    print("-" * 50)
    print(f"Số dòng: {total_rows}")
    print(f"Số cột: {total_cols}")
    print(f"Cột số: {len(numeric_cols)} | Cột phân loại/khác: {len(categorical_cols)}")
    print(f"Dòng trùng lặp: {duplicated_rows} ({duplicate_ratio}%)")

    if critical_issues:
        print("\n❌ Dataset CHƯA ổn cho phân tích thuộc tính:")
        for issue in critical_issues:
            print(f" - {issue}")
    else:
        print("\n✅ Dataset đạt điều kiện tối thiểu để phân tích thuộc tính.")

    if warnings:
        print("\n⚠️ Khuyến nghị làm sạch thêm:")
        for warning in warnings:
            print(f" - {warning}")

    print("\n📌 Tỷ lệ thiếu dữ liệu theo cột:")
    for col, pct in result["missing_ratio_percent"].items():
        print(f" - {col}: {pct}%")

    return result


def connect_db():
    import psycopg2

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def run_upload():
    from tqdm import tqdm
    from app.core.features.preprocess import preprocess_leaf_image
    from app.core.features.texture import extract_texture_features

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
    from app.core.features.preprocess import preprocess_leaf_image
    from app.core.features.texture import extract_texture_features

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["upload", "analyze"],
        default="upload",
        help="upload: trích xuất + upload ảnh, analyze: đánh giá dataset leave_data",
    )
    parser.add_argument(
        "--data-path",
        default="leave_data.csv",
        help="Đường dẫn dataset cho mode analyze (.csv/.xlsx/.xls)",
    )
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_leave_data(args.data_path)
    else:
        run_upload()
