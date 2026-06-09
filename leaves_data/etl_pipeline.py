import os
import glob
from tqdm import tqdm
from dotenv import load_dotenv

from src.pipeline import process_single_image
from src.db.postgres_repo import connect_db, insert_pg, check_connection

load_dotenv()

def run_etl():
    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH ETL (Extract, Transform, Load)")
    print("="*50)

    if not check_connection():
        print("Dừng tiến trình do lỗi Database.")
        return

    data_dir = os.getenv("DATA_DIR", "../leaves_data")
    if not os.path.exists(data_dir):
        print(f"❌ [LỖI] Không tìm thấy thư mục ảnh: {data_dir}")
        return

    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

    total_images = len(image_paths)
    print(f"\n📁 Tìm thấy {total_images} ảnh trong thư mục.")
    if total_images == 0:
        return

    conn = connect_db()
    success_count = 0
    error_count = 0

    for img_path in tqdm(image_paths, desc="Đang trích xuất & lưu DB", unit="ảnh"):
        filename = os.path.basename(img_path)
        
        species = os.path.basename(os.path.dirname(img_path))

        try:
            feats = process_single_image(img_path)
            
            if any(v is None for v in feats.values()):
                raise ValueError("Một hoặc nhiều đặc trưng trả về None.")

            insert_pg(conn, filename, img_path, feats, species)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"⚠️ Bỏ qua {filename} - Lỗi: {str(e)}")

    conn.close()

    print("\n" + "="*50)
    print(f"✅ QUÁ TRÌNH HOÀN TẤT!")
    print(f"   - Thành công: {success_count}/{total_images} ảnh")
    print(f"   - Thất bại:   {error_count} ảnh")
    print("="*50)

if __name__ == "__main__":
    run_etl()