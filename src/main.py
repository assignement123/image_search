import argparse
import numpy as np
import os
import csv
from src.pipeline import process_single_image

def save_to_csv(feats, image_path, output_path):
    """Lưu toàn bộ vector vào file CSV"""
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Ghi tiêu đề: Đặc trưng, Số chiều, Giá trị cụ thể
        writer.writerow(['Feature Group', 'Dimension Index', 'Value'])
        
        for group, vector in feats.items():
            if vector is not None:
                for i, val in enumerate(vector):
                    writer.writerow([group.upper(), i, val])
    print(f"--- Đã xuất dữ liệu debug ra: {output_path} ---")

def test_extract_single(image_path, export_csv=False):
    if not os.path.exists(image_path):
        print(f"[LỖI] Không tìm thấy file ảnh tại: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU TRÍCH XUẤT: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    try:
        feats = process_single_image(image_path)
        
        total_dims = 0
        print(f"{'ĐẶC TRƯNG':<15} | {'SỐ CHIỀU':<10} | {'PREVIEW (3 giá trị đầu)'}")
        print("-" * 60)
        
        for key, vector in feats.items():
            if vector is not None:
                dim = len(vector)
                total_dims += dim
                preview = np.round(vector[:3], 4).tolist()
                print(f"{key.upper():<15} | {dim:<10} | {preview} ...")
            else:
                print(f"{key.upper():<15} | {'LỖI':<10} | Không thể trích xuất")
        
        print("-" * 60)
        print(f"{'TỔNG CỘNG':<15} | {total_dims:<10} |")
        print(f"{'='*60}\n")

        # Nếu có yêu cầu export CSV
        if export_csv:
            out_name = os.path.splitext(os.path.basename(image_path))[0] + "_debug.csv"
            save_to_csv(feats, image_path, out_name)
        
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Quá trình xử lý thất bại: {e}\n")

def main():
    ap = argparse.ArgumentParser(description="Công cụ Test Trích xuất Đặc trưng Lá cây")
    ap.add_argument("--test", type=str, help="Đường dẫn đến 1 file ảnh lá cần test")
    ap.add_argument("--export", action="store_true", help="Xuất kết quả ra file CSV để debug")
    
    args = ap.parse_args()

    if args.test:
        test_extract_single(args.test, export_csv=args.export)
    else:
        print("\n[INFO] Vui lòng sử dụng cờ --test kèm đường dẫn ảnh.")
        print("Ví dụ: python src/main.py --test leaves_data/1_Phyllostachys_edulis/1001.jpg --export\n")

if __name__ == "__main__":
    main()