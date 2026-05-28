# run_debug.py
import os
import sys
import argparse
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

print(f"[DEBUG] BASE_DIR = {BASE_DIR}")

# Import an toàn hơn
try:
    from src.debug.preprocess import debug_preprocess
    from src.debug.shape import debug_shape          # ← lỗi ở đây
    from src.debug.texture import debug_texture
    from src.debug.color import debug_color_moments
    from src.debug.vein import debug_vein_features

    from src.core.preprocess import preprocess_leaf

except ImportError as e:
    print(f"[IMPORT ERROR] {e}")
    print("Danh sách file trong src/debug:")
    try:
        print(os.listdir(BASE_DIR / "src" / "debug"))
    except:
        pass
    sys.exit(1)
def main():
    ap = argparse.ArgumentParser(description="Công cụ Debug Trích xuất Đặc trưng")
    ap.add_argument("image", help="Ảnh lá cần debug")
    ap.add_argument("--out", default="debug_out", help="Thư mục lưu ảnh debug")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[LỖI] Không tìm thấy ảnh: {args.image}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  BẮT ĐẦU DEBUG: {os.path.basename(args.image)}")
    print(f"  Output folder: {args.out}")
    print(f"{'═'*70}\n")

    try:
        # Gọi hàm preprocess
        img, contour, gray, mask, leaf_area = preprocess_leaf(args.image)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

        # Gọi các hàm debug
        debug_preprocess(args.image, img, gray, mask, contour, leaf_area, args.out)
        debug_shape(contour, args.out)
        debug_texture(gray_masked, mask, args.out)
        debug_color_moments(img, mask, args.out)
        debug_vein_features(img, mask, leaf_area, args.out)

        print(f"\n{'═'*70}")
        print(f"  HOÀN TẤT DEBUG — Kiểm tra thư mục: {args.out}")
        print(f"{'═'*70}\n")

    except Exception as e:
        print(f"[LỖI] Trong quá trình debug: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()