import os
import sys
import argparse
import cv2

from src.debug.preprocess import debug_preprocess
from src.debug.shape import debug_shape
from src.debug.texture import debug_texture
from src.debug.color import debug_color_moments
from src.debug.vein import debug_vein_features

from core.preprocess import preprocess_leaf


def main():
    ap = argparse.ArgumentParser(description="Công cụ Debug Trích xuất Đặc trưng")
    ap.add_argument("image", help="Ảnh lá cần debug")
    ap.add_argument("--out", default="debug_out", help="Thư mục lưu ảnh debug")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[LỖI] Không tìm thấy: {args.image}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  BẮT ĐẦU DEBUG: {os.path.basename(args.image)}")
    print(f"{'═'*60}")

    img, contour, gray, mask, leaf_area = preprocess_leaf(args.image)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    debug_preprocess(args.image, img, gray, mask, contour, args.out)
    debug_shape(contour, args.out)
    debug_texture(gray_masked, mask, args.out)
    debug_color_moments(img, mask, args.out)
    debug_vein_features(img, mask, leaf_area, args.out)

    print(f"\n{'═'*60}")
    print(f"  HOÀN TẤT — Kiểm tra thư mục: {args.out}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()