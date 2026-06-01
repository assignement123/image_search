# preprocess.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
FIXED_THRESH = 210

def _p(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def _save(title: str, img: np.ndarray, path: str, cmap: str = 'gray') -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → {os.path.basename(path)}")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH NỘI BỘ
# ════════════════════════════════════════════════════════════════════

def _p(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def _save(title: str, img: np.ndarray, path: str, cmap: str = 'gray') -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → {os.path.basename(path)}")


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Lấp lỗ hổng bên trong mask bằng flood fill an toàn.
    Thêm border đen 1px → seed (0,0) luôn là nền, không bao giờ là lá.
    """
    bordered  = cv2.copyMakeBorder(mask, 1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT, value=0)
    flood     = bordered.copy()
    h, w      = flood.shape
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]   # cắt border, lấy lỗ hổng
    return cv2.bitwise_or(mask, holes)


# ════════════════════════════════════════════════════════════════════
# NGƯỠNG CỐ ĐỊNH — nền trắng chuẩn hóa
# ════════════════════════════════════════════════════════════════════
#
#  Tại sao dùng ngưỡng cố định thay Otsu:
#    - Otsu tìm ngưỡng tách 2 lớp dựa trên histogram.
#    - Với lá sáng (lá non, lá vàng, lá khô): histogram lá và nền
#      gần nhau → Otsu đặt ngưỡng quá cao hoặc quá thấp.
#    - Database đã chuẩn hóa nền TRẮNG (pixel ~230–255) →
#      ngưỡng cố định 210 luôn nằm trong vùng nền, không bao giờ
#      cắt vào lá dù lá sáng đến đâu.
#
#  Điều chỉnh nếu cần:
#    - Tăng lên 220 nếu lá rất sáng vẫn bị lẫn vào nền
#    - Giảm xuống 200 nếu nền không hoàn toàn trắng (hơi vàng/xám)
#
FIXED_THRESH = 210

# ════════════════════════════════════════════════════════════════════
# DEBUG PREPROCESS
# ════════════════════════════════════════════════════════════════════

def debug_preprocess(
    image_path: str,
    img: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
    leaf_area: int,
    out_dir: str,
) -> None:
    """
    Visualize từng bước của preprocess_leaf.
    Nhận dữ liệu đã tính sẵn → không gọi preprocess_leaf() lại lần 2.
    """
    print("\n[BƯỚC 1] Tiền xử lý ảnh (nền trắng)")
    os.makedirs(out_dir, exist_ok=True)

    # 00 — Ảnh gốc
    _save("00 — Ảnh gốc", img, _p(out_dir, "step1_00_original.png"))

    # 01 — Grayscale
    _save("01 — Ảnh xám (Grayscale)", gray, _p(out_dir, "step1_01_gray.png"))

    # 02 — Histogram + ngưỡng cố định
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gray_blur.flatten(), bins=64, range=(0, 255),
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(FIXED_THRESH, color='red', lw=2.5, ls='--',
               label=f'Ngưỡng cố định = {FIXED_THRESH}')
    ymax = ax.get_ylim()[1]
    ax.fill_betweenx([0, ymax], 0, FIXED_THRESH,
                     alpha=0.12, color='orange',
                     label='Vùng lá (< ngưỡng)')
    ax.set_title(f"02 — Histogram & ngưỡng cố định ({FIXED_THRESH})",
                 fontweight='bold')
    ax.set_xlabel('Mức xám'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(_p(out_dir, "step1_02_threshold.png"), dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → step1_02_threshold.png  (ngưỡng cố định={FIXED_THRESH})")

    # 03 — Binary INV
    _, binary = cv2.threshold(gray_blur, FIXED_THRESH, 255,
                              cv2.THRESH_BINARY_INV)
    _save(f"03 — Nhị phân INV ngưỡng {FIXED_THRESH} (lá=trắng, nền=đen)",
          binary, _p(out_dir, "step1_03_binary_inv.png"))

    # 04 — MORPH_OPEN
    kernel_open  = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    _save("04 — MORPH_OPEN 3×3 (xoá nhiễu, giữ cuống mỏng)", opened,
          _p(out_dir, "step1_04_morph_open.png"))

    # 05 — MORPH_CLOSE
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    _save("05 — MORPH_CLOSE 7×7 (lấp lỗ hổng)", closed,
          _p(out_dir, "step1_05_morph_close.png"))

    # 05b — Sau _fill_holes
    closed_filled = _fill_holes(closed)
    _save("05b — Sau fill_holes (lấp lỗ cuống mỏng)",
          closed_filled, _p(out_dir, "step1_05b_fill_holes.png"))

    # 06 — Contour trên ảnh gốc
    contour_vis = cv2.drawContours(
        img.copy(), [contour.astype(np.int32)], -1, (0, 255, 0), 2)
    _save(
        f"06 — Contour lớn nhất ({len(contour)} điểm  |  CCW, điểm đầu=x_min)",
        contour_vis,
        _p(out_dir, "step1_06_contour.png"),
    )

    # 07 — Mask cuối
    _save(f"07 — Mask lá (diện tích = {leaf_area:,} px)",
          mask, _p(out_dir, "step1_07_mask.png"))

    # 08 — Mask & vùng lá sau cắt nền
    leaf_cut = cv2.bitwise_and(img, img, mask=mask)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(f"Mask  ({leaf_area:,} px)", fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(leaf_cut, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Vùng lá sau cắt nền", fontweight='bold')
    axes[1].axis('off')
    plt.suptitle("08 — Mask & vùng lá đã cắt", fontweight='bold')
    plt.tight_layout()
    plt.savefig(_p(out_dir, "step1_08_mask_and_crop.png"), dpi=110, bbox_inches='tight')
    plt.close()
    print("    → step1_08_mask_and_crop.png")

    print(f"  → Tổng: 10 ảnh (step1_00 ÷ step1_08, thêm step1_05b)")