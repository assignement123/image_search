# preprocess.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    Nhận dữ liệu đã tính sẵn từ main() thay vì đọc lại file,
    tránh gọi preprocess_leaf() hai lần.
    """
    print("\n[BƯỚC 1] Tiền xử lý ảnh (nền trắng)")

    # 00 — Ảnh gốc
    _save("00 — Ảnh gốc", img, _p(out_dir, "step1_00_original.png"))

    # 01 — Grayscale
    _save("01 — Ảnh xám (Grayscale)", gray, _p(out_dir, "step1_01_gray.png"))

    # 02 — Histogram + ngưỡng Otsu
    thresh_val, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gray.flatten(), bins=64, range=(0, 255),
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(thresh_val, color='red', lw=2.5, ls='--',
               label=f'Ngưỡng Otsu = {thresh_val:.0f}')
    # Lấy ylim sau khi đã vẽ histogram
    ymax = ax.get_ylim()[1]
    ax.fill_betweenx([0, ymax], 0, thresh_val,
                     alpha=0.12, color='orange',
                     label='Vùng lá (< ngưỡng)')
    ax.set_title("02 — Histogram & ngưỡng Otsu", fontweight='bold')
    ax.set_xlabel('Mức xám'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(_p(out_dir, "step1_02_otsu_threshold.png"), dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → step1_02_otsu_threshold.png  (ngưỡng={thresh_val:.0f})")

    # 03 — Binary INV
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _save("03 — Nhị phân Otsu INV (lá=trắng, nền=đen)",
          binary, _p(out_dir, "step1_03_binary_inv.png"))

    # 04 — MORPH_OPEN
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _save("04 — MORPH_OPEN (xoá nhiễu nhỏ)", opened,
          _p(out_dir, "step1_04_morph_open.png"))

    # 05 — MORPH_CLOSE
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    _save("05 — MORPH_CLOSE (lấp lỗ hổng)", closed,
          _p(out_dir, "step1_05_morph_close.png"))

    # 06 — Contour trên ảnh gốc
    # Hướng CCW: shoelace area > 0 → giữ nguyên; < 0 → đảo ngược
    # Điểm bắt đầu: x nhỏ nhất (tie-break: y nhỏ nhất)
    contour_vis = cv2.drawContours(
        img.copy(), [contour.astype(np.int32)], -1, (0, 255, 0), 2)
    _save(
        f"06 — Contour lớn nhất ({len(contour)} điểm  |  CCW, điểm đầu=x_min)",
        contour_vis,
        _p(out_dir, "step1_06_contour.png"),
    )

    # 07 — Mask
    _save(f"07 — Mask lá (diện tích = {leaf_area:,} px)",
          mask, _p(out_dir, "step1_07_mask.png"))

    # 08 — Mask + vùng cắt
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

    print(f"  → Tổng: 9 ảnh (step1_00 ÷ step1_08)")