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


def _save_hist(title: str, data_list: list, labels: list, colors: list,
               path: str, xlabel: str = 'Giá trị pixel') -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for data, label, color in zip(data_list, labels, colors):
        ax.hist(data, bins=50, alpha=0.65, color=color, label=label,
                edgecolor='white')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Số pixel')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → {os.path.basename(path)}")


def debug_vein_features(img: np.ndarray, mask: np.ndarray,
                        leaf_area: int, out_dir: str) -> None:
    from src.features.vein import extract_vein_features  # import cục bộ, tránh circular

    print("\n[BƯỚC 2D] Gân lá — Mật độ & hướng gân")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save("00 — Ảnh gốc", img,  _p(out_dir, "vein_00_original.png"))
    _save("01 — Ảnh xám", gray, _p(out_dir, "vein_01_gray.png"))

    # CLAHE
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _save("02 — CLAHE (tăng tương phản cục bộ)", enhanced,
          _p(out_dir, "vein_02_clahe.png"))

    # Histogram CLAHE
    _save_hist(
        "03 — Histogram xám: trước / sau CLAHE",
        [gray.flatten().astype(float), enhanced.flatten().astype(float)],
        ['Gốc', 'Sau CLAHE'],
        ['steelblue', 'coral'],
        _p(out_dir, "vein_03_hist_clahe.png"),
    )

    # Bilateral filter
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _save("04 — Bilateral filter (giữ cạnh, khử nhiễu)", smooth,
          _p(out_dir, "vein_04_bilateral.png"))

    # Diff
    diff = cv2.absdiff(enhanced, smooth)
    _save("05 — CLAHE − Bilateral = nhiễu bị khử", diff,
          _p(out_dir, "vein_05_diff_noise.png"))

    # Canny
    canny = cv2.Canny(smooth, 20, 80)
    _save("06 — Cạnh Canny (threshold 20 / 80)", canny,
          _p(out_dir, "vein_06_canny.png"))

    # Sobel magnitude
    sx   = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=5)
    sy   = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=5)
    smag = cv2.normalize(np.hypot(sx, sy), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    _save("07 — Độ lớn Sobel (ksize=5)", smag,
          _p(out_dir, "vein_07_sobel_mag.png"))

    # Kết hợp Sobel + Canny
    edges = cv2.addWeighted(smag, 0.6, canny, 0.4, 0)
    _save("08 — Kết hợp Sobel×0.6 + Canny×0.4", edges,
          _p(out_dir, "vein_08_combined.png"))

    # Cắt vùng lá
    vein_raw = cv2.bitwise_and(edges, edges, mask=mask)
    _save("09 — Cắt vùng lá", vein_raw, _p(out_dir, "vein_09_vein_raw.png"))

    # Morphology
    k3   = np.ones((3, 3), np.uint8)
    vein = cv2.morphologyEx(vein_raw, cv2.MORPH_OPEN, k3, iterations=1)
    vein = cv2.dilate(vein, k3, iterations=1)
    density = float((vein > 0).sum() / leaf_area) if leaf_area > 0 else 0.0
    _save(f"10 — Sau morphology  density={density:.4f}", vein,
          _p(out_dir, "vein_10_vein_clean.png"))

    # Angle overlay + histogram
    angle_map = np.arctan2(sy, sx) * 180.0 / np.pi % 180.0
    angle_vis = (angle_map / 180.0 * 255).astype(np.uint8)
    angle_clr = cv2.applyColorMap(angle_vis, cv2.COLORMAP_HSV)
    vein3     = cv2.cvtColor((vein > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    overlay   = cv2.bitwise_and(angle_clr, vein3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Màu = hướng gân (HSV color map)", fontweight='bold')
    axes[0].axis('off')

    if (vein > 0).any():
        angles        = angle_map[vein > 0]
        hist, edges2  = np.histogram(angles, bins=8, range=(0, 180), density=True)
        bin_ctr       = (edges2[:-1] + edges2[1:]) / 2
        axes[1].bar(bin_ctr, hist, width=20, color='steelblue',
                    edgecolor='white', alpha=0.85)
        peak = bin_ctr[np.argmax(hist)]
        axes[1].axvline(peak, color='red', ls='--', lw=2,
                        label=f'Hướng chủ đạo: {peak:.0f}°')
        axes[1].legend()

    axes[1].set_xlabel('Góc (°)')
    axes[1].set_ylabel('Mật độ')
    axes[1].set_title(f"Histogram góc gân (8 bin)\ndensity={density:.4f}",
                      fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle("11 — Phân bố hướng gân lá", fontweight='bold')
    plt.tight_layout()
    plt.savefig(_p(out_dir, "vein_11_angle_hist.png"), dpi=110, bbox_inches='tight')
    plt.close()
    print("    → vein_11_angle_hist.png")

    vec = extract_vein_features(img, mask, leaf_area)
    print(f"  Vector gân ({len(vec)} chiều): {vec.round(4)}")
    print(f"  → Tổng: 12 ảnh (vein_00 ÷ vein_11)")