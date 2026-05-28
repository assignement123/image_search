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
    from src.features.vein import extract_vein_features

    print("\n[BƯỚC 2D] Gân lá — Mật độ & hướng gân")

    # ── Ảnh gốc & xám ───────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save("00 — Ảnh gốc", img,  _p(out_dir, "vein_00_original.png"))
    _save("01 — Ảnh xám", gray, _p(out_dir, "vein_01_gray.png"))

    # ── Erode mask ───────────────────────────────────────────────────
    k_erode    = np.ones((15, 15), np.uint8)
    mask_inner = cv2.erode(mask, k_erode, iterations=1)
    _save("02 — Mask sau erode (loại viền lá)", mask_inner,
          _p(out_dir, "vein_02_mask_inner.png"))

    # ── CLAHE ────────────────────────────────────────────────────────
    clahe    = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)
    _save("03 — CLAHE", enhanced, _p(out_dir, "vein_03_clahe.png"))

    _save_hist(
        "04 — Histogram: trước / sau CLAHE",
        [gray.flatten().astype(float), enhanced.flatten().astype(float)],
        ['Gốc', 'Sau CLAHE'], ['steelblue', 'coral'],
        _p(out_dir, "vein_04_hist_clahe.png"),
    )

    # ── Gaussian + mask_inner ────────────────────────────────────────
    smooth_full = cv2.GaussianBlur(enhanced, (5, 5), 0)
    smooth      = cv2.bitwise_and(smooth_full, smooth_full, mask=mask_inner)
    _save("05 — Gaussian blur (trong mask_inner)", smooth,
          _p(out_dir, "vein_05_smooth.png"))

    # ── Otsu chỉ trên pixels trong mask ─────────────────────────────
    pixels_in_mask = smooth[mask_inner > 0]
    otsu_thresh, _ = cv2.threshold(pixels_in_mask, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = float(otsu_thresh) * 0.3
    high = float(otsu_thresh) * 0.7
    print(f"  otsu_thresh={otsu_thresh:.1f}  low={low:.1f}  high={high:.1f}")

    # ── Canny ────────────────────────────────────────────────────────
    canny = cv2.Canny(smooth, low, high)
    _save(f"06 — Canny (low={low:.0f} / high={high:.0f})", canny,
          _p(out_dir, "vein_06_canny.png"))

    # ── Apply mask_inner → vein ──────────────────────────────────────
    vein_raw = cv2.bitwise_and(canny, canny, mask=mask_inner)
    _save("07 — Canny cắt mask_inner", vein_raw,
          _p(out_dir, "vein_07_vein_raw.png"))

    k2   = np.ones((2, 2), np.uint8)
    vein = cv2.morphologyEx(vein_raw, cv2.MORPH_CLOSE, k2, iterations=1)
    inner_area = float((mask_inner > 0).sum())
    density    = float((vein > 0).sum() / inner_area) if inner_area > 0 else 0.0
    _save(f"08 — Sau OPEN  density={density:.4f}", vein,
          _p(out_dir, "vein_08_vein_clean.png"))

    # ── Sanity check ─────────────────────────────────────────────────
    if density > 0.5:
        print(f"  ⚠ density={density:.4f} > 0.5 — quá nhiều nhiễu.")
    elif density < 0.02:
        print(f"  ⚠ density={density:.4f} < 0.02 — bỏ sót gân, hạ threshold.")

    # ── Sobel chỉ để tính angle histogram ───────────────────────────
    sx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

    mask_f    = (mask_inner > 0).astype(np.float64)
    angle_map = np.arctan2(sy * mask_f, sx * mask_f) * 180.0 / np.pi % 180.0

    # ── Angle overlay + histogram ────────────────────────────────────
    angle_vis = (angle_map / 180.0 * 255).astype(np.uint8)
    angle_clr = cv2.applyColorMap(angle_vis, cv2.COLORMAP_HSV)
    vein3     = cv2.cvtColor((vein > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    overlay   = cv2.bitwise_and(angle_clr, vein3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Màu = hướng gân (HSV)", fontweight='bold')
    axes[0].axis('off')

    if (vein > 0).any():
        angles       = angle_map[vein > 0]
        hist, edges2 = np.histogram(angles, bins=8, range=(0, 180), density=True)
        bin_ctr      = (edges2[:-1] + edges2[1:]) / 2
        axes[1].bar(bin_ctr, hist, width=20, color='steelblue',
                    edgecolor='white', alpha=0.85)
        peak = bin_ctr[np.argmax(hist)]
        axes[1].axvline(peak, color='red', ls='--', lw=2,
                        label=f'Hướng chủ đạo: {peak:.0f}°')
        axes[1].legend()

    axes[1].set_xlabel('Góc (°)')
    axes[1].set_ylabel('Mật độ')
    axes[1].set_title(f"Histogram góc gân\ndensity={density:.4f}", fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle("09 — Phân bố hướng gân lá", fontweight='bold')
    plt.tight_layout()
    plt.savefig(_p(out_dir, "vein_09_angle_hist.png"), dpi=110, bbox_inches='tight')
    plt.close()
    print("    → vein_09_angle_hist.png")

    # ── Vector từ extract_vein_features (code thật) ──────────────────
    vec = extract_vein_features(img, mask, leaf_area)
    print(f"  Vector gân ({len(vec)} chiều): {vec.round(4)}")

    # ── Diagnostic ───────────────────────────────────────────────────
    print("=== DIAGNOSTIC ===")
    print(f"mask_inner: pixels > 0 = {(mask_inner > 0).sum():,}  "
          f"(mask gốc = {(mask > 0).sum():,})")
    print(f"canny:      pixels > 0 (trong mask_inner) = "
          f"{((canny > 0) & (mask_inner > 0)).sum():,}")
    print(f"vein_raw:   pixels > 0 = {(vein_raw > 0).sum():,}  "
          f"ratio = {(vein_raw > 0).sum() / inner_area:.4f}")
    print(f"vein:       pixels > 0 = {(vein > 0).sum():,}  "
          f"ratio = {(vein > 0).sum() / inner_area:.4f}")
    print(f"density (debug) = {density:.4f}  |  density (vec[0]) = {vec[0]:.4f}")
    print(f"  → Tổng: 10 ảnh (vein_00 ÷ vein_09)")