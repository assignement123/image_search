import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from src.config import LBP_P, LBP_R, GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS
from src.debug.utils import save_img, save_hist


def _p(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def _save_bar_vector(title: str, values: np.ndarray, labels: list[str], path: str,
                     colors: list[str], ylabel: str = "Giá trị chuẩn hóa") -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='#ffffff22', alpha=0.92, zorder=3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, zorder=2)

    for bar, val in zip(bars, values):
        offset = abs(bar.get_height()) * 0.03 + (0.02 if abs(val) < 1 else 0.01)
        ypos = bar.get_height() + offset
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.2f}", ha='center', va='bottom', fontsize=8.2, color='white', fontweight='bold', rotation=90)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color='#aab4c8', rotation=45, ha='right')
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)
    ax.set_ylabel(ylabel, color='#aab4c8')
    ax.tick_params(axis='y', colors='#aab4c8')
    ax.grid(axis='y', alpha=0.2, color='white', zorder=1)
    ax.spines[:].set_color('#333355')

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"    → {os.path.basename(path)}")


def debug_lbp(gray: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    """Debug riêng cho LBP: ảnh đầu vào, bản đồ LBP, histogram vector."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n[BƯỚC 2A] Texture — LBP")

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    save_img("00 — Gray masked", gray_masked, _p(out_dir, "lbp_00_gray_masked.png"))

    lbp_map = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    lbp_vis = cv2.normalize(lbp_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(lbp_vis, cmap="magma")
    plt.title(f"LBP Map (P={LBP_P}, R={LBP_R})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(_p(out_dir, "lbp_01_map.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("    → lbp_01_map.png")

    lbp_masked = lbp_map[mask > 0]
    hist, _ = np.histogram(lbp_masked, bins=np.arange(0, LBP_P + 3), range=(0, LBP_P + 2))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    labels = [str(i) for i in range(len(hist))]
    colors = ['#4c9be8' if i < len(hist) - 1 else '#f59e0b' for i in range(len(hist))]
    _save_bar_vector(
        f"LBP vector ({len(hist)}D) — histogram uniform patterns",
        hist,
        labels,
        _p(out_dir, "lbp_02_vector.png"),
        colors,
        ylabel="Tần suất chuẩn hóa",
    )

    save_hist(
        "LBP distribution — masked pixel codes",
        [lbp_masked],
        ["LBP codes"],
        ["#7cbbff"],
        _p(out_dir, "lbp_03_hist.png"),
        xlabel="LBP code",
    )

    print(f"  → Tổng: 4 ảnh (lbp_00 ÷ lbp_03)")


def _glcm_labels() -> list[str]:
    props = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
    labels = []
    for prop in props:
        for dist in GLCM_DIST:
            labels.append(f"{prop[:4]}_d{dist}")
    return labels


def debug_glcm(gray_masked: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    """Debug riêng cho GLCM: CLAHE, ảnh lượng tử hóa, ma trận GLCM, vector 20D."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n[BƯỚC 2B] Texture — GLCM")

    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        print("  ⚠ Mask rỗng, tạo placeholder cho GLCM")
        blank = np.zeros_like(gray_masked)
        save_img("00 — Gray masked", blank, _p(out_dir, "glcm_00_gray_masked.png"))
        save_img("01 — CLAHE", blank, _p(out_dir, "glcm_01_clahe.png"))
        save_img("02 — Quantized", blank, _p(out_dir, "glcm_02_quantized.png"))
        return

    cropped_gray = gray_masked[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    save_img("00 — Gray masked", gray_masked, _p(out_dir, "glcm_00_gray_masked.png"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(cropped_gray)
    eq = cv2.bitwise_and(eq, eq, mask=cropped_mask)
    save_img("01 — CLAHE", eq, _p(out_dir, "glcm_01_clahe.png"))

    q = np.clip((eq // (256 // GLCM_LEVELS)).astype(np.uint8), 0, GLCM_LEVELS - 1)
    q_vis = cv2.normalize(q, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_img(f"02 — Quantized ({GLCM_LEVELS} levels)", q_vis, _p(out_dir, "glcm_02_quantized.png"))

    glcm = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    glcm[0, 0, :, :] = 0

    mean_glcm = glcm.mean(axis=(2, 3))
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(mean_glcm, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Mean GLCM matrix")
    plt.xlabel("Gray level")
    plt.ylabel("Gray level")
    plt.tight_layout()
    plt.savefig(_p(out_dir, "glcm_03_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("    → glcm_03_matrix.png")

    props = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    raw_features = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props]).astype(np.float32)
    norm = np.linalg.norm(raw_features)
    normalized_features = raw_features / norm if norm > 1e-7 else raw_features

    labels = _glcm_labels()
    group_colors = (
        ['#4c9be8'] * len(GLCM_DIST) +
        ['#e88a4c'] * len(GLCM_DIST) +
        ['#4ce87a'] * len(GLCM_DIST) +
        ['#d36df0'] * len(GLCM_DIST) +
        ['#f59e0b'] * len(GLCM_DIST)
    )

    _save_bar_vector(
        f"GLCM vector ({len(normalized_features)}D) — {len(GLCM_DIST)} distances × {len(GLCM_ANGLES)} angles × 5 properties",
        normalized_features,
        labels,
        _p(out_dir, "glcm_04_vector.png"),
        group_colors,
        ylabel="Giá trị normalized",
    )

    save_hist(
        "GLCM properties — raw feature distribution",
        [raw_features],
        ["GLCM raw vector"],
        ["#7cbbff"],
        _p(out_dir, "glcm_05_hist.png"),
        xlabel="Feature value",
    )

    print(f"  → Tổng: 6 ảnh (glcm_00 ÷ glcm_05)")
