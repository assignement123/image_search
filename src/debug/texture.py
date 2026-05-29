import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from src.config import LBP_P, LBP_R, GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS
from src.features.texture import extract_lbp
from src.debug.utils import save_img


def _save_lbp_histogram(hist: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(len(hist))
    ax.bar(bins, hist, color="#2d8fdd", edgecolor="white", linewidth=0.6)
    ax.set_title("LBP Histogram (uniform)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("LBP bin")
    ax.set_ylabel("Normalized frequency")
    ax.set_xticks(np.arange(0, len(hist), 2))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(out_path)}")


def _save_glcm_property_chart(glcm: np.ndarray, out_path: str) -> None:
    props = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]
    dist_labels = [f"d={d}" for d in GLCM_DIST]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i, prop in enumerate(props):
        values = graycoprops(glcm, prop).mean(axis=1)
        axes[i].bar(dist_labels, values, color="#1f6f54", edgecolor="white", linewidth=0.6)
        axes[i].set_title(prop, fontsize=10)
        axes[i].grid(axis="y", alpha=0.25)
        axes[i].tick_params(axis='x', labelrotation=20)
    axes[-1].axis("off")
    fig.suptitle("GLCM Properties by Distance (angle-averaged)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(out_path)}")


def _save_glcm_matrix(glcm: np.ndarray, out_path: str) -> None:
    matrix = glcm[:, :, 0, 0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_title("GLCM Matrix (distance=1, angle=0)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Gray level j")
    ax.set_ylabel("Gray level i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"    → {os.path.basename(out_path)}")


def debug_texture(gray_masked: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Reuse the production feature extractor to guarantee consistent histogram values.
    lbp_hist = extract_lbp(gray_masked, mask)
    lbp = local_binary_pattern(gray_masked, LBP_P, LBP_R, method="uniform")
    lbp_vis = np.zeros_like(gray_masked, dtype=np.float32)
    lbp_vis[mask > 0] = lbp[mask > 0]
    lbp_vis = cv2.normalize(lbp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    glcm_input = cv2.equalizeHist(gray_masked)
    glcm_resized = cv2.resize(glcm_input, (256, 256))
    quantized = np.clip((glcm_resized // (256 // GLCM_LEVELS)).astype(np.uint8), 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(
        quantized,
        distances=GLCM_DIST,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    save_img(
        "LBP Pattern Map (masked)",
        lbp_vis,
        os.path.join(out_dir, "step2b_01_lbp_map.png"),
        cmap="turbo",
    )
    _save_lbp_histogram(lbp_hist, os.path.join(out_dir, "step2b_02_lbp_hist.png"))
    _save_glcm_property_chart(glcm, os.path.join(out_dir, "step2b_03_glcm_props.png"))
    _save_glcm_matrix(glcm, os.path.join(out_dir, "step2b_04_glcm_matrix.png"))
