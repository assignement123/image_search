import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyefd import elliptic_fourier_descriptors
from src.features.contour import _resample_contour
from src.config import HARMONICS, N_RESAMPLE

def get_path(out_dir, filename):
    return os.path.join(out_dir, filename)

def debug_shape(contour: np.ndarray, out_dir: str) -> None:
    print("\n[DEBUG SHAPE] EFD & Hình thái học")
    contour = np.asarray(contour, np.float64).reshape(-1, 2)

    rs = _resample_contour(contour, N_RESAMPLE)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rs[:, 0], rs[:, 1], 'g-', lw=1.5)
    ax.plot(rs[0, 0], rs[0, 1], 'ko', ms=10, label='Start Point')
    ax.set_title(f"01 — Resample ({N_RESAMPLE} pts)", fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig(get_path(out_dir, "shape_01_resampled.png"), dpi=110)
    plt.close()
    print("    → shape_01_resampled.png")

    coeffs_norm = elliptic_fourier_descriptors(rs, order=HARMONICS + 1, normalize=True)
    efd_vec = coeffs_norm[2:HARMONICS + 1, :].flatten().astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(efd_vec)), efd_vec, color='coral', edgecolor='white', alpha=0.85)
    ax.set_title(f"Vector EFD ({len(efd_vec)} chiều) lưu vào DB", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(get_path(out_dir, "shape_02_vector.png"), dpi=110)
    plt.close()
    print("    → shape_02_vector.png")