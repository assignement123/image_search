# shape.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pyefd import elliptic_fourier_descriptors
from src.features.contour import _resample_contour, extract_efd
from src.config import HARMONICS, N_RESAMPLE, DIM_EFD


def _p(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def debug_shape(contour: np.ndarray, out_dir: str) -> None:
    print("\n[BƯỚC 2A] EFD — Hình dạng biên lá")

    contour = np.asarray(contour, np.float64).reshape(-1, 2)

    # 00 - Contour gốc
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(contour[:, 0], contour[:, 1], 'b-', lw=1.5)
    ax.plot(contour[0, 0], contour[0, 1], 'ro', ms=8)
    ax.set_title(f"00 — Contour gốc ({len(contour)} điểm)", fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_p(out_dir, "shape_00_contour_raw.png"), dpi=110)
    plt.close()
    print("    → shape_00_contour_raw.png")

    # 01 - Resample contour
    rs = _resample_contour(contour, N_RESAMPLE)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rs[:, 0], rs[:, 1], 'g-', lw=1.5)
    ax.scatter(rs[::30, 0], rs[::30, 1], c='red', s=20)
    ax.set_title(f"01 — Resample ({N_RESAMPLE} điểm)", fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_p(out_dir, "shape_01_resampled.png"), dpi=110)
    plt.close()
    print("    → shape_01_resampled.png")

    # Tính EFD — y nguyên từ leaf_debug.py gốc
    coeffs = elliptic_fourier_descriptors(rs, order=HARMONICS, normalize=False)
    A0 = rs[:, 0].mean()
    C0 = rs[:, 1].mean()
    t  = np.linspace(0, 1, N_RESAMPLE)

    def reconstruct(n_max):
        x = np.full_like(t, A0)
        y = np.full_like(t, C0)
        for n in range(n_max):
            an, bn, cn, dn = coeffs[n]
            k = n + 1
            x += an * np.cos(2 * np.pi * k * t) + bn * np.sin(2 * np.pi * k * t)
            y += cn * np.cos(2 * np.pi * k * t) + dn * np.sin(2 * np.pi * k * t)
        return x, y

    cx, cy = rs[:, 0], rs[:, 1]

    # 02–06 Tái tạo các bậc (tất cả n_h phải ≤ HARMONICS để tránh IndexError)
    for i, n_h in enumerate([1, 3, 5, 10, HARMONICS]):
        x, y = reconstruct(n_h)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(x, y, 'b-', lw=2)
        axes[0].set_title(f'EFD space (n={n_h})', fontweight='bold')
        axes[0].set_aspect('equal'); axes[0].invert_yaxis(); axes[0].grid(alpha=0.3)
        axes[1].plot(cx, cy, 'g--', lw=1.5, label='Contour gốc')
        axes[1].plot(x, y, 'b-', lw=2, label=f'Reconstruct n={n_h}')
        axes[1].set_title('So sánh pixel space', fontweight='bold')
        axes[1].set_aspect('equal'); axes[1].invert_yaxis()
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fname = f"shape_{i+2:02d}_reconstruct_n{n_h}.png"
        plt.suptitle(f"Tái tạo EFD bậc {n_h}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(_p(out_dir, fname), dpi=110)
        plt.close()
        print(f"    → {fname}")

    # 07 - Amplitude plot
    amps = [np.sqrt(sum(c**2 for c in coeffs[n])) for n in range(1, len(coeffs))]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(amps) + 1), amps)
    ax.set_xlabel("Harmonic n"); ax.set_ylabel("Amplitude")
    ax.set_title("07 — Biên độ EFD theo bậc", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(_p(out_dir, "shape_07_amplitudes.png"), dpi=110)
    plt.close()
    print("    → shape_07_amplitudes.png")

    # 08 - So sánh vector DB (normalize=True) vs raw (normalize=False)
    efd_vec     = extract_efd(contour)               # production: normalize=True, coeffs[1:]
    efd_raw_vec = coeffs[1:, :].flatten().astype(np.float32)  # normalize=False, harmonics 2..HARMONICS (56 dims)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].bar(range(len(efd_raw_vec)), efd_raw_vec,
                color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_title("EFD vector — normalize=False (phụ thuộc scale)", fontweight='bold')
    axes[0].set_xlabel("Chiều"); axes[0].set_ylabel("Giá trị")
    axes[0].axhline(0, color='black', lw=0.5); axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(efd_vec)), efd_vec,
                color='coral', edgecolor='white', alpha=0.85)
    axes[1].set_title(
        f"EFD vector — normalized [{len(efd_vec)} chiều] ← vector lưu DB",
        fontweight='bold')
    axes[1].set_xlabel("Chiều"); axes[1].set_ylabel("Giá trị")
    axes[1].axhline(0, color='black', lw=0.5); axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle("08 — So sánh EFD normalized vs raw", fontweight='bold')
    plt.tight_layout()
    plt.savefig(_p(out_dir, "shape_08_efd_normalized_vs_raw.png"), dpi=110,
                bbox_inches='tight')
    plt.close()
    print("    → shape_08_efd_normalized_vs_raw.png")

    print(f"  EFD vector ({len(efd_vec)} chiều): "
          f"min={efd_vec.min():.4f}  max={efd_vec.max():.4f}  "
          f"mean={efd_vec.mean():.4f}")
    print(f"  → Tổng: 9 ảnh (shape_00 ÷ shape_08)")