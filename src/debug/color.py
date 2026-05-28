import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import circmean

from src.debug.utils import save_img


def _get_path(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def debug_color_moments(img: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    """
    Tạo 4 file PNG debug cho đặc trưng color_moments (HSV).

    Parameters
    ----------
    img     : ảnh BGR gốc (np.ndarray, uint8)
    mask    : mask nhị phân của vùng lá (np.ndarray, uint8)
    out_dir : thư mục lưu file PNG
    """
    print("\n[DEBUG COLOR] Color Moments (HSV)")

    # ── Chuyển sang HSV và áp mask ──────────────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mhsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    h_ch, s_ch, v_ch = cv2.split(mhsv)

    # ── Tính raw moments (tái tính để tránh double-normalize) ───────────────
    channels_px = []
    raw_moments = []
    channel_names = ['H', 'S', 'V']
    for i, ch in enumerate(cv2.split(mhsv)):
        px = ch[mask > 0].astype(np.float64)
        channels_px.append(px)
        if len(px) == 0:
            raw_moments.extend([0.0, 0.0, 0.0])
            continue
        mean = circmean(px, high=179, low=0) if i == 0 else float(np.mean(px))
        std  = float(np.std(px))
        skew = float(np.cbrt(((px - mean) ** 3).mean())) if std > 1e-6 else 0.0
        raw_moments.extend([mean, std, skew])

    raw_vec = np.array(raw_moments, dtype=np.float32)

    # L2 normalize
    norm = float(np.linalg.norm(raw_vec))
    norm_vec = raw_vec / norm if norm > 1e-10 else raw_vec

    # ── FILE 1: color_01_hsv_channels.png ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    channel_data  = [h_ch, s_ch, v_ch]
    channel_cmaps = ['hsv', 'YlOrRd', 'gray']
    channel_titles = ["Kênh H (Hue)", "Kênh S (Saturation)", "Kênh V (Value)"]

    for ax, ch_img, cmap, title in zip(axes, channel_data, channel_cmaps, channel_titles):
        ax.imshow(ch_img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='white', pad=8)
        ax.axis('off')

    plt.suptitle("HSV Channels — Vùng lá (masked)", fontsize=13,
                 fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    out1 = _get_path(out_dir, "color_01_hsv_channels.png")
    plt.savefig(out1, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("    → color_01_hsv_channels.png")

    # ── FILE 2: color_02_raw_moments.png ────────────────────────────────────
    labels_9 = ["H_mean", "H_std", "H_skew",
                 "S_mean", "S_std", "S_skew",
                 "V_mean", "V_std", "V_skew"]
    group_colors = (
        ['#4C9BE8'] * 3 +   # H — xanh lam
        ['#E88A4C'] * 3 +   # S — cam
        ['#4CE87A'] * 3      # V — xanh lá
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(range(9), raw_vec, color=group_colors,
                  edgecolor='#ffffff22', alpha=0.9, zorder=3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, zorder=2)

    for bar, val in zip(bars, raw_vec):
        ypos = bar.get_height() + (abs(bar.get_height()) * 0.03 + 0.5)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha='center', va='bottom',
                fontsize=8.5, color='white', fontweight='bold')

    ax.set_xticks(range(9))
    ax.set_xticklabels(labels_9, fontsize=10, color='#aab4c8')
    ax.set_title("Color Moments — Giá trị thô (trước normalize)",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.set_ylabel("Giá trị", color='#aab4c8')
    ax.tick_params(axis='y', colors='#aab4c8')
    ax.grid(axis='y', alpha=0.2, color='white', zorder=1)
    ax.spines[:].set_color('#333355')

    # Legend nhóm
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#4C9BE8', label='H (Hue)'),
        Patch(facecolor='#E88A4C', label='S (Saturation)'),
        Patch(facecolor='#4CE87A', label='V (Value)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right',
              facecolor='#1a1a2e', edgecolor='#555577',
              labelcolor='white', fontsize=9)

    plt.tight_layout()
    out2 = _get_path(out_dir, "color_02_raw_moments.png")
    plt.savefig(out2, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("    → color_02_raw_moments.png")

    # ── FILE 3: color_03_normalized_vector.png ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(range(9), norm_vec, color=group_colors,
                  edgecolor='#ffffff22', alpha=0.9, zorder=3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, zorder=2)

    for bar, val in zip(bars, norm_vec):
        ypos = bar.get_height() + (abs(bar.get_height()) * 0.03 + 0.005)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha='center', va='bottom',
                fontsize=8.5, color='white', fontweight='bold')

    ax.set_xticks(range(9))
    ax.set_xticklabels(labels_9, fontsize=10, color='#aab4c8')
    ax.set_title("Vector color_moments (9D) lưu vào DB — L2 normalized",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.set_ylabel("Giá trị (normalized)", color='#aab4c8')
    ax.tick_params(axis='y', colors='#aab4c8')
    ax.grid(axis='y', alpha=0.2, color='white', zorder=1)
    ax.spines[:].set_color('#333355')

    # Annotation ‖v‖ góc trên phải
    ax.text(0.98, 0.96, f"‖v‖ = {norm:.4f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='#f0e68c',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                      edgecolor='#f0e68c', alpha=0.85))

    ax.legend(handles=legend_elems, loc='upper left',
              facecolor='#1a1a2e', edgecolor='#555577',
              labelcolor='white', fontsize=9)

    plt.tight_layout()
    out3 = _get_path(out_dir, "color_03_normalized_vector.png")
    plt.savefig(out3, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("    → color_03_normalized_vector.png")

    # ── FILE 4: color_04_hue_histogram.png ──────────────────────────────────
    hue_px = channels_px[0]  # pixel Hue của vùng lá
    H_mean     = float(raw_moments[0])   # circmean
    arith_mean = float(np.mean(hue_px)) if len(hue_px) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    ax.hist(hue_px, bins=36, range=[0, 179],
            color='#4C9BE8', edgecolor='#ffffff22', alpha=0.75, zorder=3,
            label='Phân bố pixel Hue')

    ax.axvline(H_mean, color='red', linestyle='--', linewidth=1.8,
               label=f"circmean={H_mean:.1f}°", zorder=4)
    ax.axvline(arith_mean, color='#4CE87A', linestyle='--', linewidth=1.8,
               label=f"arith_mean={arith_mean:.1f}°", zorder=4)

    ax.set_xlabel("Hue (0–179)", color='#aab4c8', fontsize=11)
    ax.set_ylabel("Số pixel", color='#aab4c8', fontsize=11)
    ax.set_title("Phân bố Hue — Circular vs Arithmetic Mean",
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax.tick_params(colors='#aab4c8')
    ax.spines[:].set_color('#333355')
    ax.grid(axis='y', alpha=0.2, color='white', zorder=1)
    ax.legend(facecolor='#1a1a2e', edgecolor='#555577',
              labelcolor='white', fontsize=10)

    plt.tight_layout()
    out4 = _get_path(out_dir, "color_04_hue_histogram.png")
    plt.savefig(out4, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("    → color_04_hue_histogram.png")
