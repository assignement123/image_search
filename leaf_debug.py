"""
leaf_debug.py  —  Debug chi tiết từng bước trích xuất đặc trưng lá
════════════════════════════════════════════════════════════════════
Yêu cầu: leaf_extract.py phải nằm CÙNG THƯ MỤC

CÁCH DÙNG:
  python leaf_debug.py <ảnh_lá>
  python leaf_debug.py <ảnh_lá> --out <thư_mục_output>

OUTPUT (~41 ảnh PNG trong --out, mặc định: debug_out/):
  step1_00..08  — Tiền xử lý: grayscale, Otsu INV, morph, contour, mask
  step2a_00..07 — EFD: contour gốc, resample, tái tạo 5 bậc, biên độ
  step2b_00..05 — GLCM: equalize, quantize, matrix, 5 properties
  step2c_00..06 — Color: HSV channels, histogram H/S/V, bar chart moments
  step2d_00..11 — Gân lá: CLAHE, bilateral, Canny, Sobel, angle histogram
  step3_00      — Tổng hợp 4 vector đặc trưng (bar chart)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from pyefd import elliptic_fourier_descriptors
from scipy.stats import circmean
import os
import sys
import argparse

# ── Import toàn bộ logic từ leaf_extract.py ──────────────────────
try:
    from leaf_extract import (
        preprocess_leaf, _resample_contour,
        extract_efd, extract_glcm, extract_color_moments,
        extract_vein_features, extract_features,
        HARMONICS, N_RESAMPLE,
        GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS,
        DIM_EFD, DIM_GLCM, DIM_COLOR, DIM_VEIN,
    )
except ImportError:
    print("[LỖI] Không tìm thấy leaf_extract.py — đặt cùng thư mục!")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH LƯU ẢNH
# ════════════════════════════════════════════════════════════════════

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
    ax.set_xlabel(xlabel); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"    → {os.path.basename(path)}")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 1 — TIỀN XỬ LÝ  (9 ảnh)
# ════════════════════════════════════════════════════════════════════

def debug_preprocess(image_path: str, out_dir: str) -> None:
    print("\n[BƯỚC 1] Tiền xử lý ảnh (nền trắng)")
    p = lambda n: os.path.join(out_dir, n)

    img = cv2.imread(image_path)

    # 00 - Ảnh gốc
    _save("00 — Ảnh gốc", img, p("step1_00_original.png"))

    # 01 - Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save("01 — Ảnh xám (Grayscale)", gray, p("step1_01_gray.png"))

    # 02 - Histogram grayscale với ngưỡng Otsu
    thresh_val, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gray.flatten(), bins=64, range=(0, 255),
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(thresh_val, color='red', lw=2.5, ls='--',
               label=f'Ngưỡng Otsu = {thresh_val:.0f}')
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000],
                     0, thresh_val, alpha=0.12, color='orange',
                     label='Vùng lá (< ngưỡng)')
    ax.set_title("02 — Histogram & ngưỡng Otsu", fontweight='bold')
    ax.set_xlabel('Mức xám'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step1_02_otsu_threshold.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step1_02_otsu_threshold.png  (ngưỡng={thresh_val:.0f})")

    # 03 - Binary INV (sau Otsu)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _save("03 — Nhị phân Otsu INV (lá=trắng, nền=đen)", binary,
          p("step1_03_binary_inv.png"))

    # 04 - MORPH_OPEN
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _save("04 — MORPH_OPEN (xoá nhiễu nhỏ)", opened, p("step1_04_morph_open.png"))

    # 05 - MORPH_CLOSE
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    _save("05 — MORPH_CLOSE (lấp lỗ hổng)", closed, p("step1_05_morph_close.png"))

    # 06 - Contour trên ảnh gốc
    img2, contour, gray2, mask, leaf_area = preprocess_leaf(image_path)
    contour_vis = cv2.drawContours(
        img.copy(), [contour.astype(np.int32)], -1, (0, 255, 0), 2)
    _save(f"06 — Contour lớn nhất ({len(contour)} điểm  |  CCW)",
          contour_vis, p("step1_06_contour.png"))

    # 07 - Mask fill
    _save(f"07 — Mask lá (diện tích = {leaf_area:,} px)", mask,
          p("step1_07_mask.png"))

    # 08 - Vùng lá cắt ra
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
    plt.savefig(p("step1_08_mask_and_crop.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step1_08_mask_and_crop.png")
    print(f"  → Tổng: 9 ảnh (step1_00 ÷ step1_08)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2A — EFD  (8 ảnh)
# ════════════════════════════════════════════════════════════════════
def debug_efd(contour: np.ndarray, out_dir: str) -> None:
    print("\n[BƯỚC 2A] EFD — Hình dạng biên lá")
    p = lambda n: os.path.join(out_dir, n)

    contour = np.asarray(contour, np.float64).reshape(-1, 2)

    # ─────────────────────────────
    # 00 - Contour gốc
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(contour[:,0], contour[:,1], 'b-', lw=1.5)
    ax.plot(contour[0,0], contour[0,1], 'ro', ms=8)
    ax.set_title(f"00 — Contour gốc ({len(contour)} điểm)", fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(p("step2a_00_contour_raw.png"), dpi=110)
    plt.close()

    # ─────────────────────────────
    # 01 - Resample contour
    # ─────────────────────────────
    rs = _resample_contour(contour, N_RESAMPLE)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(rs[:,0], rs[:,1], 'g-', lw=1.5)
    ax.scatter(rs[::30,0], rs[::30,1], c='red', s=20)
    ax.set_title(f"01 — Resample ({N_RESAMPLE} điểm)", fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(p("step2a_01_resampled.png"), dpi=110)
    plt.close()

    # ─────────────────────────────
    # Tính EFD
    # ─────────────────────────────
    coeffs = elliptic_fourier_descriptors(
        rs,
        order=HARMONICS,
        normalize=False
    )

    # centroid chuẩn
    A0 = rs[:,0].mean()
    C0 = rs[:,1].mean()

    t = np.linspace(0,1,N_RESAMPLE)

    # ─────────────────────────────
    # Hàm reconstruct chuẩn
    # ─────────────────────────────
    def reconstruct(n_max):

        x = np.full_like(t, A0)
        y = np.full_like(t, C0)

        for n in range(n_max):

            an, bn, cn, dn = coeffs[n]

            k = n + 1   # harmonic index thực

            x += an*np.cos(2*np.pi*k*t) + bn*np.sin(2*np.pi*k*t)
            y += cn*np.cos(2*np.pi*k*t) + dn*np.sin(2*np.pi*k*t)

        return x, y

    cx = rs[:,0]
    cy = rs[:,1]

    # ─────────────────────────────
    # Tái tạo các bậc
    # ─────────────────────────────
    for i, n_h in enumerate([1,3,6,12,HARMONICS]):

        x,y = reconstruct(n_h)

        fig, axes = plt.subplots(1,2,figsize=(12,4))

        # Không gian Fourier
        axes[0].plot(x,y,'b-',lw=2)
        axes[0].set_title(f'EFD space (n={n_h})',fontweight='bold')
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis()
        axes[0].grid(alpha=0.3)

        # Pixel space
        axes[1].plot(cx,cy,'g--',lw=1.5,label='Contour gốc')
        axes[1].plot(x,y,'b-',lw=2,label=f'Reconstruct n={n_h}')
        axes[1].set_title('So sánh pixel space',fontweight='bold')
        axes[1].set_aspect('equal')
        axes[1].invert_yaxis()
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fname=f"step2a_{i+2:02d}_reconstruct_n{n_h}.png"

        plt.suptitle(f"Tái tạo EFD bậc {n_h}",fontweight='bold')
        plt.tight_layout()
        plt.savefig(p(fname),dpi=110)
        plt.close()

        print(f"    → {fname}")

    # ─────────────────────────────
    # Amplitude plot
    # ─────────────────────────────
    amps = [
        np.sqrt(sum(c**2 for c in coeffs[n]))
        for n in range(1,len(coeffs))
    ]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(range(1,len(amps)+1), amps)

    ax.set_xlabel("Harmonic n")
    ax.set_ylabel("Amplitude")
    ax.set_title("07 — Biên độ EFD theo bậc",fontweight='bold')

    ax.grid(axis='y',alpha=0.3)

    plt.tight_layout()
    plt.savefig(p("step2a_07_amplitudes.png"),dpi=110)
    plt.close()

    print("    → step2a_07_amplitudes.png")

# ════════════════════════════════════════════════════════════════════
# BƯỚC 2B — GLCM  (6 ảnh)
# ════════════════════════════════════════════════════════════════════

def debug_glcm(gray_img: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    print("\n[BƯỚC 2B] GLCM — Texture bề mặt lá")
    p = lambda n: os.path.join(out_dir, n)

    # 00 - Ảnh xám cắt vùng lá
    masked = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    _save("00 — Ảnh xám, cắt vùng lá", masked, p("step2b_00_masked_gray.png"))

    # 01 - Equalize (cải tiến so với bản gốc)
    eq = cv2.equalizeHist(masked)
    _save("01 — Equalize histogram (tăng phân biệt texture)", eq,
          p("step2b_01_equalized.png"))

    # 02 - Histogram so sánh trước/sau equalize
    px_before = masked[mask > 0].flatten().astype(float)
    px_after  = eq[mask > 0].flatten().astype(float)
    _save_hist("02 — Histogram trước / sau Equalize",
               [px_before, px_after], ['Gốc', 'Sau Equalize'],
               ['steelblue', 'coral'], p("step2b_02_hist_equalize.png"))

    # 03 - Quantize
    resized = cv2.resize(eq, (256, 256))
    q = np.clip((resized // (256 // GLCM_LEVELS)).astype(np.uint8),
                0, GLCM_LEVELS - 1)
    _save(f"03 — Quantize {GLCM_LEVELS} mức xám",
          (q * (255 // GLCM_LEVELS)).astype(np.uint8), p("step2b_03_quantized.png"))

    # 04 - Ma trận GLCM 4 góc
    glcm = graycomatrix(q, distances=[3], angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    angle_names = ['0°', '45°', '90°', '135°']
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        im = ax.imshow(glcm[:, :, 0, i], cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'd=3, góc={name}', fontweight='bold')
    plt.suptitle("04 — Ma trận GLCM theo 4 góc (d=3)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2b_04_glcm_matrix.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2b_04_glcm_matrix.png")

    # 05 - 5 properties theo 4 khoảng cách
    glcm_full = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANGLES,
                             levels=GLCM_LEVELS, symmetric=True, normed=True)
    props  = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, prop, clr in zip(axes, props, colors):
        vals = graycoprops(glcm_full, prop).mean(axis=1)
        ax.bar([f'd={d}' for d in GLCM_DIST], vals, color=clr, alpha=0.85,
               edgecolor='white')
        ax.set_title(prop.capitalize(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for j, v in enumerate(vals):
            ax.text(j, v + abs(v)*0.03, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=7.5)
    plt.suptitle("05 — 5 thuộc tính GLCM × 4 khoảng cách", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2b_05_properties.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2b_05_properties.png")

    vec = extract_glcm(gray_img, mask)
    print(f"  Vector GLCM ({len(vec)} chiều): {vec.round(4)}")
    print(f"  → Tổng: 6 ảnh (step2b_00 ÷ step2b_05)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2C — COLOR MOMENTS  (7 ảnh)
# ════════════════════════════════════════════════════════════════════

def debug_color_moments(img: np.ndarray, mask: np.ndarray, out_dir: str) -> None:
    print("\n[BƯỚC 2C] Color Moments — Màu sắc lá (HSV)")
    p = lambda n: os.path.join(out_dir, n)

    # 00 - Ảnh gốc
    _save("00 — Ảnh gốc", img, p("step2c_00_original.png"))

    # 01 - 3 kênh HSV
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mhsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    h, s, v = cv2.split(mhsv)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(cv2.applyColorMap((h*(255//179)).astype(np.uint8),
                                      cv2.COLORMAP_HSV))
    axes[0].set_title("Kênh H (Hue 0-179)", fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(s, cmap='Blues')
    axes[1].set_title("Kênh S (Saturation)", fontweight='bold'); axes[1].axis('off')
    axes[2].imshow(v, cmap='Greens')
    axes[2].set_title("Kênh V (Value/Brightness)", fontweight='bold'); axes[2].axis('off')
    plt.suptitle("01 — 3 kênh HSV trong vùng lá", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2c_01_hsv_channels.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_01_hsv_channels.png")

    px_h = h[mask > 0].astype(float)
    px_s = s[mask > 0].astype(float)
    px_v = v[mask > 0].astype(float)
    mean_h = circmean(px_h, high=179, low=0) if len(px_h) > 0 else 0.0

    # 02 - Histogram H với circular mean
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(px_h, bins=36, range=(0,179), color='coral', alpha=0.8, edgecolor='white')
    ax.axvline(mean_h, color='red', lw=2.5, ls='--',
               label=f'circmean = {mean_h:.1f}')
    ax.axvline(np.mean(px_h), color='navy', lw=2, ls=':',
               label=f'mean thường = {np.mean(px_h):.1f}')
    ax.set_title("02 — Histogram kênh H  (circular mean vs mean thường)",
                 fontweight='bold')
    ax.set_xlabel('H (0-179)'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step2c_02_hist_H.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_02_hist_H.png")

    # 03 - Histogram S
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(px_s, bins=32, range=(0,255), color='dodgerblue', alpha=0.8,
            edgecolor='white')
    ax.axvline(np.mean(px_s), color='red', lw=2.5, ls='--',
               label=f'mean={np.mean(px_s):.1f}  std={np.std(px_s):.1f}')
    ax.set_title("03 — Histogram kênh S (Saturation)", fontweight='bold')
    ax.set_xlabel('S (0-255)'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step2c_03_hist_S.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_03_hist_S.png")

    # 04 - Histogram V
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(px_v, bins=32, range=(0,255), color='seagreen', alpha=0.8,
            edgecolor='white')
    ax.axvline(np.mean(px_v), color='red', lw=2.5, ls='--',
               label=f'mean={np.mean(px_v):.1f}  std={np.std(px_v):.1f}')
    ax.set_title("04 — Histogram kênh V (Value/Brightness)", fontweight='bold')
    ax.set_xlabel('V (0-255)'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step2c_04_hist_V.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_04_hist_V.png")

    # 05 - 3 moments cho mỗi kênh (scatter overview)
    vec    = extract_color_moments(img, mask)
    labels = ['H_mean','H_std','H_skew','S_mean','S_std','S_skew','V_mean','V_std','V_skew']
    clrs   = ['#e74c3c']*3 + ['#3498db']*3 + ['#2ecc71']*3
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, vec, color=clrs, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vec):
        off = abs(val) * 0.04 + 0.5
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (off if val >= 0 else -off - 1.5),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title("05 — Vector Color Moments (9 chiều)\n"
                 "Đỏ: H | Xanh dương: S | Xanh lá: V",
                 fontweight='bold')
    ax.set_ylabel('Giá trị'); ax.axhline(0, color='black', lw=0.8)
    ax.grid(axis='y', alpha=0.3); plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(p("step2c_05_moments_bar.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_05_moments_bar.png")

    print(f"  Vector màu ({len(vec)} chiều): {vec.round(2)}")
    print(f"  → Tổng: 6 ảnh (step2c_00 ÷ step2c_05)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2D — VEIN FEATURES  (12 ảnh)
# ════════════════════════════════════════════════════════════════════

def debug_vein_features(img: np.ndarray, mask: np.ndarray,
                        leaf_area: int, out_dir: str) -> None:
    print("\n[BƯỚC 2D] Gân lá — Mật độ & hướng gân")
    p = lambda n: os.path.join(out_dir, n)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save("00 — Ảnh gốc", img,  p("step2d_00_original.png"))
    _save("01 — Ảnh xám", gray, p("step2d_01_gray.png"))

    # CLAHE
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _save("02 — CLAHE (tăng tương phản cục bộ)", enhanced, p("step2d_02_clahe.png"))

    # Histogram CLAHE
    _save_hist("03 — Histogram xám: trước / sau CLAHE",
               [gray.flatten().astype(float), enhanced.flatten().astype(float)],
               ['Gốc', 'Sau CLAHE'], ['steelblue', 'coral'],
               p("step2d_03_hist_clahe.png"))

    # Bilateral filter
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _save("04 — Bilateral filter (giữ cạnh, khử nhiễu)", smooth,
          p("step2d_04_bilateral.png"))

    # Diff = nhiễu đã lọc
    diff = cv2.absdiff(enhanced, smooth)
    _save("05 — CLAHE − Bilateral = nhiễu bị khử", diff,
          p("step2d_05_diff_noise.png"))

    # Canny
    canny = cv2.Canny(smooth, 20, 80)
    _save("06 — Cạnh Canny (threshold 20 / 80)", canny, p("step2d_06_canny.png"))

    # Sobel magnitude
    sx   = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=5)
    sy   = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=5)
    smag = cv2.normalize(np.hypot(sx, sy), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    _save("07 — Độ lớn Sobel (ksize=5)", smag, p("step2d_07_sobel_mag.png"))

    # Kết hợp
    edges = cv2.addWeighted(smag, 0.6, canny, 0.4, 0)
    _save("08 — Kết hợp Sobel×0.6 + Canny×0.4", edges, p("step2d_08_combined.png"))

    # Cắt vùng lá
    vein_raw = cv2.bitwise_and(edges, edges, mask=mask)
    _save("09 — Cắt vùng lá", vein_raw, p("step2d_09_vein_raw.png"))

    # Morphology
    k3   = np.ones((3, 3), np.uint8)
    vein = cv2.morphologyEx(vein_raw, cv2.MORPH_OPEN, k3, iterations=1)
    vein = cv2.dilate(vein, k3, iterations=1)
    density = float((vein > 0).sum() / leaf_area) if leaf_area > 0 else 0.0
    _save(f"10 — Sau morphology  density={density:.4f}", vein,
          p("step2d_10_vein_clean.png"))

    # Angle overlay + histogram
    angle_map = np.arctan2(sy, sx) * 180.0 / np.pi % 180.0
    angle_vis = (angle_map / 180.0 * 255).astype(np.uint8)
    angle_clr = cv2.applyColorMap(angle_vis, cv2.COLORMAP_HSV)
    vein3     = cv2.cvtColor((vein > 0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    overlay   = cv2.bitwise_and(angle_clr, vein3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Màu = hướng gân (HSV color map)", fontweight='bold')
    axes[0].axis('off')

    if (vein > 0).any():
        angles  = angle_map[vein > 0]
        hist, edges2 = np.histogram(angles, bins=8, range=(0, 180), density=True)
        bin_ctr = (edges2[:-1] + edges2[1:]) / 2
        axes[1].bar(bin_ctr, hist, width=20, color='steelblue',
                    edgecolor='white', alpha=0.85)
        peak = bin_ctr[np.argmax(hist)]
        axes[1].axvline(peak, color='red', ls='--', lw=2,
                        label=f'Hướng chủ đạo: {peak:.0f}°')
        axes[1].legend()
    axes[1].set_xlabel('Góc (°)'); axes[1].set_ylabel('Mật độ')
    axes[1].set_title(f"Histogram góc gân (8 bin)\ndensity={density:.4f}",
                      fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle("11 — Phân bố hướng gân lá", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2d_11_angle_hist.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2d_11_angle_hist.png")

    vec = extract_vein_features(img, mask, leaf_area)
    print(f"  Vector gân ({len(vec)} chiều): {vec.round(4)}")
    print(f"  → Tổng: 12 ảnh (step2d_00 ÷ step2d_11)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 3 — TÓM TẮT VECTOR  (1 ảnh)
# ════════════════════════════════════════════════════════════════════

def debug_summary(image_path: str, out_dir: str) -> None:
    print("\n[BƯỚC 3] Tổng hợp vector đặc trưng cuối cùng")
    feats = extract_features(image_path)

    total = 0
    print(f"\n  {'─'*56}")
    for k, v in feats.items():
        print(f"  {k:<6} | {len(v):>3} chiều | "
              f"min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
        total += len(v)
    print(f"  {'─'*56}")
    print(f"  {'TỔNG':<6} | {total:>3} chiều  (EFD={DIM_EFD} + "
          f"GLCM={DIM_GLCM} + Color={DIM_COLOR} + Vein={DIM_VEIN})")
    print(f"  {'─'*56}\n")

    # Bar chart 4 nhóm
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    configs = [
        (f"EFD ({DIM_EFD} chiều)",   feats["efd"],   '#3498db'),
        (f"GLCM ({DIM_GLCM} chiều)", feats["glcm"],  '#e74c3c'),
        (f"Color ({DIM_COLOR} chiều)", feats["color"], '#2ecc71'),
        (f"Vein ({DIM_VEIN} chiều)",  feats["vein"],  '#9b59b6'),
    ]
    for ax, (title, vec, clr) in zip(axes.flat, configs):
        ax.bar(range(len(vec)), vec, color=clr, alpha=0.8, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Chiều'); ax.set_ylabel('Giá trị')
        ax.axhline(0, color='black', lw=0.5); ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f"Tổng hợp 4 vector đặc trưng ({total} chiều)\n"
        f"Ảnh: {os.path.basename(image_path)}",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "step3_00_feature_summary.png")
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step3_00_feature_summary.png")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Debug chi tiết từng bước trích xuất đặc trưng lá (~41 ảnh PNG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python leaf_debug.py leaf.jpg
  python leaf_debug.py leaf.jpg --out my_debug/
        """)
    ap.add_argument("image", help="Ảnh lá cần debug")
    ap.add_argument("--out", default="debug_out",
                    help="Thư mục lưu ảnh (mặc định: debug_out/)")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[LỖI] Không tìm thấy: {args.image}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  DEBUG TRÍCH XUẤT ĐẶC TRƯNG LÁ CÂY")
    print(f"  Ảnh     : {os.path.basename(args.image)}")
    print(f"  Output  : {os.path.abspath(args.out)}/")
    print(f"  Tham số : harmonics={HARMONICS}, n_resample={N_RESAMPLE}")
    print(f"            GLCM: distances={GLCM_DIST}, levels={GLCM_LEVELS}")
    print(f"{'═'*62}")

    img, contour, gray, mask, leaf_area = preprocess_leaf(args.image)
    print(f"\n  Contour  : {len(contour)} điểm")
    print(f"  Leaf area: {leaf_area:,} px")

    debug_preprocess(args.image, args.out)
    debug_efd(contour, args.out)
    debug_glcm(gray, mask, args.out)
    debug_color_moments(img, mask, args.out)
    debug_vein_features(img, mask, leaf_area, args.out)
    debug_summary(args.image, args.out)

    saved = sorted(f for f in os.listdir(args.out) if f.endswith('.png'))
    print(f"\n{'═'*62}")
    print(f"  HOÀN TẤT — {len(saved)} ảnh PNG đã lưu vào '{args.out}/'")
    print(f"  step1_*   Tiền xử lý   ({sum(1 for f in saved if f.startswith('step1'))} ảnh)")
    print(f"  step2a_*  EFD           ({sum(1 for f in saved if f.startswith('step2a'))} ảnh)")
    print(f"  step2b_*  GLCM          ({sum(1 for f in saved if f.startswith('step2b'))} ảnh)")
    print(f"  step2c_*  Color Moments ({sum(1 for f in saved if f.startswith('step2c'))} ảnh)")
    print(f"  step2d_*  Gân lá        ({sum(1 for f in saved if f.startswith('step2d'))} ảnh)")
    print(f"  step3_*   Tổng hợp      ({sum(1 for f in saved if f.startswith('step3'))} ảnh)")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()