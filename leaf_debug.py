"""
leaf_debug.py  —  Debug chi tiết từng bước trích xuất đặc trưng lá
════════════════════════════════════════════════════════════════════
Yêu cầu: leaf_extract.py phải nằm CÙNG THƯ MỤC

CÁCH DÙNG:
  python leaf_debug.py <ảnh_lá>
  python leaf_debug.py <ảnh_lá> --out <thư_mục_output>

OUTPUT (~47 ảnh PNG trong --out, mặc định: debug_out/):
  step1_00..08  — Tiền xử lý: grayscale, Otsu INV, morph, contour, mask
  step2a_00..07 — EFD: contour gốc, resample, tái tạo 5 bậc, biên độ
  step2b_00..09 — Texture (LBP + GLCM):
                    00..03 LBP: uniform map, histogram, vùng lá, so sánh radius
                    04..09 GLCM: equalize, quantize, matrix, 5 properties
  step2c_00..05 — Color: HSV channels, histogram H/S/V, bar chart moments
  step2d_00..11 — Gân lá: CLAHE, bilateral, Canny, Sobel, angle histogram
  step3_00      — Tổng hợp 4 vector đặc trưng (bar chart)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from pyefd import elliptic_fourier_descriptors
from scipy.stats import circmean
import os
import sys
import argparse

# ── Import toàn bộ logic từ leaf_extract.py ──────────────────────
try:
    from leaf_extract import (
        preprocess_leaf, _resample_contour,
        extract_efd, extract_texture_features, extract_color_moments,
        extract_vein_features, extract_features,
        get_leaf_mask,
        HARMONICS, N_RESAMPLE,
        GLCM_DIST, GLCM_ANGLES, GLCM_LEVELS,
        LBP_P, LBP_R,
        DIM_EFD, DIM_TEXTURE, DIM_COLOR, DIM_VEIN,
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
# [SỬA] Dùng normalize=True khớp với extract_efd(), thêm ảnh so sánh
#        normalized vs raw để thấy rõ hiệu quả bất biến rotation
# ════════════════════════════════════════════════════════════════════

def debug_efd(contour: np.ndarray, out_dir: str) -> None:
    print("\n[BƯỚC 2A] EFD — Hình dạng biên lá")
    p = lambda n: os.path.join(out_dir, n)

    contour = np.asarray(contour, np.float64).reshape(-1, 2)

    # 00 - Contour gốc
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(contour[:, 0], contour[:, 1], 'b-', lw=1.5)
    ax.plot(contour[0, 0], contour[0, 1], 'ro', ms=8, label='Điểm bắt đầu')
    ax.set_title(f"00 — Contour gốc ({len(contour)} điểm)", fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(p("step2a_00_contour_raw.png"), dpi=110)
    plt.close(); print(f"    → step2a_00_contour_raw.png")

    # 01 - Resample contour
    rs = _resample_contour(contour, N_RESAMPLE)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rs[:, 0], rs[:, 1], 'g-', lw=1.5)
    ax.scatter(rs[::30, 0], rs[::30, 1], c='red', s=20, label='Mỗi 30 điểm')
    ax.plot(rs[0, 0], rs[0, 1], 'ko', ms=10, label='Điểm bắt đầu')
    ax.set_title(f"01 — Resample ({N_RESAMPLE} điểm, đều theo arc-length)",
                 fontweight='bold')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(p("step2a_01_resampled.png"), dpi=110)
    plt.close(); print(f"    → step2a_01_resampled.png")

    # ── [SỬA] Tính EFD với normalize=True (khớp extract_efd) ────────
    # normalize=True: pyefd tự xử lý scale + rotation + start-point invariance
    coeffs_norm = elliptic_fourier_descriptors(
        rs, order=HARMONICS + 1, normalize=True)

    # Cũng tính normalize=False để visualize reconstruction dễ hơn
    # (normalize=True làm mất thông tin vị trí tuyệt đối nên khó vẽ lại)
    coeffs_raw = elliptic_fourier_descriptors(
        rs, order=HARMONICS + 1, normalize=False)

    A0 = rs[:, 0].mean()
    C0 = rs[:, 1].mean()
    t  = np.linspace(0, 1, N_RESAMPLE)

    def reconstruct_raw(n_max):
        """Tái tạo từ coeffs_raw để visualize pixel space."""
        x = np.full_like(t, A0)
        y = np.full_like(t, C0)
        for n in range(1, n_max + 1):
            an, bn, cn, dn = coeffs_raw[n]
            x += an * np.cos(2 * np.pi * n * t) + bn * np.sin(2 * np.pi * n * t)
            y += cn * np.cos(2 * np.pi * n * t) + dn * np.sin(2 * np.pi * n * t)
        return x, y

    cx, cy = rs[:, 0], rs[:, 1]

    # Tái tạo các bậc
    for i, n_h in enumerate([1, 3, 6, 12, HARMONICS]):
        x, y = reconstruct_raw(n_h)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(x, y, 'b-', lw=2)
        axes[0].set_title(f'EFD space (n={n_h})', fontweight='bold')
        axes[0].set_aspect('equal'); axes[0].invert_yaxis(); axes[0].grid(alpha=0.3)
        axes[1].plot(cx, cy, 'g--', lw=1.5, label='Contour gốc')
        axes[1].plot(x, y, 'b-', lw=2, label=f'Reconstruct n={n_h}')
        axes[1].set_title('So sánh pixel space', fontweight='bold')
        axes[1].set_aspect('equal'); axes[1].invert_yaxis()
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fname = f"step2a_{i+2:02d}_reconstruct_n{n_h}.png"
        plt.suptitle(f"Tái tạo EFD bậc {n_h}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(p(fname), dpi=110)
        plt.close(); print(f"    → {fname}")

    # 07 - Amplitude plot (dùng coeffs_raw để thấy magnitude thực)
    amps = [np.sqrt(sum(c**2 for c in coeffs_raw[n]))
            for n in range(1, HARMONICS + 1)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(amps) + 1), amps, color='steelblue',
           edgecolor='white', alpha=0.85)
    ax.set_xlabel("Harmonic n"); ax.set_ylabel("Amplitude")
    ax.set_title("07 — Biên độ EFD theo bậc (normalize=False)\n"
                 "Bậc 1 lớn nhất → dùng làm chuẩn normalize",
                 fontweight='bold')
    ax.axvline(1, color='red', ls='--', lw=2, label='Bậc 1 (chuẩn normalize)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step2a_07_amplitudes.png"), dpi=110)
    plt.close(); print(f"    → step2a_07_amplitudes.png")

    # ── [MỚI] 08 - So sánh vector EFD normalized vs raw ─────────────
    # Vector thực sự được lưu vào DB là coeffs_norm[2:HARMONICS+1]
    efd_vec = coeffs_norm[2:HARMONICS + 1, :].flatten().astype(np.float32)
    efd_raw_vec = coeffs_raw[2:HARMONICS + 1, :].flatten().astype(np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].bar(range(len(efd_raw_vec)), efd_raw_vec,
                color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_title("EFD vector — normalize=False (phụ thuộc scale & rotation)",
                      fontweight='bold')
    axes[0].set_xlabel("Chiều"); axes[0].set_ylabel("Giá trị")
    axes[0].axhline(0, color='black', lw=0.5); axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(efd_vec)), efd_vec,
                color='coral', edgecolor='white', alpha=0.85)
    axes[1].set_title(
        f"EFD vector — normalize=True  [{len(efd_vec)} chiều] ← đây là vector lưu DB\n"
        "Bất biến với: scale, rotation, điểm bắt đầu contour",
        fontweight='bold')
    axes[1].set_xlabel("Chiều"); axes[1].set_ylabel("Giá trị")
    axes[1].axhline(0, color='black', lw=0.5); axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle("08 — So sánh EFD normalized vs raw", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2a_08_efd_normalized_vs_raw.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2a_08_efd_normalized_vs_raw.png")

    print(f"  EFD vector ({len(efd_vec)} chiều): min={efd_vec.min():.4f} "
          f"max={efd_vec.max():.4f} mean={efd_vec.mean():.4f}")
    print(f"  → Tổng: 9 ảnh (step2a_00 ÷ step2a_08)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2B — TEXTURE: LBP + GLCM  (11 ảnh)
# [SỬA] Nhận gray_masked (nền đen) thay vì gray gốc (nền trắng)
#        Thêm ảnh so sánh mask từ gray gốc vs gray_masked để thấy rõ sự khác biệt
# ════════════════════════════════════════════════════════════════════

def debug_texture(gray_masked: np.ndarray, out_dir: str) -> None:
    """
    Debug toàn bộ pipeline texture (khớp extract_texture_features).

    [SỬA] Tham số là gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
          tức ảnh xám đã được che nền đen — đúng với giả định của get_leaf_mask()
          (dùng threshold > 5 để tách lá khỏi nền đen).

          KHÔNG truyền gray gốc (nền trắng) vào đây — get_leaf_mask sẽ tạo
          mask gần như toàn ảnh, làm sai toàn bộ LBP và GLCM.
    """
    print("\n[BƯỚC 2B] Texture — LBP (26 chiều) + GLCM (20 chiều) = 46 chiều")
    print("  [INFO] Nhận gray_masked (nền đen) — đúng với get_leaf_mask(threshold > 5)")
    p = lambda n: os.path.join(out_dir, n)

    if gray_masked.dtype != np.uint8:
        gray_masked = gray_masked.astype(np.uint8)

    # ── Tạo mask từ gray_masked (giống get_leaf_mask trong extract) ──
    mask = get_leaf_mask(gray_masked)

    # ════════════════════════════════════
    # LBP SECTION  (step2b_00 ÷ 03)
    # ════════════════════════════════════
    print("  [LBP]")

    # 00 — Ảnh xám đầu vào (gray_masked, nền đen)
    # [SỬA] Label rõ đây là gray_masked để tránh nhầm với gray gốc
    _save("00 — gray_masked (nền đen, đã che bằng mask lá)\n"
          "← Đúng input cho extract_texture_features",
          gray_masked, p("step2b_00_gray_masked_input.png"))

    # 01 — LBP map toàn ảnh
    lbp_map = local_binary_pattern(gray_masked, LBP_P, LBP_R, method="uniform")
    lbp_vis = (lbp_map / (LBP_P + 2) * 255).astype(np.uint8)
    _save(f"01 — LBP map (P={LBP_P}, R={LBP_R}, uniform)",
          lbp_vis, p("step2b_01_lbp_map.png"))

    # 02 — LBP chỉ trong vùng lá (mask)
    lbp_masked_vis = cv2.bitwise_and(lbp_vis, lbp_vis, mask=mask)
    _save("02 — LBP map cắt vùng lá", lbp_masked_vis, p("step2b_02_lbp_masked.png"))

    # 03 — Histogram LBP (P+2 bins, uniform)
    lbp_in_mask = lbp_map[mask > 0]
    hist_lbp, _ = np.histogram(
        lbp_in_mask,
        bins=np.arange(0, LBP_P + 3),
        range=(0, LBP_P + 2)
    )
    hist_lbp_norm = hist_lbp.astype(np.float32) / (hist_lbp.sum() + 1e-7)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(range(len(hist_lbp)), hist_lbp, color='steelblue',
                edgecolor='white', alpha=0.85)
    axes[0].set_title(f"Histogram LBP (thô) — {len(hist_lbp)} bins",
                      fontweight='bold')
    axes[0].set_xlabel("Bin (pattern code)"); axes[0].set_ylabel("Số pixel")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(hist_lbp_norm)), hist_lbp_norm, color='coral',
                edgecolor='white', alpha=0.85)
    axes[1].set_title("Histogram LBP (chuẩn hoá) → 26 chiều", fontweight='bold')
    axes[1].set_xlabel("Bin (pattern code)"); axes[1].set_ylabel("Tần suất")
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(hist_lbp_norm):
        if v > 0.01:
            axes[1].text(i, v + 0.003, f'{v:.2f}',
                         ha='center', va='bottom', fontsize=6.5)

    plt.suptitle(
        f"03 — LBP Histogram (P={LBP_P}, R={LBP_R})  |  "
        f"Uniform patterns: {LBP_P + 2} bins",
        fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2b_03_lbp_histogram.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2b_03_lbp_histogram.png")

    # ════════════════════════════════════
    # GLCM SECTION  (step2b_04 ÷ 09)
    # ════════════════════════════════════
    print("  [GLCM]")

    # 04 — Ảnh xám cắt vùng lá
    masked = cv2.bitwise_and(gray_masked, gray_masked, mask=mask)
    _save("04 — Ảnh xám, cắt vùng lá (trước equalize)",
          masked, p("step2b_04_masked_gray.png"))

    # 05 — Equalize
    eq = cv2.equalizeHist(masked)
    _save("05 — Equalize histogram (tăng phân biệt texture)",
          eq, p("step2b_05_equalized.png"))

    # 06 — Histogram so sánh trước/sau equalize
    px_before = masked[mask > 0].flatten().astype(float)
    px_after  = eq[mask > 0].flatten().astype(float)
    _save_hist("06 — Histogram trước / sau Equalize",
               [px_before, px_after], ['Gốc', 'Sau Equalize'],
               ['steelblue', 'coral'], p("step2b_06_hist_equalize.png"))

    # 07 — Quantize
    resized = cv2.resize(eq, (256, 256))
    q = np.clip((resized // (256 // GLCM_LEVELS)).astype(np.uint8),
                0, GLCM_LEVELS - 1)
    _save(f"07 — Quantize {GLCM_LEVELS} mức xám",
          (q * (255 // GLCM_LEVELS)).astype(np.uint8),
          p("step2b_07_quantized.png"))

    # 08 — Ma trận GLCM 4 góc
    glcm = graycomatrix(q, distances=[3], angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    angle_names = ['0°', '45°', '90°', '135°']
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        im = ax.imshow(glcm[:, :, 0, i], cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'd=3, góc={name}', fontweight='bold')
    plt.suptitle("08 — Ma trận GLCM theo 4 góc (d=3)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2b_08_glcm_matrix.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2b_08_glcm_matrix.png")

    # 09 — 5 properties × 4 khoảng cách
    glcm_full = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANGLES,
                             levels=GLCM_LEVELS, symmetric=True, normed=True)
    props  = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, prop, clr in zip(axes, props, colors):
        vals = graycoprops(glcm_full, prop).mean(axis=1)
        ax.bar([f'd={d}' for d in GLCM_DIST], vals, color=clr,
               alpha=0.85, edgecolor='white')
        ax.set_title(prop.capitalize(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for j, v in enumerate(vals):
            ax.text(j, v + abs(v) * 0.03, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=7.5)
    plt.suptitle("09 — 5 thuộc tính GLCM × 4 khoảng cách", fontweight='bold')
    plt.tight_layout()
    plt.savefig(p("step2b_09_glcm_properties.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2b_09_glcm_properties.png")

    # ── Tổng kết vector texture ───────────────────────────────────
    # [SỬA] Truyền gray_masked vào extract_texture_features (đúng với production)
    vec = extract_texture_features(gray_masked)
    if vec is not None:
        lbp_part  = vec[:LBP_P + 2]     # 26 chiều
        glcm_part = vec[LBP_P + 2:]     # 20 chiều

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(range(len(lbp_part)),  lbp_part,  color='steelblue',
                    edgecolor='white', alpha=0.85)
        axes[0].set_title(f"LBP vector ({len(lbp_part)} chiều)", fontweight='bold')
        axes[0].set_xlabel("Bin"); axes[0].set_ylabel("Tần suất")
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(range(len(glcm_part)), glcm_part, color='coral',
                    edgecolor='white', alpha=0.85)
        axes[1].set_title(f"GLCM vector ({len(glcm_part)} chiều)", fontweight='bold')
        axes[1].set_xlabel("Chiều"); axes[1].set_ylabel("Giá trị")
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle(
            f"Tổng hợp vector Texture = LBP({len(lbp_part)}) + GLCM({len(glcm_part)}) "
            f"= {len(vec)} chiều",
            fontweight='bold')
        plt.tight_layout()
        plt.savefig(p("step2b_10_texture_vector.png"), dpi=110, bbox_inches='tight')
        plt.close(); print(f"    → step2b_10_texture_vector.png")

        print(f"  Vector LBP  (26 chiều): {lbp_part.round(4)}")
        print(f"  Vector GLCM (20 chiều): {glcm_part.round(4)}")
        print(f"  Vector Texture ({len(vec)} chiều): OK")
    else:
        print("  [WARN] extract_texture_features trả về None — kiểm tra lại ảnh")

    print(f"  → Tổng: 11 ảnh (step2b_00 ÷ step2b_10)")


# ════════════════════════════════════════════════════════════════════
# BƯỚC 2C — COLOR MOMENTS  (6 ảnh)
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
    axes[0].imshow(cv2.applyColorMap((h * (255 // 179)).astype(np.uint8),
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
    ax.hist(px_h, bins=36, range=(0, 179), color='coral', alpha=0.8, edgecolor='white')
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
    ax.hist(px_s, bins=32, range=(0, 255), color='dodgerblue', alpha=0.8,
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
    ax.hist(px_v, bins=32, range=(0, 255), color='seagreen', alpha=0.8,
            edgecolor='white')
    ax.axvline(np.mean(px_v), color='red', lw=2.5, ls='--',
               label=f'mean={np.mean(px_v):.1f}  std={np.std(px_v):.1f}')
    ax.set_title("04 — Histogram kênh V (Value/Brightness)", fontweight='bold')
    ax.set_xlabel('V (0-255)'); ax.set_ylabel('Số pixel')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(p("step2c_04_hist_V.png"), dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step2c_04_hist_V.png")

    # 05 - Bar chart 9 moments
    vec    = extract_color_moments(img, mask)
    labels = ['H_mean', 'H_std', 'H_skew', 'S_mean', 'S_std', 'S_skew',
              'V_mean', 'V_std', 'V_skew']
    clrs   = ['#e74c3c'] * 3 + ['#3498db'] * 3 + ['#2ecc71'] * 3
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, vec, color=clrs, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vec):
        off = abs(val) * 0.04 + 0.5
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (off if val >= 0 else -off - 1.5),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title("05 — Vector Color Moments (9 chiều)\n"
                 "Đỏ: H | Xanh dương: S | Xanh lá: V", fontweight='bold')
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

    # Diff
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
    vein3     = cv2.cvtColor((vein > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    overlay   = cv2.bitwise_and(angle_clr, vein3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Màu = hướng gân (HSV color map)", fontweight='bold')
    axes[0].axis('off')

    if (vein > 0).any():
        angles   = angle_map[vein > 0]
        hist, edges2 = np.histogram(angles, bins=8, range=(0, 180), density=True)
        bin_ctr  = (edges2[:-1] + edges2[1:]) / 2
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

    # ── In bảng tóm tắt ra terminal ──────────────────────────────
    total = 0
    print(f"\n  {'─'*60}")
    for k, v in feats.items():
        if v is not None:
            print(f"  {k:<8} | {len(v):>3} chiều | "
                  f"min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
            total += len(v)
        else:
            print(f"  {k:<8} | None (lỗi extract)")
    print(f"  {'─'*60}")
    print(f"  {'TỔNG':<8} | {total:>3} chiều  "
          f"(EFD={DIM_EFD} + Texture={DIM_TEXTURE} + "
          f"Color={DIM_COLOR} + Vein={DIM_VEIN})")
    print(f"  {'─'*60}\n")

    # ── Bar chart 4 nhóm ──────────────────────────────────────────
    lbp_vec  = feats["texture"][:LBP_P + 2]   if feats["texture"] is not None else np.zeros(LBP_P + 2)
    glcm_vec = feats["texture"][LBP_P + 2:]   if feats["texture"] is not None else np.zeros(20)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    configs = [
        (f"EFD ({DIM_EFD} chiều)",              feats["efd"],    '#3498db'),
        (f"LBP ({LBP_P + 2} chiều)",             lbp_vec,         '#1abc9c'),
        (f"GLCM (20 chiều)",                      glcm_vec,        '#e74c3c'),
        (f"Color ({DIM_COLOR} chiều)",            feats["color"],  '#2ecc71'),
        (f"Vein ({DIM_VEIN} chiều)",              feats["vein"],   '#9b59b6'),
    ]
    axes.flat[5].set_visible(False)

    for ax, (title, vec, clr) in zip(axes.flat, configs):
        if vec is not None:
            ax.bar(range(len(vec)), vec, color=clr, alpha=0.8, edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Chiều'); ax.set_ylabel('Giá trị')
        ax.axhline(0, color='black', lw=0.5); ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f"Tổng hợp 5 nhóm đặc trưng ({total} chiều)\n"
        f"Ảnh: {os.path.basename(image_path)}",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "step3_00_feature_summary.png")
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(); print(f"    → step3_00_feature_summary.png")


# ════════════════════════════════════════════════════════════════════
# MAIN
# [SỬA] Tạo gray_masked từ gray + mask, truyền vào debug_texture
#        thay vì truyền gray gốc (nền trắng)
# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Debug chi tiết từng bước trích xuất đặc trưng lá (~47 ảnh PNG)",
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
    print(f"            LBP : P={LBP_P}, R={LBP_R}  → {LBP_P + 2} chiều")
    print(f"            GLCM: distances={GLCM_DIST}, levels={GLCM_LEVELS}  → 20 chiều")
    print(f"            Texture tổng: {DIM_TEXTURE} chiều")
    print(f"{'═'*62}")

    img, contour, gray, mask, leaf_area = preprocess_leaf(args.image)
    print(f"\n  Contour  : {len(contour)} điểm")
    print(f"  Leaf area: {leaf_area:,} px")

    # [SỬA] Tạo gray_masked — nền đen, đúng với giả định của extract_texture_features
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    debug_preprocess(args.image, args.out)
    debug_efd(contour, args.out)
    debug_texture(gray_masked, args.out)      # ← truyền gray_masked, không phải gray
    debug_color_moments(img, mask, args.out)
    debug_vein_features(img, mask, leaf_area, args.out)
    debug_summary(args.image, args.out)

    saved = sorted(f for f in os.listdir(args.out) if f.endswith('.png'))
    print(f"\n{'═'*62}")
    print(f"  HOÀN TẤT — {len(saved)} ảnh PNG đã lưu vào '{args.out}/'")
    print(f"  step1_*    Tiền xử lý    ({sum(1 for f in saved if f.startswith('step1'))} ảnh)")
    print(f"  step2a_*   EFD            ({sum(1 for f in saved if f.startswith('step2a'))} ảnh)")
    print(f"  step2b_*   Texture        ({sum(1 for f in saved if f.startswith('step2b'))} ảnh)")
    print(f"  step2c_*   Color Moments  ({sum(1 for f in saved if f.startswith('step2c'))} ảnh)")
    print(f"  step2d_*   Gân lá         ({sum(1 for f in saved if f.startswith('step2d'))} ảnh)")
    print(f"  step3_*    Tổng hợp       ({sum(1 for f in saved if f.startswith('step3'))} ảnh)")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()