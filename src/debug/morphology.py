import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features.contour import extract_morphology


def _p(out_dir: str, filename: str) -> str:
    return os.path.join(out_dir, filename)


def _save_vector(title: str, values: np.ndarray, labels: list[str], path: str, colors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor='#ffffff22', alpha=0.92, zorder=3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8, zorder=2)

    for bar, val in zip(bars, values):
        offset = abs(bar.get_height()) * 0.03 + 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.4f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='white',
            fontweight='bold',
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, color='#aab4c8')
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)
    ax.set_ylabel('Giá trị chuẩn hóa', color='#aab4c8')
    ax.tick_params(axis='y', colors='#aab4c8')
    ax.grid(axis='y', alpha=0.2, color='white', zorder=1)
    ax.spines[:].set_color('#333355')

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"    → {os.path.basename(path)}")


def debug_morphology(contour: np.ndarray, out_dir: str) -> dict:
    """Debug riêng cho morphology: contour, bounding box, convex hull và vector 4D."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n[DEBUG MORPH] Morphology — aspect ratio / circularity / solidity / extent")

    contour = np.asarray(contour, np.float64).reshape(-1, 2)
    contour_i = contour.astype(np.float32)

    if len(contour) < 3:
        print("  ⚠ Contour quá ngắn, morphology sẽ là vector 0.")
        values = np.zeros(4, dtype=np.float32)
    else:
        values = extract_morphology(contour)

    area      = float(cv2.contourArea(contour_i)) if len(contour) >= 3 else 0.0
    perimeter = float(cv2.arcLength(contour_i, True)) if len(contour) >= 2 else 0.0
    x, y, w, h = cv2.boundingRect(contour_i) if len(contour) >= 1 else (0, 0, 1, 1)
    hull      = cv2.convexHull(contour_i) if len(contour) >= 3 else contour_i.reshape(-1, 1, 2)
    hull_area = float(cv2.contourArea(hull)) if len(contour) >= 3 else 0.0

    # fitEllipse axes (dùng để hiển thị, khớp với extract_morphology)
    ellipse_info = None
    if len(contour) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour_i)
        ellipse_info = {"cx": cx, "cy": cy, "MA": MA, "ma": ma, "angle": angle}

    print(f"  contour_points = {len(contour)}")
    print(f"  bbox = (x={x}, y={y}, w={w}, h={h})")
    print(f"  area = {area:.2f}  perimeter = {perimeter:.2f}  hull_area = {hull_area:.2f}")
    print(f"  aspect_ratio = {values[0]:.4f}  circularity = {values[1]:.4f}  "
          f"solidity = {values[2]:.4f}  extent = {values[3]:.4f}")

    # ── 00: contour, hull, bounding box, ellipse ─────────────────────────────
    margin   = 24
    canvas_h = max(h + 2 * margin, 2 * margin + 1)
    canvas_w = max(w + 2 * margin, 2 * margin + 1)
    canvas   = np.full((canvas_h, canvas_w, 3), 255, np.uint8)

    shift        = np.array([margin - x, margin - y], dtype=np.float64)
    shifted      = np.round(contour + shift).astype(np.int32)
    hull_shifted = cv2.convexHull(shifted.reshape(-1, 1, 2)) if len(shifted) >= 3 else shifted.reshape(-1, 1, 2)

    cv2.drawContours(canvas, [shifted], -1, (40, 120, 255), 2)
    if len(shifted) >= 3:
        cv2.drawContours(canvas, [hull_shifted], -1, (255, 165, 0), 2)
    cv2.rectangle(canvas, (margin, margin), (margin + w, margin + h), (76, 175, 80), 2)

    # Vẽ ellipse nếu có
    if ellipse_info is not None:
        ex = int(round(ellipse_info["cx"] - x + margin))
        ey = int(round(ellipse_info["cy"] - y + margin))
        eMA = int(round(ellipse_info["MA"] / 2))
        ema = int(round(ellipse_info["ma"] / 2))
        cv2.ellipse(
            canvas,
            (ex, ey),
            (max(eMA, 1), max(ema, 1)),
            ellipse_info["angle"],
            0, 360,
            (220, 80, 220), 2,
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title('Contour + bounding box + convex hull + ellipse', fontsize=12,
                 fontweight='bold', color='white', pad=10)
    ax.axis('off')

    summary_text = (
        f"aspect_ratio = {values[0]:.4f}\n"
        f"circularity  = {values[1]:.4f}\n"
        f"solidity     = {values[2]:.4f}\n"
        f"extent       = {values[3]:.4f}\n"
        f"area         = {area:.1f}px²\n"
        f"perimeter    = {perimeter:.1f}px"
    )
    ax.text(
        0.03, 0.97, summary_text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10, color='white', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a4a',
                  edgecolor='#8ab4f8', alpha=0.92),
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(40/255, 120/255, 255/255),  label='Contour'),
        Patch(facecolor=(255/255, 165/255, 0),        label='Convex hull'),
        Patch(facecolor=(76/255, 175/255, 80/255),    label='Bounding box'),
        Patch(facecolor=(220/255, 80/255, 220/255),   label='Fit ellipse'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=8, facecolor='#2a2a4a', labelcolor='white',
              edgecolor='#555577')

    plt.tight_layout()
    plt.savefig(_p(out_dir, 'morph_00_geometry.png'), dpi=110,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print('    → morph_00_geometry.png')

    # ── 01: vector 4D ────────────────────────────────────────────────────────
    labels = ['aspect', 'circularity', 'solidity', 'extent']
    colors = ['#4c9be8', '#e88a4c', '#4ce87a', '#e84c9b']
    _save_vector(
        'Morphology vector (4D) — lưu vào DB',
        values,
        labels,
        _p(out_dir, 'morph_01_vector.png'),
        colors,
    )

    # ── 02: diagnostic text panel ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.axis('off')

    text = (
        'Feature definition\n'
        '  aspect_ratio = minor_axis / major_axis  (fitEllipse, rotation-invariant)\n'
        '  circularity  = 4π × area / perimeter²\n'
        '  solidity     = area / convex_hull_area\n'
        '  extent       = area / bounding_rect_area\n\n'
        'Why this matters\n'
        '  - aspect_ratio phản ánh lá dài/ngắn, không bị ảnh hưởng góc xoay\n'
        '  - circularity cao khi biên gần tròn, thấp khi lá nhọn/răng cưa\n'
        '  - solidity thấp nếu biên lõm hoặc có nhiều thùy sâu\n'
        '  - extent phân biệt hình elip vs tam giác vs chữ nhật\n'
    )
    ax.text(
        0.03, 0.95, text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=11, color='white', family='monospace', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.65', facecolor='#22263d',
                  edgecolor='#555577', alpha=0.95),
    )
    ax.text(
        0.97, 0.06,
        f"vector = {values.round(4).tolist()}",
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=10, color='#f0e68c', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a',
                  edgecolor='#f0e68c', alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(_p(out_dir, 'morph_02_notes.png'), dpi=110,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print('    → morph_02_notes.png')

    # ── JSON summary ─────────────────────────────────────────────────────────
    result = {
        "vector": [float(v) for v in values.tolist()],
        "labels": ["aspect_ratio", "circularity", "solidity", "extent"],
        "display_labels": ["Aspect ratio", "Circularity", "Solidity", "Extent"],
        "meanings": [
            "minor_axis / major_axis (fitEllipse, rotation-invariant)",
            "4π × area / perimeter²",
            "area / convex_hull_area",
            "area / bounding_rect_area",
        ],
        "contour_points": int(len(contour)),
        "area":      float(area),
        "perimeter": float(perimeter),
        "hull_area": float(hull_area),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "ellipse": {
            "cx":    float(ellipse_info["cx"])    if ellipse_info else None,
            "cy":    float(ellipse_info["cy"])    if ellipse_info else None,
            "MA":    float(ellipse_info["MA"])    if ellipse_info else None,
            "ma":    float(ellipse_info["ma"])    if ellipse_info else None,
            "angle": float(ellipse_info["angle"]) if ellipse_info else None,
        },
    }

    with open(_p(out_dir, 'morphology.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('  → Tổng: 3 ảnh (morph_00 ÷ morph_02) + morphology.json')

    return result