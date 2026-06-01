import cv2
import numpy as np


# # ════════════════════════════════════════════════════════════════════
# # TIỆN ÍCH NỘI BỘ
# # ════════════════════════════════════════════════════════════════════

# def _shoelace_signed_area(contour: np.ndarray) -> float:
#     x, y = contour[:, 0], contour[:, 1]
#     return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# def _remove_self_intersections(contour: np.ndarray) -> np.ndarray:
#     """
#     Vẽ contour lên ảnh tạm rồi re-extract → loại bỏ self-intersection.
#     Cần thiết để shoelace area tính đúng dấu.
#     """
#     pts = contour.astype(np.int32)
#     x0  = pts[:, 0].min() - 5
#     y0  = pts[:, 1].min() - 5
#     w   = int(pts[:, 0].max() - x0) + 6
#     h   = int(pts[:, 1].max() - y0) + 6
#     loc = pts.copy(); loc[:, 0] -= x0; loc[:, 1] -= y0
#     tmp = np.zeros((h, w), np.uint8)
#     cv2.drawContours(tmp, [loc], -1, 255, -1)
#     cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not cnts:
#         return contour
#     clean = max(cnts, key=cv2.contourArea).squeeze()
#     if clean.ndim == 1:
#         clean = clean.reshape(-1, 2)
#     clean = clean.astype(np.float64)
#     clean[:, 0] += x0
#     clean[:, 1] += y0
#     return clean


# def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
#     """
#     Chuẩn hóa contour:
#     1. Loại bỏ self-intersection (cần thiết để tính shoelace đúng)
#     2. Đảm bảo hướng CCW — trong OpenCV (y tăng xuống dưới),
#        shoelace > 0 tương ứng CW → cần đảo
#     3. Điểm bắt đầu: x nhỏ nhất (argmin đơn giản, tie-break tự nhiên theo index)
#     """
#     contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)

#     # Bước 1: loại bỏ self-intersection
#     contour = _remove_self_intersections(contour)

#     # Bước 2: đảm bảo CCW — gốc dùng > 0 để đảo (OpenCV y-down: > 0 = CW)
#     if _shoelace_signed_area(contour) > 0:
#         contour = contour[::-1].copy()

#     # Bước 3: điểm bắt đầu = x nhỏ nhất (argmin, khớp với DB đã build)
#     idx = int(np.argmin(contour[:, 0]))
#     return np.roll(contour, -idx, axis=0)


# # ════════════════════════════════════════════════════════════════════
# # TIỀN XỬ LÝ ẢNH LÁ (nền TRẮNG)
# # ════════════════════════════════════════════════════════════════════
# import cv2
# import numpy as np


# def preprocess_leaf(image_path) -> tuple:
#     """
#     Tiền xử lý ảnh lá trên nền TRẮNG.
#     Trả về: (img, contour, gray, mask, leaf_area)

#     Cải tiến:
#     - GaussianBlur nhẹ trước Otsu → giảm ảnh hưởng noise/texture bề mặt
#     - Loại contour nhỏ (bóng đổ, vết bẩn) trước khi chọn contour lá
#     - Validate tỉ lệ diện tích lá / toàn ảnh để phát hiện ảnh lỗi sớm
#     - Dùng binary mask trực tiếp (thay drawContours) để tránh hở do contour tự giao
#     """
#     img = cv2.imread(str(image_path))
#     if img is None:
#         raise ValueError(f"Không đọc được ảnh: {image_path}")

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h, w = gray.shape

#     # ── 1. Blur nhẹ trước Otsu ───────────────────────────────────────
#     #
#     #  Otsu nhạy cảm với noise: texture bề mặt lá hoặc JPEG artifact
#     #  làm histogram bị răng cưa → ngưỡng lệch.
#     #  Blur 5×5 làm mượt histogram mà không ảnh hưởng biên lá.
#     #
#     gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # ── 2. Otsu trên ảnh blur — vẫn dùng toàn ảnh vì nền trắng ─────
#     #
#     #  Với nền TRẮNG: histogram có 2 đỉnh rõ (nền sáng, lá tối)
#     #  → Otsu toàn ảnh là lựa chọn đúng, KHÔNG cần percentile.
#     #  (Khác với vein: trong mask không có nền trắng → Otsu lệch)
#     #
#     _, binary = cv2.threshold(
#         gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # ── 3. Morphology — giống cũ, đủ tốt ────────────────────────────
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     # ── 4. Lọc contour nhỏ trước khi chọn lá ────────────────────────
#     #
#     #  Phiên bản cũ lấy max(contourArea) trực tiếp → đúng trong 90% ảnh.
#     #  Nhưng nếu ảnh có: bóng đổ lớn, nhãn giấy, vết bẩn nền
#     #  → contour lá bị chia nhỏ và max() chọn sai.
#     #
#     #  Fix: loại contour có area < 1% tổng ảnh trước,
#     #  rồi mới lấy lớn nhất trong số còn lại.
#     #
#     cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not cnts:
#         raise ValueError("Không tìm thấy contour lá")

#     min_area = 0.01 * h * w   # bỏ contour < 1% diện tích ảnh
#     cnts_filtered = [c for c in cnts if cv2.contourArea(c) > min_area]
#     if not cnts_filtered:
#         cnts_filtered = cnts  # fallback nếu lọc quá chặt

#     raw     = max(cnts_filtered, key=cv2.contourArea).squeeze().reshape(-1, 2)
#     contour = _fix_contour_orientation(raw)

#     # ── 5. Mask từ binary (không dùng drawContours) ──────────────────
#     #
#     #  drawContours fill đôi khi hở ở contour tự giao.
#     #  Dùng thẳng binary đã qua morphology → đảm bảo mask kín.
#     #  Vẽ lại contour chỉ để có mask khớp chính xác với contour đã fix.
#     #
#     mask = np.zeros_like(gray)
#     cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, cv2.FILLED)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     # ── 6. Validate ──────────────────────────────────────────────────
#     leaf_area = int((mask > 0).sum())

#     if leaf_area < 500:
#         raise ValueError(f"Vùng lá quá nhỏ ({leaf_area} px)")

#     # Lá thật nên chiếm 5%–90% diện tích ảnh
#     ratio = leaf_area / (h * w)
#     if ratio < 0.05:
#         raise ValueError(f"Lá quá nhỏ so với ảnh ({ratio:.1%}) — kiểm tra lại ảnh hoặc nền")
#     if ratio > 0.90:
#         raise ValueError(f"Vùng lá chiếm {ratio:.1%} ảnh — có thể Otsu bị lật (nền tối?)")

#     return img, contour, gray, mask, leaf_area



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH NỘI BỘ
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH CONTOUR
# ════════════════════════════════════════════════════════════════════

def _shoelace_signed_area(contour: np.ndarray) -> float:
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _remove_self_intersections(contour: np.ndarray) -> np.ndarray:
    """Vẽ contour lên ảnh tạm rồi re-extract → loại bỏ self-intersection."""
    pts = contour.astype(np.int32)
    x0  = pts[:, 0].min() - 5
    y0  = pts[:, 1].min() - 5
    w   = int(pts[:, 0].max() - x0) + 6
    h   = int(pts[:, 1].max() - y0) + 6
    loc = pts.copy(); loc[:, 0] -= x0; loc[:, 1] -= y0
    tmp = np.zeros((h, w), np.uint8)
    cv2.drawContours(tmp, [loc], -1, 255, -1)
    cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return contour
    clean = max(cnts, key=cv2.contourArea).squeeze()
    if clean.ndim == 1:
        clean = clean.reshape(-1, 2)
    clean = clean.astype(np.float64)
    clean[:, 0] += x0
    clean[:, 1] += y0
    return clean


def _fix_contour_orientation(contour: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa contour:
    1. Loại bỏ self-intersection
    2. Đảm bảo hướng CCW (OpenCV y-down: shoelace > 0 = CW → cần đảo)
    3. Điểm bắt đầu: x nhỏ nhất
    """
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    contour = _remove_self_intersections(contour)
    if _shoelace_signed_area(contour) > 0:
        contour = contour[::-1].copy()
    idx = int(np.argmin(contour[:, 0]))
    return np.roll(contour, -idx, axis=0)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Lấp lỗ hổng bên trong mask bằng flood fill an toàn.
    Thêm border đen 1px → seed (0,0) luôn là nền, không bao giờ là lá.
    """
    bordered  = cv2.copyMakeBorder(mask, 1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT, value=0)
    flood     = bordered.copy()
    h, w      = flood.shape
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]   # cắt border, lấy lỗ hổng
    return cv2.bitwise_or(mask, holes)


# ════════════════════════════════════════════════════════════════════
# NGƯỠNG CỐ ĐỊNH — nền trắng chuẩn hóa
# ════════════════════════════════════════════════════════════════════

FIXED_THRESH = 210


# ════════════════════════════════════════════════════════════════════
# TIỀN XỬ LÝ ẢNH LÁ (nền TRẮNG)
# ════════════════════════════════════════════════════════════════════

def preprocess_leaf(image_path) -> tuple:
    """
    Tiền xử lý ảnh lá trên nền TRẮNG.
    Trả về: (img, contour, gray, mask, leaf_area)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── 1. Blur nhẹ — giảm JPEG artifact trước khi threshold ────────
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 2. Ngưỡng cố định — ổn định hơn Otsu với nền trắng chuẩn hóa
    _, binary = cv2.threshold(gray_blur, FIXED_THRESH, 255,
                              cv2.THRESH_BINARY_INV)

    # ── 3. Morphology ────────────────────────────────────────────────
    kernel_open  = np.ones((3, 3), np.uint8)   # không xóa cuống lá mỏng
    kernel_close = np.ones((7, 7), np.uint8)   # lấp lỗ hổng
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # ── 4. Lọc contour nhỏ (bóng đổ, vết bẩn) ──────────────────────
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("Không tìm thấy contour lá")

    min_area      = 0.01 * h * w
    cnts_filtered = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts_filtered:
        cnts_filtered = cnts   # fallback

    raw     = max(cnts_filtered, key=cv2.contourArea).squeeze().reshape(-1, 2)
    contour = _fix_contour_orientation(raw)

    # ── 5. Mask ──────────────────────────────────────────────────────
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = _fill_holes(mask)

    # ── 6. Validate ──────────────────────────────────────────────────
    leaf_area = int((mask > 0).sum())

    if leaf_area < 500:
        raise ValueError(f"Vùng lá quá nhỏ ({leaf_area} px)")

    ratio = leaf_area / (h * w)
    if ratio < 0.05:
        raise ValueError(
            f"Lá quá nhỏ so với ảnh ({ratio:.1%}) — kiểm tra lại ảnh hoặc nền")
    if ratio > 0.90:
        raise ValueError(
            f"Vùng lá chiếm {ratio:.1%} — có thể nền không đủ trắng "
            f"(thử tăng FIXED_THRESH)")

    return img, contour, gray, mask, leaf_area
