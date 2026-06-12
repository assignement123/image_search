import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors
from scipy.interpolate import interp1d
from src.config import HARMONICS, N_RESAMPLE, DIM_EFD


# ════════════════════════════════════════════════════════════════════
# TIỆN ÍCH NỘI BỘ
# ════════════════════════════════════════════════════════════════════

def _resample_contour(contour: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    """
    Resample contour thành n điểm, đều theo độ dài cung (arc-length).
    endpoint=False — tránh duplicate điểm đầu/cuối.
    """
    contour = np.asarray(contour, np.float64)
    if len(contour) < 4:
        return contour

    diffs   = np.diff(contour, axis=0)
    cumdist = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total   = cumdist[-1]

    if total < 1e-6:
        return contour

    t  = np.linspace(0, total, n, endpoint=False)
    fx = interp1d(cumdist, contour[:, 0])
    fy = interp1d(cumdist, contour[:, 1])

    return np.stack([fx(t), fy(t)], axis=1)


def _normalize_efd_sign(coeffs: np.ndarray) -> np.ndarray:
    """
    Fix sign ambiguity của pyefd normalization.

    pyefd chia scale bằng np.abs(A₁), không phải A₁.
    Khi A₁ < 0 sau bước rotation, kết quả là A₁=-1 và toàn bộ
    harmonics bị đổi dấu — gây cosine similarity sai cho các lá
    cùng loài nhưng rơi vào khác "nhánh" normalization.

    Fix: nếu A₁ < 0, flip dấu toàn bộ coefficient array.
    Sau fix: A₁=+1 luôn, D₁>0 monotonic, harmonics 2..N nhất quán.

    Verified: D₁ monotonically decreasing 1.0→0.09 cho aspect ratio 1:1→15:1.
    """
    if coeffs[0, 0] < 0:
        coeffs = -coeffs
    return coeffs


# ════════════════════════════════════════════════════════════════════
# EFD — HÌNH DẠNG BIÊN LÁ
# ════════════════════════════════════════════════════════════════════

def extract_efd(contour: np.ndarray) -> np.ndarray:
    """
    Elliptic Fourier Descriptors — hình dạng biên lá.
    Bất biến với translation, scale, rotation, starting point.

    Feature vector (DIM_EFD = 1 + (HARMONICS-1)*4 chiều):
      [D₁  |  h2_a h2_b h2_c h2_d  |  ...  |  h{H}_a h{H}_b h{H}_c h{H}_d]
        ↑                         ↑
      eccentricity           harmonics 2..HARMONICS

    Sau pyefd normalize=True và sign fix:
      A₁ = +1   (cố định — bỏ)
      B₁ ≈  0   (cố định — bỏ)
      C₁ ≈  0   (cố định: phase shift làm C₁=0 — bỏ)
      D₁ > 0    (THAY ĐỔI: = b/a của ellipse cơ bản)
                  circle=1.0, oval≈0.6, lanceolate≈0.4, bamboo≈0.17
    """
    if contour is None or len(contour) < 4:
        return np.zeros(DIM_EFD, np.float32)

    contour = np.asarray(contour, np.float64).reshape(-1, 2)
    contour = _resample_contour(contour, N_RESAMPLE)

    if len(contour) < 2 * HARMONICS:
        return np.zeros(DIM_EFD, np.float32)

    coeffs = elliptic_fourier_descriptors(
        contour, order=HARMONICS, normalize=True)

    # Fix sign ambiguity: đảm bảo A₁=+1 và D₁>0 nhất quán
    coeffs = _normalize_efd_sign(coeffs)

    # D₁ = coeffs[0,3]: eccentricity (b/a), phân biệt tròn vs hẹp
    # harmonics 2..HARMONICS: chi tiết shape (đầu nhọn, răng cưa, bất đối xứng)
    d1   = coeffs[0, 3:4]           # shape (1,)
    rest = coeffs[1:, :].flatten()  # shape ((HARMONICS-1)*4,)
    return np.concatenate([d1, rest]).astype(np.float32)


def extract_efd_flipped(contour: np.ndarray) -> np.ndarray:
    """
    EFD của contour lật gương (flip trục x) — reflection invariance.
    Dùng trong search: similarity = max(sim(db, q), sim(db, q_flip))
    """
    if contour is None or len(contour) < 4:
        return np.zeros(DIM_EFD, np.float32)
    flipped = np.asarray(contour, np.float64).reshape(-1, 2).copy()
    flipped[:, 0] = -flipped[:, 0]
    return extract_efd(flipped)


# ════════════════════════════════════════════════════════════════════
# MORPHOLOGY — HÌNH THÁI HỌC CỦA LÁ
# ════════════════════════════════════════════════════════════════════

def extract_morphology(contour: np.ndarray) -> np.ndarray:
    contour_i = contour.astype(np.float32)
    area      = cv2.contourArea(contour_i)
    perimeter = cv2.arcLength(contour_i, True)

    # ── Aspect ratio dùng fitEllipse thay boundingRect ──────────
    # fitEllipse fit theo trục thực của lá, không bị ảnh hưởng góc xoay
    if len(contour_i) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour_i)
        # Luôn lấy min/max để aspect_ratio ≤ 1
        aspect_ratio = float(min(MA, ma)) / max(MA, ma) if max(MA, ma) > 0 else 0.0
    else:
        x, y, w, h = cv2.boundingRect(contour_i)
        aspect_ratio = float(min(w,h)) / max(w,h) if max(w,h) > 0 else 0.0

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    hull      = cv2.convexHull(contour_i)
    hull_area = cv2.contourArea(hull)
    solidity  = float(area) / hull_area if hull_area > 0 else 0.0

    # ── Extent: độ lấp đầy bounding box ─────────────────────────
    # Phân biệt lá hình tam giác vs elip dù cùng aspect_ratio
    x, y, w, h = cv2.boundingRect(contour_i)
    extent = area / (w * h) if (w * h) > 0 else 0.0

    return np.array([
        aspect_ratio,   # [0,1]: lá tròn→1.0, lá dài→0.2
        circularity,    # [0,1]: tròn→1.0, răng cưa/nhọn→thấp
        solidity,       # [0,1]: nguyên→0.99, thùy sâu→0.6
        extent,         # [0,1]: elip→π/4≈0.78, hình thoi→0.5
    ], dtype=np.float32)
