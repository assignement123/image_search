import numpy as np

# --- CẤU HÌNH EFD (Hình dáng) ---
HARMONICS  = 15                      # harmonics 1..15: đủ để capture răng cưa vừa (Prunus, Betula)
N_RESAMPLE = 600                     # 600 >> 2×15 = Nyquist thỏa mãn tốt
DIM_EFD    = 1 + (HARMONICS - 1) * 4  # 57 chiều: D₁ (eccentricity) + harmonics 2..15

# --- CẤU HÌNH TEXTURE (LBP & GLCM) ---
LBP_P = 24
LBP_R = 3
DIM_LBP = LBP_P + 2            # 26 chiều

GLCM_LEVELS = 64
GLCM_DIST = [1, 3, 5, 7]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
DIM_GLCM = 20                  # 20 chiều

# --- KÍCH THƯỚC CHIỀU CÒN LẠI ---
DIM_MORPHOLOGY = 4
DIM_COLOR = 9
DIM_VEIN = 17   # [density] + [16-bin angle histogram, rotation-invariant]