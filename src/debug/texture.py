import os
import matplotlib.pyplot as plt

def _p(out_dir, filename):
    return os.path.join(out_dir, filename)

def debug_texture(gray, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.imshow(gray, cmap="gray")
    plt.title("Texture - LBP + GLCM")
    plt.axis("off")
    plt.savefig(_p(out_dir, "step2b_01_texture.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("✓ Debug Texture hoàn thành")