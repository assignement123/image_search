import os
import matplotlib.pyplot as plt
import cv2
def _p(out_dir, filename):
    return os.path.join(out_dir, filename)

def debug_color_moments(img, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Color Moments")
    plt.axis("off")
    plt.savefig(_p(out_dir, "step2c_01_color.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("✓ Debug Color hoàn thành")