import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def save_img(title: str, img: np.ndarray, path: str, cmap: str = 'gray') -> None:
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

def save_hist(title: str, data_list: list, labels: list, colors: list,
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