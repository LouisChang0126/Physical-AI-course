import numpy as np
import matplotlib.pyplot as plt

def part1():
    points = np.load('../semantic_3d_pointcloud/point.npy')
    colors = np.load('../semantic_3d_pointcloud/color0255.npy')

    # remove ceiling & floor
    y_min, y_max = np.percentile(points[:, 1], [25, 60])
    mask = (points[:, 1] > y_min) & (points[:, 1] < y_max)
    points = points[mask]
    colors = colors[mask]

    # plot x-z plane
    x = points[:, 0]
    z = points[:, 2]
    print(x.max(), x.min(), z.max(), z.min())
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, z, c=colors / 255.0, s=0.1)

    # === draw origin marker ===
    plt.scatter(0.15872648367586087, 0.25272562492515155, c='red', s=80, marker='x', linewidths=2, label='(0,0)')
    plt.legend(loc='upper right', fontsize=8)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/map.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print("saved as results/map.png")

if __name__ == "__main__":
    part1()