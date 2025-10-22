import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import random
import math
from collections import namedtuple

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
    meta = {
        "x_min": x.min(),
        "x_max": x.max(),
        "z_min": z.min(),
        "z_max": z.max()
    }
    print(x.max(), x.min(), z.max(), z.min())
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, z, c=colors / 255.0, s=0.1)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/map.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print("saved as results/map.png")
    return meta

def part2(meta):
    # RRT Hyperparameters
    MAX_ITER = 6000
    STEP_SIZE = 0.02
    GOAL_THRESHOLD = 0.02

    Node = namedtuple("Node", ["x", "z", "parent"])

    # load map
    map_img = cv2.imread("../results/map.png")
    if map_img is None:
        raise FileNotFoundError("../results/map.png not found")
    h, w, _ = map_img.shape
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    # occupancy map (full image)
    _, occ_map = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    occ_map = 255 - occ_map
    occ_map = occ_map > 0

    non_white_mask = np.any(map_img != 255, axis=2)  # True where pixel is not white
    ys, xs = np.where(non_white_mask)
    v_min, v_max = int(ys.min()), int(ys.max())  # rows
    u_min, u_max = int(xs.min()), int(xs.max())  # cols

    # transfer coordinates between pixel & world
    def pixel_to_world(u, v):
        if (u_max - u_min) == 0 or (v_max - v_min) == 0:
            raise ValueError("degenerate non-white bbox")
        x = meta['x_min'] + ( (u - u_min) / (u_max - u_min) ) * (meta['x_max'] - meta['x_min'])
        z = meta['z_min'] + ( (v - v_min) / (v_max - v_min) ) * (meta['z_max'] - meta['z_min'])
        return float(x), float(z)

    def world_to_pixel(x, z):
        if (meta['x_max'] - meta['x_min']) == 0 or (meta['z_max'] - meta['z_min']) == 0:
            raise ValueError("degenerate world bbox")
        u = u_min + ( (x - meta['x_min']) / (meta['x_max'] - meta['x_min']) ) * (u_max - u_min)
        v = v_min + ( (z - meta['z_min']) / (meta['z_max'] - meta['z_min']) ) * (v_max - v_min)
        return int(round(u)), int(round(v))

    # Parse color map from CSV
    df = pd.read_csv("../color_coding_semantic_segmentation_classes.csv", dtype=str)
    color_dict = {}
    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip().lower()
        rgb_str = row.get("Color_Code (R,G,B)", "")
        if not name or not rgb_str:
            continue
        rgb = list(int(c) for c in rgb_str.strip("()").split(","))
        bgr = [rgb[2], rgb[1], rgb[0]]
        color_dict[name] = bgr
        
    # target color region
    def find_target_region(img, target_name):
        if target_name not in color_dict:
            raise ValueError(f"Target '{target_name}' does not exist.")
        target_color = np.array(color_dict[target_name])
        mask = np.all(img == target_color, axis=-1)
        coords = np.column_stack(np.where(mask))  # rows, cols -> v,u
        return coords

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest(nodes, point):
        dists = [distance((n.x, n.z), point) for n in nodes]
        return nodes[np.argmin(dists)]

    def collision_free(a, b):
        xs_world = np.linspace(a[0], b[0], 50)
        zs_world = np.linspace(a[1], b[1], 50)
        for xw, zw in zip(xs_world, zs_world):
            u, v = world_to_pixel(xw, zw)
            if u < 0 or v < 0 or u >= w or v >= h:
                return False
            if occ_map[v, u]:
                return False
        return True

    # RRT in world coordinates
    def RRT(start, goal):
        nodes = [Node(start[0], start[1], -1)]
        edges = []
        for i in range(MAX_ITER):
            if random.random() < 0.1:
                sample = goal
            else:
                x_rand = random.uniform(meta['x_min'], meta['x_max'])
                z_rand = random.uniform(meta['z_min'], meta['z_max'])
                sample = (x_rand, z_rand)

            nearest_node = nearest(nodes, sample)
            theta = math.atan2(sample[1] - nearest_node.z, sample[0] - nearest_node.x)
            new_x = nearest_node.x + STEP_SIZE * math.cos(theta)
            new_z = nearest_node.z + STEP_SIZE * math.sin(theta)

            if not collision_free((nearest_node.x, nearest_node.z), (new_x, new_z)):
                continue

            new_node = Node(new_x, new_z, nodes.index(nearest_node))
            nodes.append(new_node)
            edges.append(((nearest_node.x, nearest_node.z), (new_x, new_z)))

            if distance((new_x, new_z), goal) < GOAL_THRESHOLD:
                path = [(new_x, new_z)]
                parent = new_node.parent
                while parent != -1:
                    n = nodes[parent]
                    path.append((n.x, n.z))
                    parent = n.parent
                path.reverse()
                print(f"Goal reached in {i} iterations!")
                return path, edges
        print("Failed to reach goal.")
        return None, edges

    def get_start_point(img, goal_pixel):
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.plot(goal_pixel[0], goal_pixel[1], 'go', markersize=5)
        plt.title("Select start point (click anywhere on image)")
        pts = plt.ginput(1, timeout=0)
        plt.close()
        if not pts:
            raise ValueError("No point selected.")
        u, v = map(int, pts[0])
        return pixel_to_world(u, v)

    target = input("Enter target class (rack, cushion, sofa, stair, and cooktop): ").strip().lower()
    coords = find_target_region(map_img, target)
    if coords.size == 0:
        raise ValueError(f"No region found for {target}.")

    v_mean, u_mean = np.mean(coords, axis=0).astype(int)
    goal_world = pixel_to_world(u_mean, v_mean)
    print(f"Goal (world): {goal_world}")

    start_world = get_start_point(map_img, (u_mean, v_mean))
    print(f"Start (world): {start_world}")

    path, edges = RRT(start_world, goal_world)

    # draw edges & path on output image by converting world to pixel
    out = map_img.copy()
    for (a, b) in edges:
        p1 = world_to_pixel(a[0], a[1])
        p2 = world_to_pixel(b[0], b[1])
        cv2.line(out, p1, p2, (255, 0, 0), 1)

    if path:
        cv2.circle(out, world_to_pixel(*start_world), 7, (0, 0, 255), -1)
        cv2.circle(out, world_to_pixel(*goal_world), 7, (0, 255, 0), -1)
        for i in range(1, len(path)):
            cv2.line(out, world_to_pixel(*path[i-1]), world_to_pixel(*path[i]), (255, 0, 0), 3)
        cv2.imwrite(f"path_{target}.png", out)
    display = cv2.resize(out, None, fx=0.3, fy=0.3)
    cv2.imshow("RRT", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if path:
        print(f"Path: {path}")
    else:
        print("No path found.")
    return target, path



if __name__ == "__main__":
    meta = part1()
    target, path = part2(meta)
