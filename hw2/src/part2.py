import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from collections import namedtuple

# RRT Hyperparameters
MAX_ITER = 6000
STEP_SIZE = 100
GOAL_THRESHOLD = 100
SEARCH_RADIUS = 100

Node = namedtuple("Node", ["x", "y", "parent"])

# load map
map_img = cv2.imread("../results/map.png")
h, w, _ = map_img.shape
gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

# occupancy map
_, occ_map = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY) # non-white pixels are obstacles
occ_map = 255 - occ_map
occ_map = occ_map > 0

# Parse color map from CSV
df = pd.read_csv("../color_coding_semantic_segmentation_classes.csv", dtype=str)
color_dict = {}
for _, row in df.iterrows():
    name = str(row.get("Name", "")).strip().lower()
    rgb_str = row.get("Color_Code (R,G,B)", "")
    if not name or not rgb_str:
        continue
    rgb = list(int(c) for c in rgb_str.strip("()").split(","))
    bgr = [rgb[2], rgb[1], rgb[0]]  # Convert RGB to BGR
    color_dict[name] = bgr

# target color region
def find_target_region(img, target_name):
    if target_name not in color_dict:
        raise ValueError(f"Target '{target_name}' does not exist in the color map.")
    target_color = np.array(color_dict[target_name])
    mask = np.all(img == target_color, axis=-1)
    coords = np.column_stack(np.where(mask))
    return coords

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(nodes, point):
    dists = [distance((n.x, n.y), point) for n in nodes]
    return nodes[np.argmin(dists)]

def collision_free(a, b, occ_map):
    x1, y1 = int(a[0]), int(a[1])
    x2, y2 = int(b[0]), int(b[1])
    line = np.linspace((x1, y1), (x2, y2), num=50).astype(int)
    for x, y in line:
        if x < 0 or y < 0 or x >= occ_map.shape[1] or y >= occ_map.shape[0]:
            return False
        if occ_map[y, x]:
            return False
    return True

def RRT(start, goal, occ_map):
    nodes = [Node(start[0], start[1], -1)]
    edges = []
    for i in range(MAX_ITER):
        sample = goal if random.random() < 0.1 else (random.randint(0, w-1), random.randint(0, h-1))
        nearest_node = nearest(nodes, sample)
        theta = math.atan2(sample[1]-nearest_node.y, sample[0]-nearest_node.x)
        new_x = int(nearest_node.x + STEP_SIZE * math.cos(theta))
        new_y = int(nearest_node.y + STEP_SIZE * math.sin(theta))
        if new_x < 0 or new_y < 0 or new_x >= w or new_y >= h:
            continue
        if not collision_free((nearest_node.x, nearest_node.y), (new_x, new_y), occ_map):
            continue

        new_node = Node(new_x, new_y, nodes.index(nearest_node))
        nodes.append(new_node)
        edges.append(((nearest_node.x, nearest_node.y), (new_x, new_y)))

        if distance((new_x, new_y), goal) < GOAL_THRESHOLD:
            print(f"Goal reached in {i} iterations!")
            path = [(new_x, new_y)]
            parent = new_node.parent
            while parent != -1:
                n = nodes[parent]
                path.append((n.x, n.y))
                parent = n.parent
            path.reverse()
            return path, edges
    print("Failed to reach goal.")
    return None, edges


# use mouse to select start point
def get_start_point(img, goal):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.plot(goal[0], goal[1], 'go', markersize=5)  # Draw goal point
    plt.title("map.png (please click to select start point)")
    pts = plt.ginput(1, timeout=0)
    plt.close()
    if len(pts) == 0:
        raise ValueError("No point selected.")
    x, y = map(int, pts[0])
    print(f"Selected point: ({x}, {y})")
    return (x, y)


def part2():
    target = input("Enter target class (e.g., sofa, cooktop): ").strip().lower()
    coords = find_target_region(map_img, target)
    if coords.size == 0:
        raise ValueError(f"No region found for {target}.")

    gy, gx = np.mean(coords, axis=0).astype(int)
    goal = (gx, gy)

    start = get_start_point(map_img, goal)
    path, edges = RRT(start, goal, occ_map)

    out = map_img.copy()

    for (p1, p2) in edges:
        cv2.line(out, p1, p2, (255, 0, 0), 1)

    if path:
        cv2.circle(out, start, 7, (0, 0, 255), -1)
        cv2.circle(out, goal, 7, (0, 255, 0), -1)
        for i in range(1, len(path)):
            cv2.line(out, path[i-1], path[i], (255, 0, 0), 3)
        cv2.imwrite(f"path_{target}.png", out)

        display_img = cv2.resize(out, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        cv2.imshow("RRT Path", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Path saved as path_{target}.png")
    else:
        print("No valid path found.")

