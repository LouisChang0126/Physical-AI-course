import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import shutil
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from collections import namedtuple
import json

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
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, z, c=colors / 255.0, s=0.1)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/map.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    print("saved as results/map.png")
    return meta

def part2(meta):
    # RRT Hyperparameters
    MAX_ITER = 6000
    STEP_SIZE = 0.02
    GOAL_THRESHOLD = 0.03
    SAFETY_RADIUS = 0.002

    # load map
    map_img = cv2.imread("../results/map.png")
    if map_img is None:
        raise FileNotFoundError("../results/map.png not found")
    h, w, _ = map_img.shape
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    # occupancy map
    _, occ_map = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    occ_map = 255 - occ_map
    occ_map = occ_map > 0
    # occupancy map inflation
    SAFETY_RADIUS_pixels = SAFETY_RADIUS * w / (meta['x_max'] - meta['x_min'])
    k = 2 * math.ceil(SAFETY_RADIUS_pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    occ_map = cv2.dilate(occ_map.astype(np.uint8), kernel).astype(bool)

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

    # Parse color map from xlsx
    df = pd.read_excel("../color_coding_semantic_segmentation_classes.xlsx", dtype=str)
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
        target_id = int(float(df.loc[df["Name"].eq(target_name)].iloc[0, 0]))
        return target_id, coords

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest(nodes, point):
        dists = [distance((n.x, n.z), point) for n in nodes]
        return nodes[np.argmin(dists)]

    def collision_free(a, b):
        ua, va = world_to_pixel(a[0], a[1])
        ub, vb = world_to_pixel(b[0], b[1])
        pix_dist = math.hypot(ub - ua, vb - va)
        xs_world = np.linspace(a[0], b[0], max(20, int(pix_dist)))
        zs_world = np.linspace(a[1], b[1], max(20, int(pix_dist)))
        for xw, zw in zip(xs_world, zs_world):
            u, v = world_to_pixel(xw, zw)
            if u < 0 or v < 0 or u >= w or v >= h:
                return False
            if occ_map[v, u]:
                return False
        return True

    # RRT in world coordinates
    def RRT(start, goal):
        Node = namedtuple("Node", ["x", "z", "parent"])
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

    def RRT_star(start, goal):
        Node = namedtuple("Node", ["x", "z", "parent", "cost"])
        nodes = [Node(float(start[0]), float(start[1]), -1, 0.0)]

        world_diam = math.hypot(meta['x_max'] - meta['x_min'], meta['z_max'] - meta['z_min'])
        GAMMA = 0.8 * world_diam

        def steer(from_xy, to_xy, step=STEP_SIZE):
            dx, dz = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
            dist = math.hypot(dx, dz)
            if dist <= 1e-12:
                return from_xy
            scale = min(1.0, step / dist)
            return (from_xy[0] + dx * scale, from_xy[1] + dz * scale)

        def nearest_index(pt):
            best_i, best_d2 = 0, float("inf")
            for i, n in enumerate(nodes):
                d2 = (n.x - pt[0])**2 + (n.z - pt[1])**2
                if d2 < best_d2:
                    best_i, best_d2 = i, d2
            return best_i

        def neighbor_indices(center_xy, n_now):
            r = max(1.5 * STEP_SIZE, GAMMA * math.sqrt(max(1e-9, math.log(n_now + 1) / (n_now + 1))))
            r2 = r * r
            cand = []
            for i, n in enumerate(nodes):
                if (n.x - center_xy[0])**2 + (n.z - center_xy[1])**2 <= r2:
                    cand.append(i)
            # sort by distance then cap
            cand.sort(key=lambda i: (nodes[i].x - center_xy[0])**2 + (nodes[i].z - center_xy[1])**2)
            return cand[:30]

        def backtrack(last_idx):
            path = []
            p = last_idx
            while p != -1:
                n = nodes[p]
                path.append((n.x, n.z))
                p = n.parent
            path.reverse()
            return path

        goal_idx = None
        edges = []
        for it in range(MAX_ITER):
            if random.random() < 0.12:
                sample = goal
            else:
                sample = (random.uniform(meta['x_min'], meta['x_max']), random.uniform(meta['z_min'], meta['z_max']))

            # Nearest + steer
            ni = nearest_index(sample)
            xn = (nodes[ni].x, nodes[ni].z)
            x_new = steer(xn, sample, STEP_SIZE)

            if not collision_free(xn, x_new):
                continue

            # parent among neighbors
            neigh = neighbor_indices(x_new, len(nodes))
            best_parent = ni
            best_cost = nodes[ni].cost + distance(xn, x_new)

            for j in neigh:
                pj = (nodes[j].x, nodes[j].z)
                if not collision_free(pj, x_new):
                    continue
                cand_cost = nodes[j].cost + distance(pj, x_new)
                if cand_cost + 1e-12 < best_cost:
                    best_parent = j
                    best_cost = cand_cost

            # add node
            nodes.append(Node(x_new[0], x_new[1], best_parent, best_cost))
            new_idx = len(nodes) - 1

            # rewire neighbors if improves cost
            for j in neigh:
                if j == best_parent or j == new_idx:
                    continue
                pj = (nodes[j].x, nodes[j].z)
                alt_cost = nodes[new_idx].cost + distance(x_new, pj)
                if alt_cost + 1e-12 < nodes[j].cost and collision_free(x_new, pj):
                    nodes[j] = Node(nodes[j].x, nodes[j].z, new_idx, alt_cost)

            if distance(x_new, goal) < GOAL_THRESHOLD:
                goal_idx = new_idx
                print(f"Goal reached (RRT*) in {it} iterations!")
                path = backtrack(goal_idx)
                for i in range(1, len(nodes)):
                    p = nodes[i].parent
                    if p >= 0:
                        edges.append(((nodes[p].x, nodes[p].z), (nodes[i].x, nodes[i].z)))
                return path, edges

        print("Failed to reach goal (RRT*).")
        for i in range(1, len(nodes)):
            p = nodes[i].parent
            if p >= 0:
                edges.append(((nodes[p].x, nodes[p].z), (nodes[i].x, nodes[i].z)))
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

    def simplify_path(path):
        if not path or len(path) < 3:
            return path
        simplified = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = i + 1
            while j + 1 < n and collision_free(path[i], path[j + 1]):
                j += 1
            simplified.append(path[j])
            i = j
        return simplified
    
    target = input("Enter target class (rack, cushion, sofa, stair, and cooktop): ").strip().lower()
    target_id, coords = find_target_region(map_img, target)
    if coords.size == 0:
        raise ValueError(f"No region found for {target}.")

    v_mean, u_mean = np.mean(coords, axis=0).astype(int)
    goal_world = pixel_to_world(u_mean, v_mean)
    start_world = get_start_point(map_img, (u_mean, v_mean))

    path, edges = RRT_star(start_world, goal_world)
    simplify = simplify_path(path)
    
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
            cv2.line(out, world_to_pixel(*path[i-1]), world_to_pixel(*path[i]), (255, 0, 0), 5)
        for i in range(1, len(simplify)):
            cv2.line(out, world_to_pixel(*simplify[i-1]), world_to_pixel(*simplify[i]), (0, 0, 255), 3)
        cv2.imwrite(f"../results/path_{target}.png", out)
    display = cv2.resize(out, None, fx=0.2, fy=0.2)
    cv2.imshow("RRT", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if path:
        print(f"Path: {path}")
    else:
        print("No path found.")
    return target_id, simplify

def part3(target_id, path):
    # This is the scene we are going to load.
    test_scene = "../../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply"
    test_scene_info_semantic = "../../hw0/replica_v1/apartment_0/habitat/info_semantic.json"

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    }

    # This function generates a config for the simulator.
    # It contains two parts:
    # one for the simulator backend
    # one for the agent, where you can attach a bunch of sensors


    def transform_rgb_bgr(image):
        return image[:, :, [2, 1, 0]]
            
    def semantic_label_to_id(semantic_sensor_label):
        with open(test_scene_info_semantic, "r") as f:
            annotations = json.load(f)
            id_to_label = np.where(np.array(annotations["id_to_label"]) < 0, 0, annotations["id_to_label"])
            id_mask = id_to_label[semantic_sensor_label]
            return id_mask
        
    def make_simple_cfg(settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]
        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        rgb_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        #semantic snesor
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
        agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.2)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=5)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=5)
            ),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def navigateAndSee(count, action="", data_root='data_collection/second_floor/'):
        observations = sim.step(action)
        
        rgb_img = transform_rgb_bgr(observations["color_sensor"])
        # add overlay for target object
        semantic_id_mask = semantic_label_to_id(observations["semantic_sensor"])
        target_id_region = semantic_id_mask == target_id
        overlay = rgb_img.copy()
        overlay[target_id_region] = (0, 0, 255)
        rgb_img = cv2.addWeighted(rgb_img, 0.5, overlay, 0.5, 0.0)

        cv2.imshow("RGB", rgb_img)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        # print("Frame:", count)
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        
        cv2.imwrite(data_root + f"rgb/{count}.png", rgb_img)
    
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    
    
    agent_state.position = np.array([path[0][0], 0.0, path[0][1]])  # agent in world space
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)

    data_root = "data_collection/first_floor/"

    if os.path.isdir(data_root): 
        shutil.rmtree(data_root)  # WARNING: this line will delete whole directory with files

    for sub_dir in ['rgb/']:
        os.makedirs(data_root + sub_dir)

    count = 0
    action = "move_forward"

    navigateAndSee(count, action, data_root)
    
    def quat_to_yaw(w, x, y, z):
        return math.atan2(2.0 * (w * y + x * z), 1 - 2.0 * (y**2 + z**2))

    for pos_x, pos_z in path[1:]:
        while True:
            agent_state = agent.get_state()
            sensor_state = agent_state.sensor_states['color_sensor']
            x = sensor_state.position[0]
            z = sensor_state.position[2]
            rw = sensor_state.rotation.w
            ry = sensor_state.rotation.y
            yaw = quat_to_yaw(rw, 0, ry, 0)
            
            target_yaw = -math.atan2(pos_x - x, -pos_z + z)
            angle_diff = np.degrees(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))) # normalize to [-180, 180]
            # print(f"Angle diff: {angle_diff:.2f}")
            if (pos_x - x)**2 + (pos_z - z)**2 < 0.01:
                break
            elif abs(angle_diff) > 5:
                if angle_diff > 0:
                    action = "turn_left"
                    navigateAndSee(count, action, data_root)
                    print("action: LEFT")
                else:
                    action = "turn_right"
                    navigateAndSee(count, action, data_root)
                    print("action: RIGHT")
            else:
                action = "move_forward"
                navigateAndSee(count, action, data_root)
                print("action: FORWARD")
            count += 1
            delay = 15
            cv2.waitKey(delay)
    print("Navigation completed")
    
if __name__ == "__main__":
    meta = part1()
    target_id, path = part2(meta)
    path = [[x * 10000./255 for x in row] for row in path]
    part3(target_id, path)
