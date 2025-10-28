import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import sys
import argparse
import shutil
import math
import json
import glob

def part3(target_id, target, path):
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

    def frames_to_gif(frames_dir, out_gif_path, fps=10, hold_last_ms=500):
        files = glob.glob(os.path.join(frames_dir, "*.png"))
        if not files:
            raise FileNotFoundError(f"No frames found in: {frames_dir}")

        def sort_key(p):
            base = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(base)
            except ValueError:
                return base

        files = sorted(files, key=sort_key)

        imgs = []
        first_size = None
        for f in files:
            im = Image.open(f).convert("RGB")
            if first_size is None:
                first_size = im.size
            elif im.size != first_size:
                im = im.resize(first_size, Image.BILINEAR)
            imgs.append(im)

        frame_duration = max(1, int(round(1000 / fps)))
        durations = [frame_duration] * len(imgs)
        if hold_last_ms and len(durations) > 0:
            durations[-1] += int(hold_last_ms)

        imgs[0].save(
            out_gif_path,
            save_all=True,
            append_images=imgs[1:],
            duration=durations,
            loop=0,
            disposal=2,
            optimize=True,
        )
        print(f"Saved GIF: {out_gif_path} ({len(imgs)} frames)")
    
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
    frames_to_gif(data_root + "rgb/", f"../results/{target}.gif")

if __name__ == "__main__":
    target_id = 76
    target = "sofa"
    path = [[-1.909713557287445, 2.683379751841304], [-1.5534433589064658, 4.1961892323312036], [0.4246563671975674, 6.719354677674333], [0.8661191852905288, 7.585695948856784], [0.9131905104541479, 7.851233712152422]]
    part3(target_id, target, path)