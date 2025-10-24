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

def part3(target, path):
    # This is the scene we are going to load.
    test_scene = "../../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply"

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

    def transform_depth(image):
        depth_img = (image / 10 * 255).astype(np.uint8)
        return depth_img

    def transform_semantic(semantic_obs):
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
        return semantic_img

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

        agent_cfg.sensor_specifications = [rgb_sensor_spec]
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
        #print("action: ", action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("Frame:", count)
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        
        cv2.imwrite(data_root + f"rgb/{count}.png", transform_rgb_bgr(observations["color_sensor"]))
    
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
            print(f"Angle diff: {angle_diff:.2f}")
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
    target = "sofa"
    path = [[-1.5151224595669213, 1.79722167221834], [-0.7430423362339239, 1.9352082771711916], [-0.6352199033984838, 2.7120752818571665], [0.0723455282925518, 3.050452556313045], [0.7278641765052094, 2.6198212501615292], [1.5000997061409256, 2.756935460805633], [1.581722813476753, 3.536990387295528], [1.3823660864495488, 4.295544878008826], [1.1830093594223445, 5.054099368722125], [0.9836526323951402, 5.812653859435423], [1.1081113994008882, 6.5870297599564305], [0.8744011054487613, 7.335713617464472], [1.5651284083055603, 7.707256095628788], [0.830519521126775, 7.982024730793461]]

    part3(target, path)