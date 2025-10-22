import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import sys
import argparse
import shutil


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
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
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def add_mask_to_rgb(rgb_img, mask_path, alpha=0.4):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"can't find mask: {mask_path}")
    mask = cv2.resize(mask, (rgb_img.shape[1], rgb_img.shape[0]))
    mask_color = np.zeros_like(rgb_img)
    mask_color[:, :, 0] = 255
    blended = cv2.addWeighted(rgb_img, 1, mask_color, alpha, 0, mask=mask)
    return blended


def auto_navigate(sim, agent, path_coords, target_mask, output_video):
    frames = []
    for i, (x, z) in enumerate(path_coords):
        agent_state = agent.get_state()
        agent_state.position = np.array([x, sim_settings["sensor_height"], z])
        agent.set_state(agent_state)

        observations = sim.get_sensor_observations()
        rgb = transform_rgb_bgr(observations["color_sensor"])
        rgb = add_mask_to_rgb(rgb, target_mask, alpha=0.4)
        frames.append(rgb)
        print(f"Step {i+1}/{len(path_coords)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_video, fourcc, 10, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved navigation video: {output_video}")


def part3(obj_name, path):
    sim = habitat_sim.Simulator(make_simple_cfg(sim_settings))
    agent = sim.initialize_agent(sim_settings["default_agent"])

    mask_path = f"../results/mask_{obj_name}.png"
    video_path = f"../results/{obj_name}.mp4"

    auto_navigate(sim, agent, path, mask_path, video_path)
    sim.close()