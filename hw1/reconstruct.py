import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import copy


def map_png(root, subdir):
    return {int(p.stem): str(p)
            for p in (Path(root)/subdir).glob("*.png")
            if p.stem.isdigit()}


def depth_image_to_point_cloud(rgb, depth):
    # TODO: Get point cloud from rgb and depth image 
    h, w, _ = rgb.shape
    f = w / (2 * np.tan(np.pi * 0.5 / 2))
    cx, cy = w / 2, h / 2
    
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(w, h, f, f, cx, cy)
    
    color_image = o3d.geometry.Image(rgb)
    depth_meters = depth.astype(np.float32) / 255.0 * 10.0 # load.py save like this
    depth_image = o3d.geometry.Image(depth_meters)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, 
        depth_scale=1.0,  # Scaled to m
        depth_trunc=10.0,  # Truncate at 10m
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, camera_intrinsic
    )
    
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def get_FPFH(pcd, voxel_size):
    # Compute FPFH feature
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    return fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        voxel_size,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Write your own ICP function
    MAX_ITERS = 30
    TOLERANCE = 1e-6
    
    # Convert to NumPy
    T = np.array(trans_init.transformation, dtype=np.float64)
    source = np.asarray(source_down.points, dtype=np.float64)
    target = np.asarray(target_down.points, dtype=np.float64)

    result = o3d.pipelines.registration.RegistrationResult()

    # Build KD-Tree for faster search
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    # Homogeneous source for fast transforms
    source_h = np.c_[source, np.ones(len(source))]
    prev_rmse = np.inf
    thr2 = threshold * threshold

    for i in range(MAX_ITERS):
        # 1. Transform source by current T
        source_trans = (T @ source_h.T).T[:, :3]

        # 2. Build correspondences
        source_corr = []
        target_corr = []
        for p in source_trans:
            _, idx, d2 = kdtree.search_knn_vector_3d(p, 1)
            if len(idx) == 1 and d2[0] <= thr2:
                source_corr.append(p)
                target_corr.append(target[idx[0]])

        # Not enough pairs to solve a stable rigid transform
        if len(source_corr) < 6:
            break

        A = np.asarray(source_corr)  # transformed source
        B = np.asarray(target_corr)  # matched target

        # 3. Solve rigid transform using SVD
        muA, muB = A.mean(axis=0), B.mean(axis=0)
        AA, BB = A - muA, B - muB
        U, S, Vt = np.linalg.svd(AA.T @ BB)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = muB - R @ muA

        # 4. Update transform
        T_update = np.eye(4)
        T_update[:3, :3] = R
        T_update[:3, 3] = t
        T = T_update @ T

        # 5. Compute RMSE and check convergence
        A_aligned = (A @ R.T) + t
        rmse = float(np.sqrt(np.mean(np.sum((A_aligned - B) ** 2, axis=1))))
        if abs(prev_rmse - rmse) < TOLERANCE:
            prev_rmse = rmse
            break
        prev_rmse = rmse

    result.transformation = T
    result.fitness = float((len(source_corr) / max(1, len(source))))
    result.inlier_rmse = float(prev_rmse)
    return result


def reconstruct(args):
# TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    VOXEL_SIZE = 0.1
    
    # Prepare img
    rgb_map = map_png(args.data_root, "rgb")
    depth_map = map_png(args.data_root, "depth")
    ids = sorted(rgb_map.keys() & depth_map.keys())
    rgb_sequence = [rgb_map[i] for i in ids]
    depth_sequence = [depth_map[i] for i in ids]
    
    result_pcd = o3d.geometry.PointCloud()
    pred_cam_pos = [np.eye(4)]
    accumulated_pose = np.eye(4)
    
    for i in tqdm(range(len(rgb_sequence)-1)):
        # RGB-D -> Point Clouds
        source_rgb = np.asarray(o3d.io.read_image(rgb_sequence[i+1]))
        source_depth = np.asarray(o3d.io.read_image(depth_sequence[i+1]))
        source_pcd = depth_image_to_point_cloud(source_rgb, source_depth)
        
        target_rgb = np.asarray(o3d.io.read_image(rgb_sequence[i]))
        target_depth = np.asarray(o3d.io.read_image(depth_sequence[i]))
        target_pcd = depth_image_to_point_cloud(target_rgb, target_depth)

        # Downsample and FPFH
        source_pcd = preprocess_point_cloud(source_pcd, VOXEL_SIZE)
        features_source = get_FPFH(source_pcd, VOXEL_SIZE)
        
        target_pcd = preprocess_point_cloud(target_pcd, VOXEL_SIZE)
        features_target = get_FPFH(target_pcd, VOXEL_SIZE)

        # RANSAC
        result_ransac = execute_global_registration(
            source_pcd, target_pcd, features_source,
            features_target, VOXEL_SIZE
        )

        # ICP
        if args.version == 'open3d':
            trans = local_icp_algorithm(source_pcd, target_pcd, result_ransac, VOXEL_SIZE)
        else:
            trans = my_local_icp_algorithm(source_pcd, target_pcd, result_ransac, VOXEL_SIZE)

        # Accumulate transformation in global coordinate frame
        accumulated_pose = accumulated_pose @ trans.transformation

        # Transform source point cloud to global frame and merge
        pcd_source_global = copy.deepcopy(source_pcd)
        pcd_source_global.transform(accumulated_pose) # Cam coordinate -> World coordinate
        result_pcd += pcd_source_global
        # preprocess_point_cloud(result_pcd, VOXEL_SIZE)
        
        # Store camera pose
        pred_cam_pos.append(accumulated_pose.copy())

    return result_pcd, np.array(pred_cam_pos)


def remove_ceiling(pcd, starting_height, offset=0.2):
    # Remove ceiling points from pcd, so that we can see the room
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    ceiling_threshold = starting_height - offset
    mask = points[:, 1] > ceiling_threshold

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered_pcd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct(args)
    gt_pos = np.load(os.path.join(args.data_root, 'GT_pose.npy'))
    gt_pos[:, 2] *= -1 # Reflect to align reconstruct

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    pred_positions = np.array([pose[:3, 3] for pose in pred_cam_pos])
    gt_positions = gt_pos[:, :3]

    # Align prediction to GT and Compute L2 distanc
    pred_positions_align = pred_positions + gt_positions[0] - pred_positions[0]
    L2 = np.mean(np.linalg.norm(pred_positions_align - gt_positions, axis=1))
    print("Mean L2 distance: ", L2)

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # 1. Reconstructed point cloud
    alignment_transform = np.eye(4)
    alignment_transform[:3, 3] = gt_positions[0] - pred_positions[0]
    result_pcd.transform(alignment_transform)

    # Remove ceiling points
    starting_height = gt_positions[0, 1] # Y-up
    result_pcd = remove_ceiling(result_pcd, starting_height)
    
    # 2. Red line: estimated camera pose
    est_traj_lines = o3d.geometry.LineSet()
    est_traj_lines.points = o3d.utility.Vector3dVector(pred_positions_align)
    est_traj_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pred_positions)-1)])
    est_traj_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(pred_positions)-1)])

    # 3. Black line: ground truth camera pose
    gt_traj_lines = o3d.geometry.LineSet()
    gt_traj_lines.points = o3d.utility.Vector3dVector(gt_positions)
    gt_traj_lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(pred_positions)-1)])
    gt_traj_lines.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(len(pred_positions)-1)])

    o3d.visualization.draw_geometries(
        [result_pcd, est_traj_lines, gt_traj_lines],
        lookat=[0, 0, 0],
        up=[0, -1, 0],
        front=[-1, 0, 0],
        zoom=1
    )