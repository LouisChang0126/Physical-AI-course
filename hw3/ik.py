import argparse, time, os, json
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R
from scipy.linalg import pinv

# for simulator
import pybullet as p
import pybullet_data

# for geometry information
from hw3_utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix, pose_7d_to_6d, pose_6d_to_7d

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_ur5_DH_params

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 40
IK_ERROR_THRESH = 0.02

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

# this is the pybullet version
def pybullet_ik(robot_id, new_pose : list or tuple or np.ndarray, 
                max_iters : int=1000, stop_thresh : float=.001, base_pos=None):
    
    new_pos, new_rot = new_pose[:3], new_pose[3:]

    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=10,
        targetPosition=new_pos,
        targetOrientation=new_rot,
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
        restPoses=np.float32(np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi).tolist(),
        maxNumIterations=max_iters,
        residualThreshold=stop_thresh)
    joint_poses = np.array(joint_poses)

    
    return joint_poses


def your_ik(robot_id, new_pose : list or tuple or np.ndarray, 
                base_pos, max_iters : int=1000, stop_thresh : float=.001):



    joint_limits = np.asarray([
            [-3*np.pi/2, -np.pi/2], # joint1
            [-2.3562, -1],           # joint2
            [-17, 17],              # joint3
            [-17, 17],              # joint4
            [-17, 17],              # joint5
            [-17, 17],              # joint6
        ])

    # get current joint angles and gripper pos, (gripper pos is fixed)
    num_q = p.getNumJoints(robot_id)
    q_states = p.getJointStates(robot_id, range(0, num_q))
    
    tmp_q = np.asarray([x[0] for x in q_states][2:8]) # current joint angles 6d (You only need to modify this)
        
    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
    # ---          Try to implement `your_ik` function WITHOUT using ANY pybullet  --- #
    # ---          API. (40% for accuracy)                                         --- #
    # --- Note : please modify the code in `your_ik` function.                     --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####

    # TODO: update tmp_q
    # tmp_q = ? # may be more than one line

    # hint : 
    # 1. You may use `your_fk` function and jacobian matrix to do this
    # 2. Be careful when computing the delta x
    # 3. You may use some hyper parameters (i.e., step rate) in optimization loops

    ###################
    # ----- hyper-parameters -----
    step = 0.35           # step size for dq update (0~1)
    lam = 1e-3            # damping for DLS (avoid singularities)
    w_pos = 1.0           # weight for position error
    w_rot = 0.5           # weight for orientation error

    # desired pose
    des_pos = np.array(new_pose[:3], dtype=np.float64)
    des_quat = np.array(new_pose[3:], dtype=np.float64)  # (x, y, z, w)

    dh = get_ur5_DH_params()  # use the HW-provided DH set (fk.py)

    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def clamp_to_limits(qv):
        out = qv.copy()
        for j in range(6):
            lo, hi = joint_limits[j]
            out[j] = np.clip(out[j], lo, hi)
        # also wrap to (-pi, pi] to keep it numerically stable
        return wrap_pi(out)

    def pose7_from_fk(qv):
        pose7, J = your_fk(dh, qv, base_pos)  # expects (pose7, J6x6)
        pos = pose7[:3]
        quat = pose7[3:]
        return pos, quat, J

    def numeric_jacobian(qv, eps=1e-4):
        """
        Central-difference numerical Jacobian on [x,y,z,rotvec(3)].
        """
        base_pos_cur, quat_cur, _ = pose7_from_fk(qv)
        rotvec_cur = R.from_quat(quat_cur).as_rotvec()
        f0 = np.hstack([base_pos_cur, rotvec_cur])

        Jnum = np.zeros((6, 6), dtype=np.float64)
        for j in range(6):
            dq = np.zeros(6); dq[j] = eps
            qp = qv + dq; qm = qv - dq
            pos_p, quat_p, _ = pose7_from_fk(qp)
            pos_m, quat_m, _ = pose7_from_fk(qm)
            rv_p = R.from_quat(quat_p).as_rotvec()
            rv_m = R.from_quat(quat_m).as_rotvec()
            fp = np.hstack([pos_p, rv_p]); fm = np.hstack([pos_m, rv_m])
            Jnum[:, j] = (fp - fm) / (2.0 * eps)
        return Jnum

    for _ in range(max_iters):
        cur_pos, cur_quat, J = pose7_from_fk(tmp_q)

        # 6D task-space error: [position; orientation_rotvec]
        e_pos = des_pos - cur_pos
        R_des = R.from_quat(des_quat)
        R_cur = R.from_quat(cur_quat)
        R_err = R_des * R_cur.inv()
        e_rot = R_err.as_rotvec()  # angle-axis (axis*angle)

        e = np.hstack([w_pos * e_pos, w_rot * e_rot])

        if np.linalg.norm(e) < stop_thresh:
            break

        # Use FK's Jacobian if valid; else fall back to numeric Jacobian
        use_J = (J is not None) and np.all(np.isfinite(J)) and J.shape == (6, 6)
        if not use_J:
            J = numeric_jacobian(tmp_q)

        # Damped Least Squares (DLS) pseudo-inverse: J^T (J J^T + λ^2 I)^{-1}
        JT = J.T
        JJt = J @ JT
        dq = JT @ np.linalg.solve(JJt + (lam**2) * np.eye(6), e)

        # integrate, clamp, and wrap
        tmp_q = tmp_q + step * dq
        tmp_q = clamp_to_limits(tmp_q)
    
    return list(tmp_q) # 6 DoF


# TODO: [for your information]
# This function is the scoring function, we will use the same code 
# to score your algorithm using all the testcases
def score_ik(robot, testcase_files : str, visualize : bool=False):

    testcase_file_num = len(testcase_files)
    ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    ik_error_cnt = [0 for _ in range(testcase_file_num)]


    p.addUserDebugText(text = "Scoring Your Inverse Kinematic Algorithm ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)

    print("============================ Task 2 : Inverse Kinematic ============================\n")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        ik_dict = json.load(f_in)
        f_in.close()
        
        test_case_name = os.path.split(testcase_file)[-1]

        poses = ik_dict['next_poses']
        cases_num = len(ik_dict['current_joint_poses'])
        
        penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)
        ik_errors = []

        for i in range(cases_num):

            # TODO: check your default arguments of `max_iters` and `stop_thresh` are your best parameters.
            #       We will only pass default arguments of your `max_iters` and `stop_thresh`.
            your_joint_poses = your_ik(robot.robot_id, poses[i], base_pos=robot._base_position) 
            

            # You can use `pybullet_ik` to see the correct version 
            # your_joint_poses = pybullet_ik(robot.robot_id, poses[i]) 

            gt_pose = poses[i]        

            p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                        jointIndices=robot._joint_name_to_ids.values(),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=your_joint_poses,
                                        positionGains=[0.2] * len(your_joint_poses),
                                        velocityGains=[1] * len(your_joint_poses),
                                        physicsClientId=robot._physics_client_id)
            
            # warmup for 0.1 sec
            for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                p.stepSimulation()
                time.sleep(SIM_TIMESTEP)


            your_pose = robot.get_eef_pose()

            if visualize:
                color_yours = [[1,0,0], [1,0,0], [1,0,0]]
                color_gt = [[0,1,0], [0,1,0], [0,1,0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            ik_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            ik_errors.append(ik_error)
            if ik_error > IK_ERROR_THRESH:
                ik_score[file_id] -= penalty
                ik_error_cnt[file_id] += 1

        ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
        ik_errors = np.asarray(ik_errors)

        score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                    "- Mean Error : {:0.06f}\n".format(np.mean(ik_errors)) + \
                    "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                    "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                            ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)
        
        print(score_msg)
    p.removeAllUserDebugItems()

    total_ik_score = 0.0
    for file_id in range(testcase_file_num):
        total_ik_score += ik_score[file_id]
    
    print("====================================================================================")
    print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score , TASK2_SCORE_MAX))
    print("====================================================================================")

def main(args):

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.0])

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env
    robot = ur5Env(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #
    
    # warmup for 1 sec
    for _ in range(int(1 / SIM_TIMESTEP * 1)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------------------------------------------------------ #
    # --- Test your Inverse Kinematic function using one target pose --- #
    # ------------------------------------------------------------------ #
    
    # warmup for 2 secs
    p.addUserDebugText(text = "Warmup for 2 secs ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

    # test your ik solver
    testcase_files = [
        'test_case/ik_test_case_easy.json',
        'test_case/ik_test_case_medium.json',
        'test_case/ik_test_case_hard.json',
        # 'test_case/ik_test_case_ta1.json',
        # 'test_case/ik_test_case_ta2.json',
    ]

    # ------------------------------------------------------------- #
    # --- Test your Inverse Kinematic function using test cases --- #
    # ------------------------------------------------------------- #

    # scoring your algorithm
    score_ik(robot, testcase_files, visualize=args.visualize_pose)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=False, help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)