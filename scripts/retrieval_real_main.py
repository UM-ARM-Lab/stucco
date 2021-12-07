#! /usr/bin/env python
import argparse
import enum
import time

import colorama
import moveit_commander
import numpy as np
import logging
import torch
import copy

from arm_pytorch_utilities import tensor_utils
from geometry_msgs.msg import PoseStamped
from sklearn.cluster import Birch, DBSCAN, KMeans

from stucco.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from stucco.evaluation import object_robot_penetration_score
from stucco.retrieval_controller import RetrievalPredeterminedController, sample_model_points, rot_2d_mat_to_angle, \
    SklearnTrackingMethod, TrackingMethod, OurSoftTrackingMethod, SklearnPredeterminedController, KeyboardDirPressed
from stucco.env.real_env import VideoLogger
from stucco.env.arm import Levels
from stucco.env_getters.getter import EnvGetter
import os
from datetime import datetime

from stucco import cfg
from stucco.env import arm_real
from stucco import tracking, icp
from arm_pytorch_utilities.math_utils import rotate_wrt_origin
from victor_hardware_interface_msgs.msg import ControlMode

try:
    import rospy

    rospy.init_node("victor_retrieval", log_level=rospy.INFO)
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

ask_before_moving = True
CONTACT_POINT_SIZE = (0.01, 0.01, 0.3)

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True


def no_function():
    raise RuntimeError("This function shouldn't be run!")


class RealRetrievalGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "arm_real"

    @classmethod
    def env(cls, level=Levels.NO_CLUTTER, **kwargs):
        level = Levels(level)
        # env = arm_real.RealArmEnv(environment_level=level)
        env = arm_real.RealArmEnvMedusa(environment_level=level)
        return env

    @staticmethod
    def ds(env, data_dir, **kwargs):
        return None

    @staticmethod
    def controller_options(env):
        return None

    @staticmethod
    def contact_parameters(env: arm_real.RealArmEnv, **kwargs) -> tracking.ContactParameters:
        params = tracking.ContactParameters(length=0.006,
                                            penetration_length=0.002,
                                            hard_assignment_threshold=0.4,
                                            # need higher for deformable bubble
                                            intersection_tolerance=0.010)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params


def estimate_wrench_per_dir(env):
    pushed = KeyboardDirPressed()
    dir_to_wrench = {}
    print("waiting for arrow keys to be pressed to command a movement")
    with VideoLogger():
        while not rospy.is_shutdown():
            push = tuple(pushed.dir)
            if push[0] != 0 or push[1] != 0:
                if push not in dir_to_wrench:
                    dir_to_wrench[push] = []

                env.static_wrench = None
                env._temp_wrenches = []
                obs, _, done, info = env.step(push)
                dir_to_wrench[push] += env._temp_wrenches

                print(f"pushed {push} state {obs}")
            rospy.sleep(0.1)

    for k, v in dir_to_wrench.items():
        print(f"{k} ({len(v)} points): {np.mean(v, axis=0)} {np.var(v, axis=0)}")
    print("code copy friendly print")
    for k, v in dir_to_wrench.items():
        print(f"{k}: {list(np.mean(v, axis=0))}")

    combined = sum(dir_to_wrench.values(), [])
    print(f"var : {list(np.var(combined, axis=0))}")
    print(f"prec: {list(1 / np.var(combined, axis=0))}")


def confirm_pt_to_config(env, pt_to_config):
    # confirm pt to config implementation
    config = env.state
    from arm_pytorch_utilities import rand
    import torch
    rand.seed(2)
    pts = (torch.rand((20, 2)) - 0.5) * 0.3
    pts += config
    pts[:, 1] += 0.1
    d = pt_to_config(torch.from_numpy(config).view(1, -1), pts)
    d = d.view(-1)
    for i, pt in enumerate(pts):
        env.vis.ros.draw_point(f'temp.{i}', pt, height=env.REST_POS[2], label=str(round(d[i].item(), 2)),
                               color=(1, 1, 1, 1))
        print(d[i])


def debug_pt_to_config(env, pt_to_config, config, point):
    d = pt_to_config(torch.from_numpy(config).view(1, -1), point)
    d = d.view(-1)
    env.vis.ros.draw_point(f'temp.0', point[0], height=env.REST_POS[2], label=str(round(d[0].item(), 2)),
                           color=(1, 1, 1, 1))
    print(d[0])


def keyboard_control(env):
    pushed = KeyboardDirPressed()
    print("waiting for arrow keys to be pressed to command a movement")
    with VideoLogger():
        while not rospy.is_shutdown():
            env.contact_detector.get_last_contact_location(visualizer=env.vis.ros)
            if pushed.calibrate:
                env.recalibrate_static_wrench()
            push = tuple(pushed.dir)
            if push[0] != 0 or push[1] != 0:
                obs, _, done, info = env.step(push)
                print(f"pushed {push} state {obs}")
            rospy.sleep(0.1)

def _skip_update(x_history, u_history, u):
    # 3 elements in a control means to perform it but not calibrate (and ignore whether we think we're in contact or not)
    # skip if we were calibrating with the last control, or if we are currently calibrating
    return (type(u) is SpecialActions) or (len(x_history) < 2) or (
            (type(u_history[-1]) is SpecialActions) or (len(u_history[-1]) > 2))


class RealRetrievalPredeterminedController(RetrievalPredeterminedController):
    def __init__(self, contact_detector, contact_set, controls, nu=None):
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.contact_indices = []
        super().__init__(controls, nu=nu)

    def command(self, obs, info=None, visualizer=None):
        self.x_history.append(obs)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = self.controls[self.i]
            self.i += 1

        skip_update = _skip_update(self.x_history, self.u_history, u)
        if not skip_update:
            if self.contact_detector.in_contact():
                self.contact_indices.append(self.i)

            x = self.x_history[-1][:2]
            pt, dx = self.contact_detector.get_last_contact_location(visualizer=visualizer)
            info['u'] = torch.tensor(self.u_history[-1][:2])
            self.contact_set.update(x, dx, pt, info=info)

        self.u_history.append(u)
        return u, skip_update


class RealSklearnPredeterminedController(SklearnPredeterminedController):

    def command(self, obs, info=None, visualizer=None):
        self.x_history.append(obs)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = self.controls[self.i]
            self.i += 1

        skip_update = _skip_update(self.x_history, self.u_history, u)
        if not skip_update:
            self.update(obs, info, visualizer=visualizer)

        self.u_history.append(u)
        return u, skip_update


class RealOurSoftTrackingMethod(OurSoftTrackingMethod):
    def create_predetermined_controller(self, controls):
        self.ctrl = RealRetrievalPredeterminedController(self.env.contact_detector, self.contact_set, controls, nu=2)
        return self.ctrl


class RealSklearnTrackingMethod(SklearnTrackingMethod):
    def create_predetermined_controller(self, controls):
        self.ctrl = RealSklearnPredeterminedController(self.online_method, self.env.contact_detector, controls, nu=2)
        return self.ctrl


def run_retrieval(env, level, pt_to_config, method: TrackingMethod, control_wait=0.):
    input("enter to start execution")
    controls, ret_ctrl = create_predetermined_controls(level)
    ctrl: RetrievalPredeterminedController = method.create_predetermined_controller(controls)
    obs = env._obs()
    info = None
    dtype = torch.float32

    z = env._observe_ee(return_z=True)[-1]

    model_points, bb = sample_model_points(None, num_points=50, force_z=z, seed=0, name="cheezit")
    bb = bb.to(dtype=dtype)
    mph = model_points.clone().to(dtype=dtype)
    # make homogeneous [x, y, 1]
    mph[:, -1] = 1

    best_tsf_guess = None

    with VideoLogger(window_names=("medusa_flipped_inflated.rviz* - RViz", "medusa_flipped_inflated.rviz - RViz"),
                     log_external_video=True):
        while not rospy.is_shutdown() and not ctrl.done():
            best_distance = None

            with env.motion_status_input_lock:
                u, skip_update = ctrl.command(obs, info, env.vis.ros)
            method.visualize_contact_points(env)

            if env.contact_detector.in_contact() and not skip_update:
                all_configs = torch.tensor(np.array(ctrl.x_history), dtype=dtype, device=mph.device).view(-1, env.nx)
                dist_per_est_obj = []
                for this_pts in method:
                    this_pts = tensor_utils.ensure_tensor(model_points.device, dtype, this_pts)
                    T, distances, _ = icp.icp_3(this_pts.view(-1, 2), model_points[:, :2],
                                                given_init_pose=best_tsf_guess, batch=50)
                    T = T.inverse()
                    penetration = [object_robot_penetration_score(pt_to_config, all_configs, T[b], mph) for b in
                                   range(T.shape[0])]
                    score = np.abs(penetration)
                    best_tsf_index = np.argmin(score)

                    # pick object with lowest variance in its translation estimate
                    translations = T[:, :2, 2]
                    best_tsf_distances = (translations.var(dim=0).sum()).item()
                    yaws = rot_2d_mat_to_angle(T)

                    dist_per_est_obj.append(best_tsf_distances)
                    if best_distance is None or best_tsf_distances < best_distance:
                        best_distance = best_tsf_distances
                        best_tsf_guess = T[best_tsf_index].inverse()

                rospy.loginfo(f"err each obj {np.round(dist_per_est_obj, 4)}")
                best_T = best_tsf_guess.inverse()

                # transformed_model_points = mph @ best_T.transpose(-1, -2)
                # for i, pt in enumerate(transformed_model_points):
                #     pt = [pt[0], pt[1], z]
                #     env.vis.ros.draw_point(f"tmptbest.{i}", pt, color=(0, 0, 1), length=0.008)

                tf_bb = bb @ best_T.transpose(-1, -2)
                for i in range(len(tf_bb)):
                    pt = [tf_bb[i][0], tf_bb[i][1], z]
                    next_pt = [tf_bb[(i + 1) % len(tf_bb)][0], tf_bb[(i + 1) % len(tf_bb)][1], z]
                    env.vis.draw_2d_line(f"tmptbestline.{i}", pt, np.subtract(next_pt, pt), color=(0, 0, 1), size=2,
                                         scale=1)

            if u is SpecialActions.RECALIBRATE:
                env.recalibrate_static_wrench()
                obs = env._obs()
                info = None
                continue
            elif u is SpecialActions.WAIT_FOR_INPUT:
                user_input = input("Enter to continue execution")
                if user_input == "q":
                    exit(0)
                if user_input != "":
                    try:
                        next_u = [float(d) for d in user_input.split(" ")]
                        ctrl.insert_next_controls([next_u, SpecialActions.WAIT_FOR_INPUT])
                    except:
                        pass
                obs = env._obs()
                info = None
                continue

            obs, _, done, info = env.step(u[:2])
            print(f"pushed {u} state {obs}")
            rospy.sleep(control_wait)

        # after estimating pose, plan a grasp to it and attempt a grasp
        if best_tsf_guess is not None:
            guess_pose = [best_T[0, 2].item(), best_T[1, 2].item(), rot_2d_mat_to_angle(best_T.view(1, 3, 3)).item()]

            moveit_scene = moveit_commander.PlanningSceneInterface(ns=env.robot.robot_namespace)
            _, pts = method.get_labelled_moved_points(None)

            for pt_index, pt in enumerate(pts):
                pose = PoseStamped()
                pose.header.frame_id = "world"
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
                pose.pose.position.z = env.REST_POS[2]
                pose.pose.orientation.w = 1.0
                moveit_scene.add_box(f"contact_pt_{pt_index}", pose, size=CONTACT_POINT_SIZE)

            grasp_at_pose(env, guess_pose, ret_ctrl=ret_ctrl)

            for pt_index, pt in enumerate(pts):
                moveit_scene.remove_world_object(f"contact_pt_{pt_index}")

        rospy.sleep(1)


def grasp_at_pose(self: arm_real.RealArmEnvMedusa, pose, ret_ctrl=(), timeout=40):
    rospy.loginfo(f"grasp pose {pose}")

    # should be safe to return to original pose
    z_extra = 0.05
    z = self.REST_POS[2] + z_extra
    self.vis.ros.draw_point("grasptarget", [pose[0], pose[1], z], color=(0.7, 0, 0.7), scale=3)

    grasp_offset = np.array([0.42, 0])
    yaw = copy.copy(pose[2])
    yaw += np.pi / 2
    # bring between [-np.pi, 0]
    if yaw > 0 and yaw < np.pi:
        yaw -= np.pi
    elif yaw < -np.pi:
        yaw += np.pi

    grasp_offset = rotate_wrt_origin(grasp_offset, yaw)

    offset_pos = [pose[0] + grasp_offset[0], pose[1] + grasp_offset[1], z]
    orientation = [self.REST_ORIENTATION[0], self.REST_ORIENTATION[1],
                   yaw + np.pi / 2 + self.REST_ORIENTATION[2]]
    self.vis.ros.draw_point("pregrasp", offset_pos, color=(1, 0, 0), scale=3)

    resp = input("is the planned grasp good?")
    if resp == "n":
        return

    # first get to a location where planning to the previous joint config is ok
    for u in ret_ctrl:
        self.step(u)
    # self.robot.set_control_mode(control_mode=ControlMode.JOINT_POSITION, vel=self.vel)
    # self.robot.plan_to_pose(self.robot.arm_group, self.EE_LINK_NAME, offset_pos + orientation)
    # self.robot.set_control_mode(control_mode=ControlMode.CARTESIAN_IMPEDANCE, vel=self.vel)

    # cartesian to target
    obs = self._obs()
    diff = np.subtract(offset_pos[:2], obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < timeout:
        obs, _, _, _ = self.step(diff / self.MAX_PUSH_DIST, dz=z_extra, orientation=orientation)
        diff = np.subtract(offset_pos[:2], obs)
    if time.time() - start > timeout:
        return

    # self.robot.open_right_gripper(0.15)
    self.robot.gripper.move(100)
    rospy.sleep(1)

    goal_pos = [pose[0] + grasp_offset[0] * 0.6, pose[1] + grasp_offset[1] * 0.6, z]
    self.vis.ros.draw_point("graspgoal", goal_pos, color=(0.5, 0.5, 0), scale=3)

    obs = self._obs()
    diff = np.subtract(goal_pos[:2], obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < timeout:
        u = diff / self.MAX_PUSH_DIST
        u /= max(abs(u))
        obs, _, _, _ = self.step(u, dz=z_extra, orientation=orientation)
        diff = np.subtract(goal_pos[:2], obs)
    if time.time() - start > timeout:
        return

    # self.robot.close_right_gripper()
    self.robot.gripper.move(0)
    rospy.sleep(3)

    # actually retrieve the object to demonstrate we have a good grasp of it
    obs = self._obs()
    diff = np.subtract(offset_pos[:2], obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < timeout:
        u = diff / self.MAX_PUSH_DIST
        u /= max(abs(u))
        obs, _, _, _ = self.step(u, dz=z_extra, orientation=orientation)
        diff = np.subtract(offset_pos[:2], obs)

    # self.robot.open_right_gripper(0)
    self.robot.gripper.move(100)
    rospy.sleep(1)


class Levels(enum.IntEnum):
    NO_CLUTTER = 0
    FLAT_BOX = 1
    CAN_IN_FRONT = 2
    CALIBRATION = 3


class SpecialActions(enum.Enum):
    RECALIBRATE = 0
    WAIT_FOR_INPUT = 1


def create_predetermined_controls(level: Levels):
    ctrl = None
    ret_ctrl = None
    if level is Levels.CAN_IN_FRONT:
        # ctrl = [SpecialActions.RECALIBRATE] + [[0., 0.]] * 20
        ctrl = [[0.0, 0.1], SpecialActions.RECALIBRATE]
        ctrl += [[0.0, 0.9]]
        ctrl += [[0.0, 0.7]]
        # poke master chef can to the right
        ctrl += [[0.1, 0, ]]
        ctrl += [SpecialActions.RECALIBRATE]
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[0.8, 0.]]

        ctrl += [[1.0, 0], [-0.5, 0.3, None]] * 3
        ctrl += [[1.0, 0]]
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[0.0, -0.3, None], [0.1, 0.1], SpecialActions.RECALIBRATE]
        ctrl += [[1.0, 1.0], [0.3, -0.5, None]] * 3

        ctrl += [[0.6, 1.0]]
        ctrl += [[0.3, 0.6]]
        ctrl += [[-0.3, -0.6, None]]
        ctrl += [[-0.6, -1.0, None]]
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]

        # poke kettle
        ctrl += [[-0.5, -0.5, None]]
        ctrl += [[-0.5, 0], SpecialActions.RECALIBRATE]
        ctrl += [[-1.0, 0]] * 3
        ctrl += [[-0.5, 0], SpecialActions.RECALIBRATE]
        ctrl += [[-1.0, 0]] * 2
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[-0.1, -0.5, None]] * 2
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[-0.2, 0.12], SpecialActions.RECALIBRATE]
        ctrl += [[-0.9, 0.5], [-0.9, 0.6], [-0.2, -0.3, None]] * 3

        ctrl += [[0.4, 0.3, None], [-1, 0.6]] * 2
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]

        # move in front of cheezit box
        ctrl += [[0, -1.0], SpecialActions.RECALIBRATE] * 4
        ctrl += [[1.0, 0], SpecialActions.RECALIBRATE] * 9
        ctrl += [[0.5, 0], SpecialActions.RECALIBRATE] * 1

        ctrl += [[0, 0.7], SpecialActions.RECALIBRATE] * 1
        # poke cheezit box
        ctrl += [[0, 1.0], [-0.85, -0.3, None]] * 3
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]

        # # move in between to poke both
        ctrl += [[-1.0, .0, None]] * 3
        ctrl += [[-0.3, 0.3], SpecialActions.RECALIBRATE]
        ctrl += [[-1.0, 1.0]] * 3
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[0.0, 0.5], SpecialActions.RECALIBRATE]
        ctrl += [[0.0, 1.0]] * 4
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        ctrl += [[0.9, 0], SpecialActions.RECALIBRATE]
        # ctrl += [[1, 0]]
        # ctrl += [SpecialActions.WAIT_FOR_INPUT]
        # ctrl += [[0.2, 0], None]
        # ctrl += [[1, 0], None, [1.0, 0], [0.5, 0]]
        ctrl += [[0.6, 0], [-0.2, 1.0, None]] * 5

        # ctrl += [None]
        # ctrl += [[0, 1], None]

        # ret_ctrl = [[0.9, 0]]

        ret_ctrl = [[-0.9, 0]] * 2
        ret_ctrl += [[0, -1.0]] * 9
        ret_ctrl += [[0.4, -1.0]] * 4
    elif level is Levels.CALIBRATION:
        ctrl = [SpecialActions.RECALIBRATE]
        ctrl += [[0, 1.0]] * 3
        ctrl += [SpecialActions.WAIT_FOR_INPUT]

    # last one to force redraw of everything
    ctrl += [[0.0, 0.]]
    return ctrl, ret_ctrl


parser = argparse.ArgumentParser(description='Downstream task of blind object retrieval')
parser.add_argument('method',
                    choices=['ours', 'online-birch', 'online-dbscan', 'online-kmeans'],
                    help='which method to run')
args = parser.parse_args()


def main():
    level = Levels.CAN_IN_FRONT

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    colorama.init(autoreset=True)

    # from running estimate_wrench_per_dir(env) around initial configuration
    residual_precision = np.array(
        [0.2729864752636504, 0.23987382684177347, 0.2675095350336033, 1.7901320984541171, 1.938347931365699,
         1.3659710517247037])
    # ignore fz since it's usually large and noisy
    residual_precision[2] = 0

    env = arm_real.RealArmEnvMedusa(residual_precision=np.diag(residual_precision), residual_threshold=5., vel=0.25)
    contact_params = RealRetrievalGetter.contact_parameters(env)

    pt_to_config = arm_real.RealArmPointToConfig(env)

    methods_to_run = {
        'ours': RealOurSoftTrackingMethod(env, contact_params, pt_to_config),
        'online-birch': RealSklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                                  inertia_ratio=0.2, threshold=0.08),
        'online-dbscan': RealSklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.1, min_samples=1),
        'online-kmeans': RealSklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.1,
                                                   n_clusters=1, random_state=0)
    }

    # confirm_pt_to_config(env, pt_to_config)
    # estimate_wrench_per_dir(env)
    # keyboard_control(env)

    # move to the actual left side
    env.vis.clear_visualizations(["0", "0a", "1", "1a", "c", "reaction", "tmptbest", "residualmag"])

    # test grasp at pose
    # grasp_at_pose(env, [0.6395749449729919, 0.4507047235965729, -0.04405221343040466])

    run_retrieval(env, level, pt_to_config, methods_to_run[args.method])
    env.vis.clear_visualizations()


if __name__ == "__main__":
    main()
