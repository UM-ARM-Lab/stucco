#! /usr/bin/env python
import argparse
import enum

import colorama
import moveit_commander
import numpy as np
import logging
import torch

from arm_pytorch_utilities import tensor_utils
from geometry_msgs.msg import PoseStamped
from stucco.poking_controller import PokingController

from stucco.retrieval_controller import RetrievalPredeterminedController, sample_model_points, rot_2d_mat_to_angle, \
    TrackingMethod, OurSoftTrackingMethod, KeyboardDirPressed
from stucco.env.real_env import VideoLogger
from stucco.env_getters.getter import EnvGetter
import os
from datetime import datetime

from stucco import cfg, util
from stucco.env import poke_real, poke
from stucco import tracking, icp
from arm_pytorch_utilities.math_utils import rotate_wrt_origin
from stucco.tracking import ContactSet

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


class StubContactSet(ContactSet):
    def __init__(self):
        pass

    def get_batch_data_for_dynamics(self, total_num):
        pass

    def dynamics(self, x, u, contact_data):
        pass

    def update(self, x, dx, p=None, info=None):
        pass


def predetermined_poke_range():
    # y,z order of poking
    return {
        poke_real.Levels.DRILL: ((0, 0.2, 0.3, -0.2, -0.3), (0.05, 0.15, 0.25, 0.325, 0.4, 0.5)),
        poke_real.Levels.CLAMP: ((0, 0.18, -0.2), (0.05, 0.08, 0.15, 0.25)),
    }


class RealPokeGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "poke_real"

    @classmethod
    def env(cls, level=poke_real.Levels.MUSTARD, **kwargs):
        level = poke_real.Levels(level)
        env = poke_real.RealPokeEnv(environment_level=level)
        return env

    @staticmethod
    def ds(env, data_dir, **kwargs):
        return None

    @staticmethod
    def controller_options(env):
        return None

    @staticmethod
    def contact_parameters(env: poke_real.RealArmEnv, **kwargs) -> tracking.ContactParameters:
        # know we are poking a single object so can be more conservative in overestimating object length scale

        params = tracking.ContactParameters(length=1.0,  # use large length scale initially to ensure everyone is 1 body
                                            penetration_length=0.002,
                                            hard_assignment_threshold=0.4,
                                            # need higher for deformable bubble
                                            intersection_tolerance=0.010)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params


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
        else:
            self.contact_detector.draw_deformation()

        self.u_history.append(u)
        return u, skip_update


class RealOurSoftTrackingMethod(OurSoftTrackingMethod):
    def create_controller(self, controls):
        self.ctrl = RealRetrievalPredeterminedController(self.env.contact_detector, self.contact_set, controls, nu=3)
        return self.ctrl


def run_poke(env: poke_real.RealPokeEnv, seed=0, control_wait=0.):
    def hook_after_poke():
        pass

    input("enter to start execution")
    y_order, z_order = predetermined_poke_range()[env.level]
    ctrl = PokingController(env.contact_detector, StubContactSet(), y_order=y_order, z_order=z_order)

    steps = 0
    # trajectory is decomposed into pokes
    pokes = 0
    obs = env._obs()
    info = None
    dtype = torch.float32

    z = env._observe_ee(return_z=True)[-1]

    # with VideoLogger(window_names=("medusa_flipped_inflated.rviz* - RViz", "medusa_flipped_inflated.rviz - RViz"),
    #                  log_external_video=True):
    # TODO logging telemetry
    while not rospy.is_shutdown() and not ctrl.done():
        with env.motion_status_input_lock:
            action = ctrl.command(obs, info)

            if action is None:
                pokes += 1
                hook_after_poke()
            util.poke_index = pokes

            if action is not None:
                if torch.is_tensor(action):
                    action = action.cpu()

                action = np.array(action).flatten()
                obs, _, done, info = env.step(action)
                print(f"pushed {action} state {obs}")

        rospy.sleep(control_wait)

    rospy.sleep(1)


def main(args):
    level = task_map[args.task]
    # obj_name = poke_real.level_to_obj_map[level]

    np.set_printoptions(suppress=True, precision=2, linewidth=200)
    # colorama.init(autoreset=True)

    # from running estimate_wrench_per_dir(env) around initial configuration
    residual_precision = np.array(
        [0.2729864752636504, 0.23987382684177347, 0.2675095350336033, 1.7901320984541171, 1.938347931365699,
         1.3659710517247037])

    env = RealPokeGetter.env(level)

    # move to the actual left side
    # env.vis.clear_visualizations(["0", "0a", "1", "1a", "c", "reaction", "tmptbest", "residualmag"])

    run_poke(env, args.seed)
    env.vis.clear_visualizations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pokes with a real robot')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='random seed to run')
    # run parameters
    parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
    task_map = {level.name.lower(): level for level in poke_real.Levels}
    parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
    main(parser.parse_args())
