#! /usr/bin/env python
import argparse
import numpy as np
import logging
import torch

from stucco.poking_controller import PokingController

# from stucco.env.real_env import VideoLogger
from stucco.env_getters.getter import EnvGetter
import os
from datetime import datetime

from stucco import cfg
from stucco.env import poke_real
from stucco import tracking
from stucco.tracking import ContactSet

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
logger = logging.getLogger(__name__)


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
        poke_real.Levels.DRILL: ((0, 0.1, 0.2), (-0.05, 0.0, 0.05)),
        # poke_real.Levels.CLAMP: ((0, 0.18, -0.2), (0.05, 0.08, 0.15, 0.25)),
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


def run_poke(env: poke_real.RealPokeEnv, seed=0, control_wait=0.):
    def hook_after_poke():
        logger.info(f"Finished poke {pokes} for seed {seed} of level {env.level}")

    # input("enter to start execution")
    y_order, z_order = predetermined_poke_range()[env.level]
    # these are specified relative to the rest position
    y_order = (y + env.REST_POS[1] for y in y_order)
    z_order = (z + env.REST_POS[2] for z in z_order)
    ctrl = PokingController(env.contact_detector, StubContactSet(), y_order=y_order, z_order=z_order,
                            x_rest=env.REST_POS[0], push_forward_count=2)

    steps = 0
    # trajectory is decomposed into pokes
    pokes = 0
    env.recalibrate_static_wrench()
    obs = env._obs()
    info = None
    # dtype = torch.float32

    # with VideoLogger(window_names=("medusa_flipped_inflated.rviz* - RViz", "medusa_flipped_inflated.rviz - RViz"),
    #                  log_external_video=True):
    # TODO logging telemetry
    while not rospy.is_shutdown() and not ctrl.done():
        with env.motion_status_input_lock:
            action = ctrl.command(obs, info)

            if action is None:
                pokes += 1
                hook_after_poke()

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
