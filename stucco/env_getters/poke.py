import typing

from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.optim import get_device
from stucco import tracking
from stucco.env import poke
from stucco.env_getters.getter import EnvGetter
import math


class PokeGetter(EnvGetter):
    @staticmethod
    def dynamics_prefix() -> str:
        return "poke"

    @staticmethod
    def ds(env, data_dir, **kwargs):
        d = get_device()
        config = load_data.DataConfig(predict_difference=True, predict_all_dims=True, expanded_input=False)
        ds = poke.PokeDataSource(env, data_dir=data_dir, config=config, device=d, **kwargs)
        return ds

    @staticmethod
    def controller_options(env) -> typing.Tuple[dict, dict]:
        return {}, {}

    @staticmethod
    def contact_parameters(env: poke.PokeEnv, **kwargs) -> tracking.ContactParameters:
        params = tracking.ContactParameters(length=1.0, # use large length scale initially to ensure everyone is 1 body
                                            penetration_length=0.002,
                                            hard_assignment_threshold=0.4,
                                            intersection_tolerance=0.005)
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(params, k, v)
        return params

    @classmethod
    def env(cls, level=0, log_video=True, **kwargs):
        level = poke.Levels(level)
        goal = (0.25, 0.0, 0.2, 0.)
        if level in [poke.Levels.MUSTARD, poke.Levels.DRILL]:
            goal = (0.25, 0.0, 0.2, 0.)
        if level in [poke.Levels.DRILL_OPPOSITE]:
            goal = (0.42, 0.0, 0.2, math.pi)
        if level in [poke.Levels.DRILL_SLANTED]:
            goal = (0.35, 0.1, 0.2, 2 * math.pi/3)
        if level in [poke.Levels.DRILL_FALLEN]:
            goal = (0.25, 0.0, 0.2, math.pi/2, 0, 0.2)
        if level in [poke.Levels.MUSTARD_SIDEWAYS]:
            goal = (0.25, 0.0, 0.2, math.pi/2)
        if level in [poke.Levels.MUSTARD_FALLEN]:
            goal = (0.25, 0.0, 0.2, math.pi/2, 0, 0.2)
        if level in [poke.Levels.MUSTARD_FALLEN_SIDEWAYS]:
            goal = (0.25, 0.0, 0.2, math.pi/2, math.pi/2, 0.4)
        if level in [poke.Levels.HAMMER]:
            goal = (0.25, -0.18, 0.42, -math.pi / 2, math.pi, math.pi / 2)
        if level in [poke.Levels.HAMMER_1]:
            goal = (0.55, 0.0, 0.3, math.pi/2, 1.2, 0)
        if level in [poke.Levels.HAMMER_2]:
            goal = (0.3, -0.1, 0.3, -0.3, 0.4, 0.4)
        env = poke.PokeEnv(environment_level=level, goal=goal, log_video=log_video, **kwargs)
        cls.env_dir = '{}/floating'.format(poke.DIR)
        return env
