import typing

from arm_pytorch_utilities import load_data
from arm_pytorch_utilities.optim import get_device
from stucco import tracking
from stucco.env import poke
from stucco.env_getters.getter import EnvGetter


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
        params = tracking.ContactParameters(length=0.03,
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
        env = poke.PokeEnv(environment_level=level, log_video=log_video, **kwargs)
        cls.env_dir = '{}/floating'.format(poke.DIR)
        return env
