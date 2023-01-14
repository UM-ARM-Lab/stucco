import abc
import os
from typing import Optional
import logging

import cma
import numpy as np

import torch
from arm_pytorch_utilities.tensor_utils import ensure_tensor

from matplotlib import pyplot as plt
from pytorch3d.ops import utils as oputil
from pytorch3d.ops.points_alignment import SimilarityTransform, ICPSolution
from pytorch_kinematics import matrix_to_rotation_6d, rotation_6d_to_matrix
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap
from stucco import cfg
from stucco.icp.costs import RegistrationCost
from stucco.util import plot_restart_losses, apply_init_transform

logger = logging.getLogger(__name__)


def plot_qd_archive(archive):
    global restart_index
    fig, ax = plt.subplots()
    grid_archive_heatmap(archive, ax=ax, vmin=-4, vmax=0)
    # for MUSTARD task
    ax.scatter(0.25, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.suptitle(f"poke {restart_index} QD: {archive.stats.norm_qd_score}")

    plt.savefig(os.path.join(cfg.DATA_DIR, 'img/restart', f"{restart_index}.png"))
    restart_index += 1


previous_solutions = None


class QDOptimization:
    def __init__(self, registration_cost: RegistrationCost,
                 model_points_world_frame: torch.tensor,
                 init_transform: Optional[SimilarityTransform] = None,
                 sigma=0.1,
                 save_loss_plot=True):
        self.registration_cost = registration_cost
        self.X = model_points_world_frame
        self.Xt, self.num_points_X = oputil.convert_pointclouds_to_tensor(self.X)
        self.B = self.Xt.shape[0]

        self.init_transform = init_transform
        self.sigma = sigma
        self.save_loss_plot = save_loss_plot

        self.device = self.Xt.device
        self.dtype = self.Xt.dtype

        Xt, R, T, s = apply_init_transform(self.Xt, self.init_transform)
        x0 = self.get_numpy_x(R[0], T[0])
        self.scheduler = self.create_scheduler(x0)
        self.restore_previous_results()

    def run(self):
        Xt, R, T, s = apply_init_transform(self.Xt, self.init_transform)

        # initialize the transformation history
        t_history = []
        losses = []

        while not self.is_done():
            cost = self.step()
            losses.append(cost)

        R, T, rmse = self.process_final_results(None, losses)

        if oputil.is_pointclouds(self.X):
            Xt = X.update_padded(Xt)  # type: ignore

        return ICPSolution(True, rmse, Xt, SimilarityTransform(R, T, s), t_history)

    @abc.abstractmethod
    def add_solutions(self, solutions):
        pass

    @abc.abstractmethod
    def create_scheduler(self, x0, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def restore_previous_results(self):
        pass

    @abc.abstractmethod
    def process_final_results(self, s, losses):
        pass

    @abc.abstractmethod
    def is_done(self):
        return False

    def get_numpy_x(self, R, T):
        q = matrix_to_rotation_6d(R)
        x = torch.cat([q, T], dim=-1).cpu().numpy()
        return x

    def get_torch_RT(self, x):
        q_ = x[..., :6]
        t = x[..., 6:]
        qq, TT = ensure_tensor(self.device, self.dtype, q_, t)
        RR = rotation_6d_to_matrix(qq)
        return RR, TT


class CMAES(QDOptimization):
    def create_scheduler(self, x0, *args, **kwargs):
        options = {"popsize": self.B, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
        options.update(kwargs)
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma, inopts=options)
        return es

    def is_done(self):
        return self.scheduler.stop()

    def step(self):
        solutions = self.scheduler.ask()
        # convert back to R, T, s
        R, T = self.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        self.scheduler.tell(solutions, cost.cpu().numpy())
        return cost

    def add_solutions(self, solutions):
        pass

    def process_final_results(self, s, losses):
        # convert ES back to R, T
        solutions = self.scheduler.ask()
        R, T = self.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            plot_restart_losses(losses)

        return R, T, rmse


class CMAME(QDOptimization):
    def __init__(self, *args, bins=40, iterations=1000,
                 # can either specify an explicit range
                 ranges=None,
                 # or form ranges from centroid of contact points and an estimated object length scale and poke offset direction
                 object_length_scale=0.1,
                 poke_offset_direction=(0.5, 0),  # default is forward along x; |offset| < 1 to represent uncertainty
                 **kwargs):
        if "sigma" not in kwargs:
            kwargs["sigma"] = 1.0
        self.bins = bins
        self.iterations = iterations
        self.ranges = ranges
        self.m = object_length_scale
        self.poke_offset_direction = poke_offset_direction

        self.archive = None
        self.i = 0
        super(CMAME, self).__init__(*args, **kwargs)

    def _create_ranges(self):
        if self.ranges is None:
            centroid = self.Xt.mean(dim=-2).mean(dim=-2).cpu().numpy()
            # extract XY (leave Z to be searched on)
            centroid = centroid[:2]
            centroid += self.m * np.array(self.poke_offset_direction)
            self.ranges = np.array((centroid - self.m, centroid + self.m)).T

    def create_scheduler(self, x0, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=len(x0), dims=[self.bins, self.bins], ranges=self.ranges)
        emitters = [
            EvolutionStrategyEmitter(self.archive, x0=x0, sigma0=1.0, batch_size=self.B)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def is_done(self):
        return self.i >= self.iterations

    @staticmethod
    def _measure(x):
        # behavior is the xy translation
        return x[..., 6:8]

    def step(self):
        self.i += 1
        solutions = self.scheduler.ask()
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        R, T = self.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        bcs = self._measure(solutions)
        self.scheduler.tell(-cost.cpu().numpy(), bcs)
        qd = self.archive.stats.norm_qd_score
        logger.debug("step %d norm QD score: %f", self.i, qd)
        return cost

    def add_solutions(self, solutions):
        assert isinstance(solutions, np.ndarray)
        R, T = self.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, None)
        self.archive.add(solutions, -rmse.cpu().numpy(), self._measure(solutions))

    def restore_previous_results(self):
        if previous_solutions is None:
            return
        self.add_solutions(previous_solutions)

    def process_final_results(self, s, losses):
        global previous_solutions
        df = self.archive.as_pandas()
        objectives = df.objective_batch()
        solutions = df.solution_batch()
        # store to allow restoring on next step
        previous_solutions = solutions
        if len(solutions) > self.B:
            order = np.argpartition(-objectives, self.B)
            solutions = solutions[order[:self.B]]
        # convert back to R, T
        R, T = self.get_torch_RT(solutions)
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            plot_qd_archive(self.archive)

        return R, T, rmse


class CMAMEGA(CMAME):
    def __init__(self, *args, lr=0.01, **kwargs):
        self.lr = lr
        super(CMAMEGA, self).__init__(*args, **kwargs)

    def create_scheduler(self, x0, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=len(x0), dims=[self.bins, self.bins], ranges=self.ranges)
        emitters = [
            GradientArborescenceEmitter(self.archive, x0=x0, sigma0=self.sigma, lr=self.lr, grad_opt="adam",
                                        selection_rule="mu", bounds=None, batch_size=self.B - 1)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def _f(self, x):
        R, T = self.get_torch_RT(x)
        return self.registration_cost(R, T, None)

    def step(self):
        solutions = self.scheduler.ask_dqd()
        bcs = self._measure(solutions)
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        # get objective gradient and also the behavior gradient
        x = ensure_tensor(self.device, self.dtype, solutions)
        x.requires_grad = True
        cost = self._f(x)
        cost.backward()
        objective_grad = -x.grad.cpu().numpy()
        objective = -cost.detach().cpu().numpy()
        # measure (aka behavior) is just the x,y, so jacobian is just identity for the corresponding dimensions
        behavior_grad = np.zeros((bcs.shape[-1], solutions.shape[-1]))
        behavior_grad[:, 6:8] = np.eye(2)
        behavior_grad = np.tile(behavior_grad, (x.shape[0], 1, 1))

        jacobian = np.concatenate((objective_grad.reshape(1, 1, -1), behavior_grad), axis=1)
        self.scheduler.tell_dqd(objective, bcs, jacobian)

        return super(CMAMEGA, self).step()
