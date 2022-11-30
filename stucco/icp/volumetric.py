import enum

import numpy as np
import torch
import warnings
from typing import Union, Optional

from arm_pytorch_utilities.tensor_utils import ensure_tensor
from pytorch3d.ops import utils as oputil
from torch import optim

from stucco import util
from stucco.icp.sgd import ICPSolution, SimilarityTransform, _apply_similarity_transform
from stucco.icp.costs import VolumetricCost
from pytorch_kinematics.transforms import random_rotations, matrix_to_rotation_6d, rotation_6d_to_matrix

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stucco import cfg

import cma
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer

from stucco.svgd import RBF, SVGD

restart_index = 0
sgd_index = 0


def plot_restart_losses(losses):
    global restart_index, sgd_index
    losses = torch.stack(losses).cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_xlabel('restart iteration')
    ax.set_ylabel('cost')
    ax.set_yscale('log')
    fig.suptitle(f"poke {restart_index}")

    for b in range(losses.shape[1]):
        c = (b + 1) / losses.shape[1]
        ax.plot(losses[:, b], c=cm.GnBu(c))
    plt.savefig(os.path.join(cfg.DATA_DIR, 'img/restart', f"{restart_index}.png"))
    restart_index += 1
    sgd_index = 0


def plot_sgd_losses(losses):
    global restart_index, sgd_index
    losses = torch.stack(losses).cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_xlabel('sgd iteration')
    ax.set_ylabel('cost')
    ax.set_yscale('log')
    fig.suptitle(f"poke {restart_index} restart {sgd_index}")

    for b in range(losses.shape[1]):
        c = (b + 1) / losses.shape[1]
        ax.plot(losses[:, b], c=cm.PuRd(c))
    plt.savefig(os.path.join(cfg.DATA_DIR, 'img/sgd', f"{restart_index}_{sgd_index}.png"))
    sgd_index += 1


def plot_cmame_archive(archive):
    global restart_index
    from ribs.visualize import grid_archive_heatmap

    fig, ax = plt.subplots()
    grid_archive_heatmap(archive, ax=ax, vmin=-4, vmax=0)
    # for MUSTARD task
    # ax.scatter(0.25, 0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.suptitle(f"poke {restart_index}")

    plt.savefig(os.path.join(cfg.DATA_DIR, 'img/restart', f"{restart_index}.png"))
    restart_index += 1

class Optimization(enum.Enum):
    SGD = 0
    CMAES = 1
    CMAME = 2
    SVGD = 3


def iterative_closest_point_volumetric(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        max_iterations: int = 20,
        relative_rmse_thr: float = 1e-6,
        estimate_scale: bool = False,
        verbose: bool = False,
        save_loss_plot=True,
        **kwargs,
) -> ICPSolution:
    """
    Executes the iterative closest point (ICP) algorithm [1, 2] in order to find
    a similarity transformation (rotation `R`, translation `T`, and
    optionally scale `s`) between two given differently-sized sets of
    `d`-dimensional points `X` and `Y`, such that:

    `s[i] X[i] R[i] + T[i] = Y[NN[i]]`,

    for all batch indices `i` in the least squares sense. Here, Y[NN[i]] stands
    for the indices of nearest neighbors from `Y` to each point in `X`.
    Note, however, that the solution is only a local optimum.

    Args:
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        **init_transform**: A named-tuple `SimilarityTransform` of tensors
            `R`, `T, `s`, where `R` is a batch of orthonormal matrices of
            shape `(minibatch, d, d)`, `T` is a batch of translations
            of shape `(minibatch, d)` and `s` is a batch of scaling factors
            of shape `(minibatch,)`.
        **max_iterations**: The maximum number of ICP iterations.
        **relative_rmse_thr**: A threshold on the relative root mean squared error
            used to terminate the algorithm.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes the identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **sgd_iterations**: Number of epochs to run SGD for computing alignment
        **sgd_lr**: Learning rate of SGD for computing alignment
        **verbose**: If `True`, prints status messages during each ICP iteration.

    Returns:
        A named tuple `ICPSolution` with the following fields:
        **converged**: A boolean flag denoting whether the algorithm converged
            successfully (=`True`) or not (=`False`).
        **rmse**: Attained root mean squared error after termination of ICP.
        **Xt**: The point cloud `X` transformed with the final transformation
            (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
            instance of `Pointclouds`, otherwise returns `torch.Tensor`.
        **RTs**: A named tuple `SimilarityTransform` containing
        a batch of similarity transforms with fields:
            **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
            **T**: Batch of translations of shape `(minibatch, d)`.
            **s**: batch of scaling factors of shape `(minibatch, )`.
        **t_history**: A list of named tuples `SimilarityTransform`
            the transformation parameters after each ICP iteration.
    """

    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    b, size_X, dim = Xt.shape
    Xt, R, T, s = apply_init_transform(Xt, init_transform)

    # clone the initial point cloud
    Xt_init = Xt.clone()

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []
    losses = []

    # --- SGD
    for iteration in range(max_iterations):
        # get the alignment of the nearest neighbors from Yt with Xt_init
        sim_transform, rmse = volumetric_points_alignment(
            volumetric_cost,
            Xt_init,
            estimate_scale=estimate_scale,
            R=R,
            T=T,
            s=s,
            save_loss_plot=save_loss_plot,
            **kwargs
        )
        R, T, s = sim_transform
        losses.append(rmse)

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                    f"ICP iteration {iteration}: mean/max rmse = "
                    + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                    + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            print(rmse_msg)

        # check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if save_loss_plot:
        plot_restart_losses(losses)

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


def apply_init_transform(Xt, init_transform: Optional[SimilarityTransform] = None):
    b, size_X, dim = Xt.shape
    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            assert (
                    R.shape == torch.Size((b, dim, dim))
                    and T.shape == torch.Size((b, dim))
                    and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            )
        # apply the init transform to the input point cloud
        Xt = _apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = oputil.eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)
    return Xt, R, T, s


def iterative_closest_point_volumetric_cmaes(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        sigma=0.1,
        save_loss_plot=True,
        **kwargs,
) -> ICPSolution:
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Xt, R, T, s = apply_init_transform(Xt, init_transform)
    b = Xt.shape[0]

    converged = False

    # initialize the transformation history
    t_history = []
    losses = []

    # --- CMA-ES
    device = Xt.device
    dtype = Xt.dtype

    def get_torch_RT(x):
        q_ = x[:, :6]
        t = x[:, 6:]
        qq, TT = ensure_tensor(device, dtype, q_, t)
        RR = rotation_6d_to_matrix(qq)
        return RR, TT

    # TODO investigate why we're always offset by a little
    q0 = matrix_to_rotation_6d(R[0])
    T0 = T[0]
    x0 = torch.cat([q0, T0]).cpu().numpy()
    options = {"popsize": b, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
    options.update(kwargs)
    es = cma.CMAEvolutionStrategy(x0=x0, sigma0=sigma, inopts=options)
    while not es.stop():
        solutions = es.ask()
        # convert back to R, T, s
        R, T = get_torch_RT(np.stack(solutions))
        cost = volumetric_cost(R, T, s)
        losses.append(cost)
        # cost = cost.numpy()
        es.tell(solutions, cost.cpu().numpy())

    # convert ES back to R, T
    solutions = es.ask()
    R, T = get_torch_RT(np.stack(solutions))
    # print("init")
    # print(T0)
    # print("sol")
    # print(T[0])
    rmse = volumetric_cost(R, T, s)

    if save_loss_plot:
        plot_restart_losses(losses)

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


def iterative_closest_point_volumetric_cmame(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        bins=40,
        iterations=1000,
        ranges=None,
        save_loss_plot=True,
        **kwargs,
) -> ICPSolution:
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Xt, R, T, s = apply_init_transform(Xt, init_transform)
    b = Xt.shape[0]

    # initialize the transformation history
    t_history = []
    losses = []

    # --- CMA-ME
    device = Xt.device
    dtype = Xt.dtype

    def get_torch_RT(x):
        q_ = x[:, :6]
        t = x[:, 6:]
        qq, TT = ensure_tensor(device, dtype, q_, t)
        RR = rotation_6d_to_matrix(qq)
        return RR, TT

    q0 = matrix_to_rotation_6d(R[0])
    T0 = T[0]
    x0 = torch.cat([q0, T0]).cpu().numpy()

    # extract ranges from the boundaries of the free voxels
    if ranges is None:
        if isinstance(volumetric_cost.free_voxels, util.VoxelGrid):
            ranges = volumetric_cost.free_voxels.range_per_dim[:2]
        else:
            raise RuntimeError("Range not given and cannot be inferred from the freespace voxels")

    archive = GridArchive([bins, bins], ranges)
    emitters = [
        ImprovementEmitter(archive, x0, 1.0, batch_size=b)
    ]
    optimizer = Optimizer(archive, emitters)
    for i in range(iterations):
        solutions = optimizer.ask()
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        R, T = get_torch_RT(np.stack(solutions))
        cost = volumetric_cost(R, T, s)
        losses.append(cost)
        # behavior is the xy translation
        bcs = solutions[:, 6:8]
        optimizer.tell(-cost.cpu().numpy(), bcs)

    df = archive.as_pandas()
    objectives = df.batch_objectives()
    solutions = df.batch_solutions()
    if len(solutions) > b:
        order = np.argpartition(-objectives, b)
        solutions = solutions[order[:b]]
    # convert back to R, T
    R, T = get_torch_RT(solutions)
    rmse = volumetric_cost(R, T, s)

    if save_loss_plot:
        plot_cmame_archive(archive)

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(True, rmse, Xt, SimilarityTransform(R, T, s), t_history)


class CostProb:
    def __init__(self, cost, scale=1.):
        self.cost = cost
        self.scale = scale

    def log_prob(self, X, s):
        # turn into R, T
        q = X[:, :6]
        T = X[:, 6:9]
        R = rotation_6d_to_matrix(q)
        c = self.cost(R, T, s)
        # p = N exp(-c * self.scale)
        # logp \propto -c * self.scale
        return -c * self.scale


def iterative_closest_point_volumetric_svgd(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        max_iterations: int = 300,
        lr=0.005,
        kernel_scale=0.01,  # None indicates to use the median heuristic
        cost_scale=5.,
        save_loss_plot=True,
) -> ICPSolution:
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Xt, R, T, s = apply_init_transform(Xt, init_transform)

    converged = False

    # initialize the transformation history
    t_history = []
    losses = []

    # SVGD
    K = RBF(kernel_scale)

    q = matrix_to_rotation_6d(R)
    params = torch.cat([q, T], dim=1)
    svgd = SVGD(CostProb(volumetric_cost, scale=cost_scale), K, optim.Adam([params], lr=lr))
    for i in range(max_iterations):
        # convert back to R, T, s
        logprob = svgd.step(params, s)
        cost = -logprob / svgd.P.scale
        losses.append(cost)

    # convert ES back to R, T
    q = params[:, :6]
    T = params[:, 6:9]
    R = rotation_6d_to_matrix(q)
    rmse = volumetric_cost(R, T, s)

    if save_loss_plot:
        plot_restart_losses(losses)

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


def volumetric_points_alignment(
        volumetric_cost: VolumetricCost,
        X: Union[torch.Tensor, "Pointclouds"],
        estimate_scale: bool = False,
        R: torch.Tensor = None, T: torch.tensor = None, s: torch.tensor = None,
        iterations: int = 50,
        lr: float = 0.01,
        save_loss_plot=True,
        verbose=False
) -> tuple[SimilarityTransform, torch.tensor]:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense using gradient descent

    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **sgd_iterations**: Number of epochs to run
        **lr**: Learning rate

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = oputil.convert_pointclouds_to_tensor(X)
    b, n, dim = Xt.shape

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    if R is None:
        R = random_rotations(b, dtype=Xt.dtype, device=Xt.device)
        T = torch.randn((b, dim), dtype=Xt.dtype, device=Xt.device)
        s = torch.ones(b, dtype=Xt.dtype, device=Xt.device)

    # convert to non-redundant representation
    q = matrix_to_rotation_6d(R)

    # set them up as parameters for training
    q.requires_grad = True
    T.requires_grad = True
    if estimate_scale:
        s.requires_grad = True

    optimizer = torch.optim.Adam([q, T, s], lr=lr)

    def get_usable_transform_representation():
        nonlocal T
        # we get a more usable representation of R
        R = rotation_6d_to_matrix(q)
        return R, T

    losses = []

    for epoch in range(iterations):
        R, T = get_usable_transform_representation()

        total_loss = volumetric_cost(R, T, s)
        total_loss.mean().backward()
        losses.append(total_loss.detach())

        # visualize gradients on the losses
        volumetric_cost.visualize(R, T, s)

        optimizer.step()
        optimizer.zero_grad()

    if save_loss_plot:
        plot_sgd_losses(losses)

    if verbose:
        print(f"pose loss {total_loss.mean().item()}")
    R, T = get_usable_transform_representation()
    return SimilarityTransform(R.detach().clone(), T.detach().clone(),
                               s.detach().clone()), total_loss.detach().clone()
