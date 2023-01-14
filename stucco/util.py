import os
from typing import Optional

import matplotlib
import torch
from matplotlib import pyplot as plt, cm as cm

import pytorch_kinematics
from pytorch3d.ops import utils as oputil
from pytorch3d.ops.points_alignment import SimilarityTransform

from stucco import cfg


def matrix_to_pos_rot(m):
    pos = m[:3, 3]
    rot = pytorch_kinematics.matrix_to_quaternion(m[:3, :3])
    rot = pytorch_kinematics.transforms.wxyz_to_xyzw(rot)
    return pos, rot


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


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


def apply_init_transform(Xt, init_transform: Optional[SimilarityTransform] = None):
    b, size_X, dim = Xt.shape
    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            R = R.clone()
            T = T.clone()
            s = s.clone()
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
        Xt = apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = oputil.eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)
    return Xt, R, T, s


def apply_similarity_transform(
        X: torch.Tensor, R: torch.Tensor, T: torch.Tensor = None, s: torch.Tensor = None
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    if s is not None:
        R = s[:, None, None] * R
    X = R @ X.transpose(-1, -2)
    if T is not None:
        X = X + T[:, :, None]
    return X.transpose(-1, -2)
