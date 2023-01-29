import enum

import numpy as np
import torch

from pytorch_kinematics import transforms as tf
from stucco.icp.stein import icp_stein
from stucco.icp.methods import icp, icp_2, icp_3, icp_pytorch3d, icp_pytorch3d_sgd, icp_volumetric, icp_mpc, \
    icp_medial_constraints


class ICPMethod(enum.Enum):
    ICP = 0
    ICP_SGD = 1

    VOLUMETRIC = 2
    VOLUMETRIC_ICP_INIT = 7
    VOLUMETRIC_NO_FREESPACE = 3
    VOLUMETRIC_LIMITED_REINIT = 9
    VOLUMETRIC_CMAES = 10
    VOLUMETRIC_CMAME = 14
    VOLUMETRIC_CMAMEGA = 15
    VOLUMETRIC_SVGD = 11
    VOLUMETRIC_LIMITED_REINIT_FULL = 13

    ICP_SGD_VOLUMETRIC_NO_ALIGNMENT = 4
    ICP_REVERSE = 5
    ICP_SGD_REVERSE = 6

    # freespace baselines
    MEDIAL_CONSTRAINT = 8
    CVO = 12

    NONE = 16


class InitMethod(enum.Enum):
    ORIGIN = 0
    CONTACT_CENTROID = 1
    RANDOM = 2


def registration_method_uses_only_contact_points(reg_method: ICPMethod):
    if reg_method in [ICPMethod.ICP, ICPMethod.ICP_SGD, ICPMethod.ICP_REVERSE,
                      ICPMethod.ICP_SGD_REVERSE, ICPMethod.VOLUMETRIC_NO_FREESPACE]:
        return True
    return False


def initialize_transform_estimates(B, freespace_ranges, init_method: InitMethod, contact_points,
                                   device="cpu", dtype=torch.float):
    # translation is 0,0,0
    best_tsf_guess = random_upright_transforms(B, dtype, device)
    if init_method == InitMethod.ORIGIN:
        pass
    elif init_method == InitMethod.CONTACT_CENTROID:
        centroid = contact_points.mean(dim=0).to(device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = centroid
    elif init_method == InitMethod.RANDOM:
        trans = np.random.uniform(freespace_ranges[:, 0], freespace_ranges[:, 1], (B, 3))
        trans = torch.tensor(trans, device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = trans
    else:
        raise RuntimeError(f"Unsupported initialization method: {init_method}")
    return best_tsf_guess


def random_upright_transforms(B, dtype, device, translation=None):
    # initialize guesses with a prior; since we're trying to retrieve an object, it's reasonable to have the prior
    # that the object only varies in yaw (and is upright)
    axis_angle = torch.zeros((B, 3), dtype=dtype, device=device)
    axis_angle[:, -1] = torch.rand(B, dtype=dtype, device=device) * 2 * np.pi
    R = tf.axis_angle_to_matrix(axis_angle)
    init_pose = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    init_pose[:, :3, :3] = R
    if translation is not None:
        init_pose[:, :3, 3] = translation
    return init_pose
