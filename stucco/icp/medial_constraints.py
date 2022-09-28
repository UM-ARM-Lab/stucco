# implements a version of https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2723710/haugo2020iterative.pdf?sequence=2
import torch
import numpy as np

from pytorch3d.ops import utils as oputil
from pytorch_kinematics.transforms import random_rotations, matrix_to_rotation_6d, rotation_6d_to_matrix, Transform3d
from arm_pytorch_utilities.tensor_utils import ensure_tensor

import scipy.optimize as spo

from stucco import util
from stucco.icp.sgd import ICPSolution, SimilarityTransform, _apply_similarity_transform


def iterative_closest_point_medial_constraint(
        obj_sdf: util.ObjectFrameSDF,
        freespace_voxels: util.VoxelGrid,
        X: torch.Tensor,
        init_transform: SimilarityTransform = None,
        max_iterations: int = 100,
        relative_rmse_thr: float = 1e-6,
        verbose: bool = False,
) -> ICPSolution:
    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    b, size_X, dim = Xt.shape

    # clone the initial point cloud
    Xt_init = Xt.clone()

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

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []

    # the main loop over ICP iterations
    for iteration in range(max_iterations):
        # get the alignment of the nearest neighbors from Yt with Xt_init
        sim_transform, rmse = minimal_medial_constraint_alignment(
            obj_sdf, freespace_voxels,
            Xt_init,
            R=R,
            T=T,
        )
        R, T, s = sim_transform

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

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


def minimal_medial_constraint_alignment(
        obj_sdf: util.ObjectFrameSDF,
        freespace_voxels: util.VoxelGrid,
        X,  # surface points
        R: torch.Tensor = None, T: torch.tensor = None,
) -> tuple[SimilarityTransform, torch.tensor]:
    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = oputil.convert_pointclouds_to_tensor(X)
    b, n, dim = Xt.shape
    dtype = X.dtype
    device = X.device
    if R is None:
        R = random_rotations(b, dtype=Xt.dtype, device=Xt.device)
        T = torch.randn((b, dim), dtype=Xt.dtype, device=Xt.device)
    else:
        R = R.clone()
        T = T.clone()
    s = Xt.new_ones(b)

    model_interior_points = obj_sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < -0.01)

    def get_torch_RT(x):
        q_ = x[:6]
        t = x[6:]
        qq, TT = ensure_tensor(device, dtype, q_, t)
        RR = rotation_6d_to_matrix(qq)
        return RR.unsqueeze(0), TT.unsqueeze(0)

    def cost(x, i):
        R, T = get_torch_RT(x)
        # transform points from world frame to object frame
        world_to_object_frame = Transform3d(rot=R, pos=T, device=device, dtype=dtype).inverse()
        # sdf at the surface points should be 0
        pts = world_to_object_frame.transform_points(Xt[i])
        v, _ = obj_sdf.gt_sdf(pts)
        v = v ** 2
        v = v.sum()
        return v.item()

    def constraint(x):
        R, T = get_torch_RT(x)
        Rt = R.transpose(-1, -2)
        tt = (-Rt @ T.reshape(-1, 3, 1)).squeeze(-1)
        pts_interior_world = _apply_similarity_transform(model_interior_points, Rt, tt)
        # sdf of freespace points should be > 0
        # conversely, no interior points should lie in a freespace voxel
        occupied = freespace_voxels[pts_interior_world]
        # voxels should be 1 where it is known free space, otherwise 0
        # return occupied.reshape(-1).cpu().numpy()
        return occupied.sum().item()

    # respect_freespace_constraint = NonlinearConstraint(constraint, 0, 0)

    # convert to non-redundant representation
    q = matrix_to_rotation_6d(R)

    total_loss = []
    # TODO try to batch / parallelize the solving
    for i in range(b):
        # q, T are the parameters of the problem
        x0 = torch.cat([q[i], T[i]]).cpu().numpy()
        # set up SQP
        # TODO add constraints
        res = spo.minimize(cost, x0, args=(i,),
                           # constraints=[{
                           #     "type": "eq",
                           #     "fun": constraint,
                           # }]
                           tol=0.01,
                           options={"disp": True}
                           )
        # res = spo.dual_annealing(cost, [(-2, 2)] * 9, x0=x0, args=(i,))
        # extract q, t from res
        r, t = get_torch_RT(res.x)
        print(x0 - res.x)
        R[i] = r
        T[i] = t
        total_loss.append(res.fun)

    return SimilarityTransform(R, T, s), torch.tensor(total_loss, dtype=dtype, device=device)
