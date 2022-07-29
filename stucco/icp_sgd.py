import torch
import warnings
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union
from torch.nn import MSELoss

from pytorch3d.ops import knn_points
from pytorch3d.ops import utils as oputil
from pytorch3d.structures import utils as strutil
from pytorch3d.ops.points_alignment import ICPSolution, SimilarityTransform, _apply_similarity_transform
from pytorch3d.ops.knn import _KNN
from pytorch3d.transforms import random_rotations, matrix_to_rotation_6d, rotation_6d_to_matrix


def iterative_closest_point_sgd(
        X: Union[torch.Tensor, "Pointclouds"],
        Y: Union[torch.Tensor, "Pointclouds"],
        init_transform: Optional[SimilarityTransform] = None,
        max_iterations: int = 100,
        relative_rmse_thr: float = 1e-6,
        estimate_scale: bool = False,
        allow_reflection: bool = False,
        sgd_iterations: int = 50,
        sgd_lr: float = 0.002,
        verbose: bool = False,
        learn_translation=True,
        pose_cost=None
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

    References:
        [1] Besl & McKay: A Method for Registration of 3-D Shapes. TPAMI, 1992.
        [2] https://en.wikipedia.org/wiki/Iterative_closest_point
    """

    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    b, size_X, dim = Xt.shape

    if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
        raise ValueError(
            "Point sets X and Y have to have the same "
            + "number of batches and data dimensions."
        )

    if ((num_points_Y < Yt.shape[1]).any() or (num_points_X < Xt.shape[1]).any()) and (
            num_points_Y != num_points_X
    ).any():
        # we have a heterogeneous input (e.g. because X/Y is
        # an instance of Pointclouds)
        mask_X = (
                torch.arange(size_X, dtype=torch.int64, device=Xt.device)[None]
                < num_points_X[:, None]
        ).type_as(Xt)
    else:
        mask_X = Xt.new_ones(b, size_X)

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
        knn_res = knn_points(Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True)
        # closest point
        Xt_nn_points = knn_res.knn[:, :, 0, :]

        # get the alignment of the nearest neighbors from Yt with Xt_init
        sim_transform, rmse = corresponding_points_alignment_sgd(
            Xt_init,
            Xt_nn_points,
            knn_res,
            weights=mask_X,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection,
            R=R,
            T=T,
            s=s,
            iterations=sgd_iterations,
            lr=sgd_lr,
            pose_cost=pose_cost,
            learn_translation=learn_translation
        )
        R, T, s = sim_transform

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # use the reported RMSE which includes other losses
        # # compute the root mean squared error
        # Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)
        # rmse = oputil.wmean(Xt_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]

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


def corresponding_points_alignment_sgd(
        X: Union[torch.Tensor, "Pointclouds"],
        Y: Union[torch.Tensor, "Pointclouds"],
        knn_res: _KNN,
        weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
        estimate_scale: bool = False,
        allow_reflection: bool = False,
        eps: float = 1e-9,
        R: torch.Tensor = None, T: torch.tensor = None, s: torch.tensor = None,
        iterations: int = 50,
        lr: float = 0.001,
        pose_cost=None,
        learn_translation: bool = True,
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
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.
        **sgd_iterations**: Number of epochs to run
        **sgd_lr**: Learning rate

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.

    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = oputil.convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = strutil.list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
                torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
                < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = oputil.wmean(Xt, weight=weights, eps=eps)
    Ymu = oputil.wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    if R is None:
        R = random_rotations(b, dtype=Xt.dtype, device=Xt.device)
        T = torch.randn((b, dim), dtype=Xt.dtype, device=Xt.device)
        s = torch.ones(b, dtype=Xt.dtype, device=Xt.device)

    # convert to quaternions for non-redundant representation
    q = matrix_to_rotation_6d(R)

    # set them up as parameters for training
    q.requires_grad = True
    T.requires_grad = True
    if estimate_scale:
        s.requires_grad = True

    optimizer = torch.optim.Adam([q, T, s], lr=lr)
    loss = MSELoss(reduction='none')

    def get_usable_transform_representation():
        nonlocal T
        # we get a more usable representation of R
        R = rotation_6d_to_matrix(q)
        if not learn_translation:
            # we compute T based on the specifics of the problem
            if estimate_scale:
                # s is trained in this case
                T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
            else:
                T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]
        return R, T

    for epoch in range(iterations):
        R, T = get_usable_transform_representation()

        if learn_translation:
            yhat = _apply_similarity_transform(Xt, R, T, s)
            this_loss = loss(yhat, Yt)
        else:
            yhat = s[:, None, None] * torch.bmm(Xc, R)
            this_loss = loss(yhat, Yc)
        # since using reduction None, need to reduce it to per batch
        this_loss = this_loss.sum(dim=1).sum(dim=1)

        if pose_cost is not None:
            other_loss = pose_cost(knn_res, R, T, s)
            this_loss += other_loss
            # this_loss = other_loss

        this_loss.mean().backward()
        # visualize gradients on the losses
        pose_cost.visualize(R, T, s)

        optimizer.step()
        optimizer.zero_grad()

    print(f"rmse loss {this_loss.mean().item() - other_loss.mean().item()} pose loss {other_loss.mean().item()}")
    R, T = get_usable_transform_representation()
    return SimilarityTransform(R.detach().clone(), T.detach().clone(), s.detach().clone()), this_loss.detach().clone()
