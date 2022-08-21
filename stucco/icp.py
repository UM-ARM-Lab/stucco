import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import time

import pytorch_kinematics
from pytorch_kinematics.transforms import wxyz_to_xyzw, xyzw_to_wxyz

from pytorch3d.ops import iterative_closest_point
from pytorch3d.ops import knn_points
from pytorch3d.ops.points_alignment import SimilarityTransform
import pytorch3d.transforms as tf
from stucco.icp_sgd import iterative_closest_point_sgd
from stucco import util


# from https://github.com/richardos/icp
def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.
    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:
        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean) * (xp - xp_mean)
        s_y_yp += (y - y_mean) * (yp - yp_mean)
        s_x_yp += (x - x_mean) * (yp - yp_mean)
        s_y_xp += (y - y_mean) * (xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    translation_y = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """
    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.
    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs the should exist
    :param verbose: whether to print informative messages about the process (default: False)
    :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
             transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points


# from https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Mxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def nearest_neighbor_torch(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: bxMxm array of points
        dst: bxNxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    dist = torch.cdist(src, dst)
    knn = dist.topk(1, largest=False)
    return knn


def best_fit_transform_torch(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: bxNxm numpy array of corresponding points
      B: bxNxm numpy array of corresponding points
    Returns:
      T: bx(m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: bxmxm rotation matrix
      t: bxmx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[-1]
    b = A.shape[0]

    # translate points to their centroids
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B

    # Orthogonal Procrustes Problem
    # minimize E(R,t) = sum_{i,j} ||bb_i - Raa_j - t||^2
    # equivalent to minimizing ||BB - R AA||^2
    # rotation matrix
    H = AA.transpose(-1, -2) @ BB
    U, S, Vt = torch.svd(H)
    # assume H is full rank, then the minimizing R and t are unique
    R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)

    # special reflection case
    reflected = torch.det(R) < 0
    Vt[reflected, m - 1, :] *= -1
    R[reflected] = Vt[reflected].transpose(-1, -2) @ U[reflected].transpose(-1, -2)

    # translation
    t = centroid_B.transpose(-1, -2) - (R @ centroid_A.transpose(-1, -2))

    # homogeneous transformation
    T = torch.eye(m + 1, dtype=A.dtype, device=A.device).repeat(b, 1, 1)
    T[:, :m, :m] = R
    T[:, :m, m] = t.view(b, -1)

    return T, R, t


def icp_2(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Mxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def init_random_transform_with_given_init(m, batch, dtype, device, given_init_pose=None):
    # apply some random initial poses
    if m > 2:
        import pytorch_kinematics.transforms as tf
        R = tf.random_rotations(batch, dtype=dtype, device=device)
    else:
        theta = torch.rand(batch, dtype=dtype, device=device) * math.pi * 2
        Rtop = torch.cat([torch.cos(theta).view(-1, 1), -torch.sin(theta).view(-1, 1)], dim=1)
        Rbot = torch.cat([torch.sin(theta).view(-1, 1), torch.cos(theta).view(-1, 1)], dim=1)
        R = torch.cat((Rtop.unsqueeze(-1), Rbot.unsqueeze(-1)), dim=-1)

    init_pose = torch.eye(m + 1, dtype=dtype, device=device).repeat(batch, 1, 1)
    init_pose[:, :m, :m] = R[:, :m, :m]
    if given_init_pose is not None:
        # check if it's given as a batch
        if len(given_init_pose.shape) == 3:
            init_pose = given_init_pose.clone()
        else:
            init_pose[0] = given_init_pose
    return init_pose


def icp_3(A, B, A_normals=None, B_normals=None, normal_scale=0.1, given_init_pose=None, max_iterations=20,
          tolerance=1e-4, batch=5, vis=None):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Mxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        A_normals: Mxm numpy array of source mD surface normal vectors
        B_normals: Nxm numpy array of destination mD surface normal vectors
        given_init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = torch.ones((m + 1, A.shape[0]), dtype=A.dtype, device=A.device)
    dst = torch.ones((B.shape[0], m + 1), dtype=A.dtype, device=A.device)
    src[:m, :] = torch.clone(A.transpose(0, 1))
    dst[:, :m] = torch.clone(B)
    src = src.repeat(batch, 1, 1)
    dst = dst.repeat(batch, 1, 1)

    # apply some random initial poses
    init_pose = init_random_transform_with_given_init(m, batch, A.dtype, A.device, given_init_pose=given_init_pose)

    # apply the initial pose estimation
    src = init_pose @ src
    src_normals = A_normals if normal_scale > 0 else None
    dst_normals = B_normals if normal_scale > 0 else None
    if src_normals is not None and dst_normals is not None:
        # NOTE normally we need to multiply by the transpose of the inverse to transform normals, but since we are sure
        # the transform does not include scale, we can just use the matrix itself
        # NOTE normals have to be transformed in the opposite direction as points!
        src_normals = src_normals.repeat(batch, 1, 1) @ init_pose[:, :m, :m].transpose(-1, -2)
        dst_normals = dst_normals.repeat(batch, 1, 1)

    prev_error = 0
    err_list = []

    if vis is not None:
        for j in range(A.shape[0]):
            pt = src[0, :m, j]
            vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
            if src_normals is not None:
                vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)

    i = 0
    distances = None
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        # if given normals, scale and append them to find nearest neighbours
        p = src[:, :m, :].transpose(-2, -1)
        q = dst[:, :, :m]
        if src_normals is not None:
            p = torch.cat((p, src_normals * normal_scale), dim=-1)
            q = torch.cat((q, dst_normals * normal_scale), dim=-1)
        distances, indices = nearest_neighbor_torch(p, q)
        # currently only have a single batch so flatten
        distances = distances.view(batch, -1)
        indices = indices.view(batch, -1)

        fit_from = src[:, :m, :].transpose(-2, -1)
        to_fit = []
        for b in range(batch):
            to_fit.append(dst[b, indices[b], :m])
        to_fit = torch.stack(to_fit)
        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform_torch(fit_from, to_fit)

        # update the current source
        src = T @ src
        if src_normals is not None and dst_normals is not None:
            src_normals = src_normals @ T[:, :m, :m].transpose(-1, -2)

        if vis is not None:
            for j in range(A.shape[0]):
                pt = src[0, :m, j]
                vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
                if src_normals is not None:
                    vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)

        # check error
        mean_error = torch.mean(distances)
        err_list.append(mean_error.item())
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform_torch(A.repeat(batch, 1, 1), src[:, :m, :].transpose(-2, -1))

    if vis is not None:
        # final evaluation
        src = torch.ones((m + 1, A.shape[0]), dtype=A.dtype, device=A.device)
        src[:m, :] = torch.clone(A.transpose(0, 1))
        src = src.repeat(batch, 1, 1)
        src = T @ src
        p = src[:, :m, :].transpose(-2, -1)
        q = dst[:, :, :m]
        distances, indices = nearest_neighbor_torch(p, q)
        # currently only have a single batch so flatten
        distances = distances.view(batch, -1)
        mean_error = torch.mean(distances)
        err_list.append(mean_error.item())
        if src_normals is not None and dst_normals is not None:
            # NOTE normally we need to multiply by the transpose of the inverse to transform normals, but since we are sure
            # the transform does not include scale, we can just use the matrix itself
            src_normals = A_normals.repeat(batch, 1, 1) @ T[:, :m, :m].transpose(-1, -2)

        for j in range(A.shape[0]):
            pt = src[0, :m, j]
            vis.draw_point(f"impt.{j}", pt, color=(0, 1, 0), length=0.003)
            if src_normals is not None:
                vis.draw_2d_line(f"imn.{j}", pt, -src_normals[0, j], color=(0, 0.4, 0), size=2., scale=0.03)
        for dist in err_list:
            print(dist)

    # convert to RMSE
    if distances is not None:
        distances = torch.sqrt(distances.square().sum(dim=1))
    return T, distances, i


def icp_pytorch3d(A, B, given_init_pose=None, batch=30):
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    res = iterative_closest_point(A.repeat(batch, 1, 1), B.repeat(batch, 1, 1), init_transform=given_init_pose,
                                  allow_reflection=True)
    T = torch.eye(4).repeat(batch, 1, 1)
    T[:, :3, :3] = res.RTs.R.transpose(-1, -2)
    T[:, :3, 3] = res.RTs.T
    distances = res.rmse
    return T, distances


def icp_pytorch3d_sgd(A, B, given_init_pose=None, batch=30, **kwargs):
    # initialize transform with closed form solution
    # T, distances = icp_pytorch3d(A, B, given_init_pose=given_init_pose, batch=batch)
    # given_init_pose = T.inverse()

    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)
    given_init_pose = SimilarityTransform(given_init_pose[:, :3, :3],
                                          given_init_pose[:, :3, 3],
                                          torch.ones(batch, device=A.device, dtype=A.dtype))

    res = iterative_closest_point_sgd(A.repeat(batch, 1, 1), B.repeat(batch, 1, 1), init_transform=given_init_pose,
                                      allow_reflection=True, **kwargs)
    T = torch.eye(4).repeat(batch, 1, 1)
    T[:, :3, :3] = res.RTs.R
    T[:, :3, 3] = res.RTs.T
    distances = res.rmse
    return T, distances


# def icp_simpleicp(A, B, given_init_pose=None, batch=30):
#     from simpleicp import PointCloud, SimpleICP
#
#     Ts = []
#     for b in range(batch):
#         pc_fix = PointCloud(A.numpy(), columns=["x", "y", "z"])
#         pc_mov = PointCloud(B.numpy(), columns=["x", "y", "z"])
#         icp_4 = SimpleICP()
#         icp_4.add_point_clouds(pc_fix, pc_mov)
#         # upright prior
#         rbp_observed_values = (0., 0., np.random.rand() * 360, 0., 0., 0.)
#         rbp_observation_weights = (10., 10., 0., 0., 0., 0.)
#         T, X_mov_transformed, rigid_body_transformation_params = icp_4.run(max_overlap_distance=1,
#                                                                            rbp_observed_values=rbp_observed_values,
#                                                                            rbp_observation_weights=rbp_observation_weights)
#         Ts.append(T)
#     T = torch.from_numpy(np.stack(Ts))
#     distances = None
#     return T, distances


# Stein ICP from https://bitbucket.org/fafz/stein-icp/src/master/
def squared_distance(x1, x2):
    """
    Computes squared distance matrix between two arrays of row vectors.

    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res


def get_kernel(x, y, lengthscale=1.0):
    pairwise_dists = squared_distance(x / lengthscale, y / lengthscale)
    h_sq = 0.5 * torch.median(pairwise_dists) / math.log(x.shape[0])
    return torch.exp(-0.5 * pairwise_dists / h_sq)


def phi(X, score_func):
    """

    Code originally from:
        https://colab.research.google.com/github/activatedgeek/stein-gradient/blob/master/Stein.ipynb#scrollTo=2fPfeNqDrYgo
    """
    X = X.detach().requires_grad_(True)
    K_XX = get_kernel(X, X.detach())
    grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]
    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi


def normalize_clouds(source_torch, target_torch):
    """Normalizes the cloud coordinates to lie withing [0, 1]."""
    max_s = torch.max(torch.abs(source_torch))

    max_t = torch.max(torch.abs(target_torch))

    abs_max = torch.max(max_s, max_t)
    source_torch = source_torch / abs_max
    target_torch = target_torch / abs_max

    return source_torch, target_torch, abs_max


def initialize_particles(particle_count, device, from_, to_):
    b = 1
    x = (from_[0] - to_[0]) * torch.rand(particle_count, b, device=device) + to_[0]
    y = (from_[1] - to_[1]) * torch.rand(particle_count, b, device=device) + to_[1]
    z = (from_[2] - to_[2]) * torch.rand(particle_count, b, device=device) + to_[2]
    r = (from_[3] - to_[3]) * torch.rand(particle_count, b, device=device) + to_[3]
    p = (from_[4] - to_[4]) * torch.rand(particle_count, b, device=device) + to_[4]
    ya = (from_[5] - to_[5]) * torch.rand(particle_count, b, device=device) + to_[5]

    return x, y, z, r, p, ya


def initialize_particles_from_transform(T):
    x, y, z = T[:, :3, 3].transpose(0, 1)
    r, p, ya = tf.matrix_to_euler_angles(T[:, :3, :3], "XYZ").transpose(0, 1)
    return x, y, z, r, p, ya


def get_stein_pairs(
        source,  # particle_count x batchsize x 3
        transformed_source,  # particle_count x batchsize x 3
        target,  # particle_count x batchsize x 3
        max_dist  # threshold distance to reject matching pair
):
    matches = knn_points(transformed_source, target, None, None, K=1, return_nn=True)
    # Get the target cloud at the paired indices
    target_paired = matches.knn[:, :, 0, :]

    source = ignore_far_away_matches(source, matches.dists.squeeze(), max_dist)
    transformed_source = ignore_far_away_matches(transformed_source, matches.dists.squeeze(), max_dist)
    target_paired = ignore_far_away_matches(target_paired, matches.dists.squeeze(), max_dist)

    return (source, transformed_source, target_paired)


# @torch.jit.script
def ignore_far_away_matches(cloud, dist_norm, max_dist):
    cloud[dist_norm > max_dist] = 0
    return cloud


class SteinSGDICP:

    def __init__(
            self,
            source_cloud,
            target_cloud,
            x, y, z, r, p, ya,
            normalize_clouds=True,
            iterations=150,
            batch_size=150,
            max_dist=1.,
            optimizer_class="Adam",
            lr=0.01
    ):
        self.particle_size = x.shape[0]

        self.x = x.view(self.particle_size, 1, 1)
        self.y = y.view(self.particle_size, 1, 1)
        self.z = z.view(self.particle_size, 1, 1)
        self.r = r.view(self.particle_size, 1, 1)
        self.p = p.view(self.particle_size, 1, 1)
        self.ya = ya.view(self.particle_size, 1, 1)

        self.device = target_cloud.device

        self.iter = iterations
        self.batch_size = batch_size

        self.max_dist = max_dist

        self.optimizer_class = optimizer_class
        self.lr = lr

        self.source_cloud = source_cloud
        self.target_cloud = target_cloud
        self.normalize_clouds = normalize_clouds

        self.stein_v_grads = None

    def vectorized_rotation(self):
        # r, p and ya are particle sizx1x1 column tensors where n is the number of particles
        # output is particle sizex3x3

        T = torch.stack(
            [torch.stack(
                [torch.cos(self.p) * torch.cos(self.ya),
                 (torch.sin(self.r) * torch.sin(self.p) * torch.cos(self.ya) - torch.cos(self.r) * torch.sin(self.ya)),
                 (torch.sin(self.r) * torch.sin(self.ya) + torch.cos(self.r) * torch.sin(self.p) * torch.cos(self.ya))
                 ], dim=3).squeeze(1),
             torch.stack(
                 [torch.cos(self.p) * torch.sin(self.ya),
                  (torch.cos(self.r) * torch.cos(self.ya) + torch.sin(self.r) * torch.sin(self.p) * torch.sin(self.ya)),
                  (torch.cos(self.r) * torch.sin(self.p) * torch.sin(self.ya) - torch.sin(self.r) * torch.cos(self.ya))
                  ], dim=3).squeeze(1),
             torch.stack(
                 [-torch.sin(self.p),
                  torch.sin(self.r) * torch.cos(self.p),
                  torch.cos(self.r) * torch.cos(self.p)
                  ], dim=3).squeeze(1)],
            dim=2).squeeze()

        return T

    def vectorized_trans(self):
        """Returns a tensor containing the translation parameters.

        Returns:
            Returns a Nx3 tensor containing the translation parameters
        """
        return torch.cat([self.x, self.y, self.z], dim=1)

    def partial_derivatives_stein(self):
        A = torch.cos(self.ya)
        B = torch.sin(self.ya)
        C = torch.cos(self.p)
        D = torch.sin(self.p)
        E = torch.cos(self.r)
        F = torch.sin(self.r)

        DE = D * E
        DF = D * F
        AC = A * C
        AF = A * F
        AE = A * E
        ADE = A * DE

        ADF = A * DF
        BC = B * C
        BE = B * E
        BF = B * F
        BDE = B * DE

        torch_0 = torch.zeros(self.particle_size, 1, 1, device=self.device)
        partial_roll = torch.stack([
            torch.stack([torch_0, ADE + BF, BE - ADF], dim=3).squeeze(1),
            torch.stack([torch_0, -AF + BDE, B * -DF - AE], dim=3).squeeze(1),
            torch.stack([torch_0, C * E, C * -F], dim=3).squeeze(1)
        ], dim=2).squeeze()

        partial_pitch = torch.stack([
            torch.stack([A * -D, AC * F, AC * E], dim=3).squeeze(1),
            torch.stack([B * -D, BC * F, BC * E], dim=3).squeeze(1),
            torch.stack([-C, -DF, -DE], dim=3).squeeze(1)
        ], dim=2).squeeze()

        partial_yaw = torch.stack([
            torch.stack([-BC, -B * DF - AE, AF - BDE], dim=3).squeeze(1),
            torch.stack([AC, -BE + ADF, ADE + BF], dim=3).squeeze(1),
            torch.stack([torch_0, torch_0, torch_0], dim=3).squeeze(1)
        ], dim=2).squeeze()

        return partial_roll, partial_pitch, partial_yaw

    def stein_grads(
            self,
            source_resized,  # each cloud here is particle_size x batchsize x 3
            transformed_paired,
            target_paired,
            scaling_factor  # scaling with point size as required by stein formulation.
    ):
        # Get the number of dummy points (zero points inserted during filtering) to get
        # mean by excluding their counts
        zero_points_size = (transformed_paired.sum(dim=2) == 0).sum(dim=1)
        partial_derivatives = self.partial_derivatives_stein()

        error = transformed_paired - target_paired

        input_ = torch.empty(self.particle_size, 6, device=self.device)  # particle_count  x 6
        gradient_cost = torch.zeros_like(input_)

        # For xyz gradients, get sum of error in xyz dim independently for each particle
        # sumerr = error.sum(dim=1)

        # Zero matching will make grad to nan since currently we are not monitoring low
        # matching pairs. So set it to 1 if we have number of zero=batch size (i.e. non matching pair)
        a = torch.zeros_like(zero_points_size)
        # set a to 1 where dummy points == batch size
        a[(torch.where(zero_points_size == self.batch_size))] = 1

        gradient_cost[:, 0:3] = error.sum(dim=1) / (
                (self.batch_size - zero_points_size.view(self.particle_size, 1)) + a.view(self.particle_size,
                                                                                          1))  # divide by no. of points after excluding dummy points
        gradient_cost[:, 3] = torch.einsum('pbc,pbc->pb', [error, (
            torch.einsum('prc,pbc->pbr', [(partial_derivatives)[0], source_resized]))]).sum(dim=1) / ((
                                                                                                              self.batch_size - zero_points_size) + a)  # .view(self.particle_size,1).squeeze())
        gradient_cost[:, 4] = torch.einsum('pbc,pbc->pb', [error, (
            torch.einsum('prc,pbc->pbr', [(partial_derivatives)[1], source_resized]))]).sum(dim=1) / ((
                                                                                                              self.batch_size - zero_points_size) + a)  # .view(self.particle_size,1).squeeze())
        gradient_cost[:, 5] = torch.einsum('pbc,pbc->pb', [error, (
            torch.einsum('prc,pbc->pbr', [(partial_derivatives)[2], source_resized]))]).sum(dim=1) / ((
                                                                                                              self.batch_size - zero_points_size) + a)  # .view(self.particle_size,1).squeeze())

        return gradient_cost * scaling_factor

    def stein_align(self):
        abs_max = 1
        s_tor = self.source_cloud
        t_tor = self.target_cloud
        gradient_scaling_factor = s_tor.shape[0]

        if self.normalize_clouds:
            s_tor, t_tor, abs_max = normalize_clouds(self.source_cloud, self.target_cloud)
            self.x = self.x / abs_max
            self.y = self.y / abs_max
            self.z = self.z / abs_max
            # If we normalize cloud, we need to scale gradients with normalizing factor to be stable
            gradient_scaling_factor = abs_max * s_tor.shape[0]

        def batch_generator():
            while True:
                indexes = np.random.randint(0, len(s_tor), self.batch_size)
                yield s_tor[indexes]

        mini_batch_sampler = batch_generator()

        parameters = [self.x, self.y, self.z, self.r, self.p, self.ya]
        optimizer = None
        print("Optimizing using: ", end="")
        if self.optimizer_class == "Adam":
            print("Adam")
            optimizer = torch.optim.Adam(parameters, lr=self.lr)
        elif self.optimizer_class == "RMSprop":
            print("Prop")
            optimizer = torch.optim.RMSprop(
                parameters,
                lr=self.lr,
                weight_decay=1e-8,
                momentum=0.9
            )
        elif self.optimizer_class == "SGD":
            print("SGD")
            optimizer = torch.optim.SGD(parameters, lr=self.lr)
        elif self.optimizer_class == "Adagrad":
            print("Adagrad")
            optimizer = torch.optim.Adagrad(parameters, lr=self.lr)
        else:
            raise NotImplementedError

        max_dist = self.max_dist / abs_max

        t_start = time.time()

        for epoch in range(self.iter):
            # Get a minibatch

            minibatch1 = next(mini_batch_sampler)

            minibatch = minibatch1.expand(self.particle_size, self.batch_size, 3).clone()

            trans_mat = self.vectorized_trans()
            rot_mat = self.vectorized_rotation()

            s_transformed = torch.einsum(
                "pmc,prc->pmr",
                [minibatch, rot_mat]
            ) + trans_mat.view(self.particle_size, 1, 3)

            source_resized, transformed_paired, target_paired = get_stein_pairs(
                minibatch,
                s_transformed,
                t_tor.expand(self.particle_size, t_tor.shape[0], 3).clone(),
                max_dist

            )

            mean_grads = self.stein_grads(
                source_resized,
                transformed_paired,
                target_paired,
                gradient_scaling_factor
            )

            stein_v_grads = phi(
                torch.stack([self.x, self.y, self.z, self.r, self.p, self.ya]).squeeze().T,
                -mean_grads
            )
            self.stein_v_grads = stein_v_grads
            #
            self.x.grad = -stein_v_grads[:, 0].view(self.particle_size, 1, 1).clone()
            self.y.grad = -stein_v_grads[:, 1].view(self.particle_size, 1, 1).clone()
            self.z.grad = -stein_v_grads[:, 2].view(self.particle_size, 1, 1).clone()

            self.r.grad = -stein_v_grads[:, 3].view(self.particle_size, 1, 1).clone()
            self.p.grad = -stein_v_grads[:, 4].view(self.particle_size, 1, 1).clone()
            self.ya.grad = -stein_v_grads[:, 5].view(self.particle_size, 1, 1).clone()

            optimizer.step()
            optimizer.zero_grad()

        t_end = time.time()

        particle_pose = torch.stack([
            self.x * abs_max,
            self.y * abs_max,
            self.z * abs_max,
            self.r,
            self.p,
            self.ya
        ]).squeeze().T

        print("Stein ICP computation: {:.2f}s".format(t_end - t_start))
        return particle_pose


def plot_kde(x, y, z, roll, pitch, yaw, bwd, do_shading, max_dist):
    import matplotlib.pylab as plt
    import seaborn as sns
    colors = "rgbcmy"

    def plot_one_dimension(data, label, color):
        sns.kdeplot(
            data.squeeze().cpu().detach().numpy(),
            label=label,
            color=color,
            bw_adjust=bwd,
            shade=do_shading
        )

    _, ax = plt.subplots(1)
    plot_one_dimension(x, "X", colors[0])
    plot_one_dimension(y, "Y", colors[1])
    plot_one_dimension(z, "Z", colors[2])
    plot_one_dimension(roll, "Roll", colors[3])
    plot_one_dimension(pitch, "Pitch", colors[4])
    plot_one_dimension(yaw, "Yaw", colors[5])
    ax.legend()


def plot_samples(x, y, z, roll, pitch, yaw):
    import matplotlib.pylab as plt
    colors = "rgbcmy"

    def plot_one_dimension(data, label, color):
        plt.plot(
            data.squeeze().cpu().detach().numpy(),
            label=label,
            color=color,
            linestyle="none",
            marker="o"
        )

    _, ax = plt.subplots(1)
    plot_one_dimension(x, "X", colors[0])
    plot_one_dimension(y, "Y", colors[1])
    plot_one_dimension(z, "Z", colors[2])
    plot_one_dimension(roll, "Roll", colors[3])
    plot_one_dimension(pitch, "Pitch", colors[4])
    plot_one_dimension(yaw, "Yaw", colors[5])
    plt.xlabel("Particle #")
    plt.ylabel("Parameter Value")
    ax.legend()


def do_plot(particle_poses, max_dist):
    import matplotlib.pylab as plt
    plot_kde(
        particle_poses[:, 0],
        particle_poses[:, 1],
        particle_poses[:, 2],
        particle_poses[:, 3],
        particle_poses[:, 4],
        particle_poses[:, 5],
        0.0254,
        True,
        max_dist
    )

    plot_samples(
        particle_poses[:, 0],
        particle_poses[:, 1],
        particle_poses[:, 2],
        particle_poses[:, 3],
        particle_poses[:, 4],
        particle_poses[:, 5]
    )
    plt.show()


def icp_stein(A, B, given_init_pose=None, batch=30, max_dist=0.8):
    # batch becomes particles
    device = A.device

    # initialize pose guess from given_init_pose
    if given_init_pose is not None:
        pose_guess = initialize_particles_from_transform(given_init_pose.inverse())
    else:
        from_ = torch.tensor([-0.02, -0.02, -0.1, 0., 0., -math.pi]).to(device)
        to_ = torch.tensor([0.02, 0.02, -0.2, 0., 0., math.pi]).to(device)
        pose_guess = initialize_particles(batch, device, from_, to_)

    # do_plot(torch.stack(pose_guess).transpose(0, 1), max_dist)

    stein = SteinSGDICP(
        A,
        B,
        pose_guess[0], pose_guess[1], pose_guess[2], pose_guess[3], pose_guess[4], pose_guess[5],
        max_dist=max_dist,
        optimizer_class="Adam",
        iterations=50,
        lr=0.01,
        batch_size=50,
        normalize_clouds=False
    )
    particle_poses = stein.stein_align()
    R = stein.vectorized_rotation()
    t = stein.vectorized_trans()

    # do_plot(particle_poses, max_dist)

    elems = ["x", "y", "z", "r", "p", "ya"]
    for i, elem in enumerate(elems):
        print(f"{elem}: {particle_poses[:, i].mean()} ({particle_poses[:, i].std()})")

    T = torch.eye(4).repeat(batch, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.squeeze(-1)
    distances = stein.stein_v_grads.abs().sum(dim=1)
    return T, distances


def icp_mpc(A, B, cost_func, given_init_pose=None, batch=30, horizon=10, num_samples=100,
            max_rot_mag=0.01, max_trans_mag=0.01, iterations=10,
            rot_sigma=0.001, trans_sigma=0.05, draw_mesh=None):
    from pytorch_mppi import mppi
    from pytorch_kinematics.transforms import rotation_6d_to_matrix
    # use the given cost function and run MPC with deltas on the transforms to optimize it
    given_init_pose = init_random_transform_with_given_init(A.shape[1], batch, A.dtype, A.device,
                                                            given_init_pose=given_init_pose)

    d = A.device
    nx = 16  # 4 x 4 transformation matrix
    # TODO use axis angle representation for delta rotation
    nu_r = 6
    nu_t = 3
    noise_sigma = torch.diag(torch.tensor([rot_sigma] * nu_r + [trans_sigma] * nu_t, device=d))

    def dynamics(transform, delta_transform):
        # convert flattened delta to its separate 6D rotations and 3D translations
        N = transform.shape[0]
        dH = torch.eye(4, dtype=transform.dtype, device=d).repeat(N, 1, 1)
        dH[:, :3, :3] = rotation_6d_to_matrix(delta_transform[:, :6])
        dH[:, :3, 3] = delta_transform[:, 6:]
        return (dH @ transform.view(dH.shape)).view(N, nx)

    ctrl = mppi.MPPI(dynamics, cost_func, nx, noise_sigma, num_samples=num_samples, horizon=horizon,
                     device=d,
                     u_min=torch.tensor([-max_rot_mag] * nu_r + [-max_trans_mag] * nu_t, device=d),
                     u_max=torch.tensor([max_rot_mag] * nu_r + [max_trans_mag] * nu_t, device=d))

    visual_obj_id_map = {}
    obs = given_init_pose[0].inverse().contiguous()
    for i in range(iterations):
        delta_H = ctrl.command(obs.view(-1))
        rollout = ctrl.get_rollouts(obs.view(-1)).squeeze()
        # visualize current state after each MPC step
        if draw_mesh is not None:
            pos, rot = util.matrix_to_pos_rot(obs.view(4, 4))
            id = visual_obj_id_map.get(i, None)
            visual_obj_id_map[i] = draw_mesh("state", (pos, rot), (0., i / iterations, i / iterations, 0.5),
                                             object_id=id)
            # visualize rollout
            for j, x in enumerate(rollout):
                pos, rot = util.matrix_to_pos_rot(x.view(4, 4))
                internal_id = (j + 1) * iterations
                id = visual_obj_id_map.get(internal_id, None)
                visual_obj_id_map[internal_id] = draw_mesh("rollout", (pos, rot), (
                (j + 1) / (len(rollout) + 1), i / iterations, i / iterations, 0.5),
                                                           object_id=id)

        obs = dynamics(obs.unsqueeze(0), delta_H.unsqueeze(0))

    # TODO can't really do anything with the batch size?
    obs = obs.view(4, 4).repeat(batch, 1, 1)
    return obs, cost_func(obs)
