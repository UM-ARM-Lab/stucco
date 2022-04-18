import argparse
import enum
import time
import math

import matplotlib
import torch
import pybullet as p
import numpy as np
import logging
import os
from datetime import datetime

import gpytorch
from matplotlib import cm
from matplotlib import pyplot as plt
import pymeshlab

from stucco.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from stucco.defines import NO_CONTACT_ID
from stucco.env.env import InfoKeys

from arm_pytorch_utilities import rand, tensor_utils, math_utils
from arm_pytorch_utilities.grad import jacobian

from stucco import cfg
from stucco import icp, tracking
from stucco.env import arm
from stucco.env.arm import Levels
from stucco.env_getters.arm import RetrievalGetter
from stucco.env.pybullet_env import state_action_color_pairs

from stucco.retrieval_controller import rot_2d_mat_to_angle, \
    sample_model_points, pose_error, TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, KeyboardController, PHDFilterTrackingMethod

from torchmcubes import marching_cubes, grid_interp
from stucco.env.pybullet_env import make_box, DebugDrawer, closest_point_on_surface, surface_normal_at_point, \
    ContactInfo, make_sphere
import pybullet_data
from pybullet_object_models import ycb_objects
import pytorch_kinematics.transforms as tf

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def test_icp(target_obj_id, vis):
    # z = env._observe_ee(return_z=True)[-1]
    # # test ICP using fixed set of points
    # o = p.getBasePositionAndOrientation(env.target_object_id)[0]
    # contact_points = np.stack([
    #     [o[0] - 0.045, o[1] - 0.05],
    #     [o[0] - 0.05, o[1] - 0.01],
    #     [o[0] - 0.045, o[1] + 0.02],
    #     [o[0] - 0.045, o[1] + 0.04],
    #     [o[0] - 0.01, o[1] + 0.05]
    # ])
    # actions = np.stack([
    #     [0.7, -0.7],
    #     [0.9, 0.2],
    #     [0.8, 0],
    #     [0.5, 0.6],
    #     [0, -0.8]
    # ])
    # contact_points = np.stack(contact_points)
    #
    # angle = 0.5
    # dx = -0.4
    # dy = 0.2
    # c, s = math.cos(angle), math.sin(angle)
    # rot = np.array([[c, -s],
    #                 [s, c]])
    # contact_points = np.dot(contact_points, rot.T)
    # contact_points[:, 0] += dx
    # contact_points[:, 1] += dy
    # actions = np.dot(actions, rot.T)
    #
    # state_c, action_c = state_action_color_pairs[0]
    # env.visualize_state_actions("fixed", contact_points, actions, state_c, action_c, 0.05)

    # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
    model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=100, force_z=None, mid_z=0.05,
                                                         seed=0, clean_cache=False, random_sample_sigma=0.2,
                                                         name="mustard_normal", vis=vis, restricted_points=(
            [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))

    device, dtype = model_points.device, model_points.dtype
    pose = p.getBasePositionAndOrientation(target_obj_id)
    link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points = link_to_current_tf.transform_points(model_points)
    model_normals = link_to_current_tf.transform_normals(model_normals)

    for i, pt in enumerate(model_points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -model_normals[i], color=(0, 0, 0), size=2., scale=0.03)

    # perform ICP and visualize the transformed points
    # history, transformed_contact_points = icp.icp(model_points[:, :2], contact_points,
    #                                               point_pairs_threshold=len(contact_points), verbose=True)

    # better to have few A than few B and then invert the transform
    # T, distances, i = icp.icp_2(contact_points, model_points[:, :2])
    # transformed_contact_points = np.dot(np.c_[contact_points, np.ones((contact_points.shape[0], 1))], T.T)
    # T, distances, i = icp.icp_2(model_points[:, :2], contact_points)
    # transformed_model_points = np.dot(np.c_[model_points[:, :2], np.ones((model_points.shape[0], 1))],
    #                                   np.linalg.inv(T).T)
    # for i, pt in enumerate(transformed_model_points):
    #     pt = [pt[0], pt[1], z]
    #     env.vis.draw_point(f"tmpt.{i}", pt, color=(0, 1, 0), length=0.003)


def franke(X, Y):
    term1 = .75 * torch.exp(-((9 * X - 2).pow(2) + (9 * Y - 2).pow(2)) / 4)
    term2 = .75 * torch.exp(-((9 * X + 1).pow(2)) / 49 - (9 * Y + 1) / 10)
    term3 = .5 * torch.exp(-((9 * X - 7).pow(2) + (9 * Y - 3).pow(2)) / 4)
    term4 = .2 * torch.exp(-(9 * X - 4).pow(2) - (9 * Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2 * (9 * X - 2) * 9 / 4 * term1 - 2 * (9 * X + 1) * 9 / 49 * term2 + \
          -2 * (9 * X - 7) * 9 / 4 * term3 + 2 * (9 * X - 4) * 9 * term4
    dfy = -2 * (9 * Y - 2) * 9 / 4 * term1 - 9 / 10 * term2 + \
          -2 * (9 * Y - 3) * 9 / 4 * term3 + 2 * (9 * Y - 7) * 9 * term4

    return f, dfx, dfy


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=train_y.shape[1] - 1)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def project_to_plane(n, v):
    """Project vector v onto plane defined by its normal vector n"""
    # ensure unit vector
    nhat = n / n.norm(p=2)
    along_n = nhat @ v
    return v - along_n * nhat


from mesh_to_sdf import mesh_to_voxels
import trimesh
import skimage


def create_sdf(path):
    mesh = trimesh.load(path)
    voxels = mesh_to_voxels(mesh, 64, pad=True)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()
    return voxels


def sample_dx_on_tange_plane(n, alpha, num_samples=100):
    v0 = n.clone()
    v0[0] += 1
    v1 = torch.cross(n, v0)
    v2 = torch.cross(n, v1)
    v1 /= v1.norm()
    v2 /= v2.norm()
    angles = torch.linspace(0, math.pi * 2, num_samples)
    dx_samples = (torch.cos(angles).view(-1, 1) * v1 + torch.sin(angles).view(-1, 1) * v2) * alpha
    return dx_samples


class PlotPointType(enum.Enum):
    NONE = 0
    ICP_MODEL_POINTS = 1
    ERROR_AT_MODEL_POINTS = 2
    ERROR_AT_GP_SURFACE = 3


def test_existing_method_3d(gpscale=5, alpha=0.01, timesteps=202, training_iter=50, verify_numerical_gradients=False,
                            mesh_surface_alpha=1., build_model=False, clean_cache=False,
                            icp_period=5, approximate_uncertainty_with_icp_error_variance=False,
                            eval_period=10, plot_per_eval_period=1, model_name="mustard_normal",
                            plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS, run_name="icp_dist_debug"):
    extrude_objects_in_z = False
    z = 0.1
    h = 2 if extrude_objects_in_z else 0.15

    physics_client = p.connect(p.GUI)  # p.GUI for GUI or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    dd = DebugDrawer(0.8, 0.8)
    dd.toggle_3d(True)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -10)

    planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    # objId = make_box([0.4, 0.15, h], [-0.4, 0, z], [0, 0, -np.pi / 2])
    # objId = make_sphere(h, [0., 0, z])
    # ranges = np.array([[-.2, .2], [-.2, .2], [-.1, .4]])

    objId = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"),
                       [0., 0., z * 3],
                       p.getQuaternionFromEuler([0, 0, -1]), globalScaling=2.5)
    ranges = np.array([[-.2, .2], [-.2, .2], [0, .5]])

    # sdf = create_sdf(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', 'collision_vhacd.obj'))

    for _ in range(1000):
        p.stepSimulation()

    target_obj_id = objId
    vis = dd
    name = f"{model_name} {run_name}".strip()

    # these are in object frame (aligned with [0,0,0], [0,0,0,1]
    model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=100, force_z=None, mid_z=0.05,
                                                         seed=0, clean_cache=build_model, random_sample_sigma=0.2,
                                                         name=model_name, vis=vis, restricted_points=(
            [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))

    device, dtype = model_points.device, model_points.dtype
    pose = p.getBasePositionAndOrientation(target_obj_id)
    link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points_world_transformed_ground_truth = link_to_current_tf.transform_points(model_points)
    model_normals_world_transformed_ground_truth = link_to_current_tf.transform_normals(model_normals)
    if build_model:
        for i, pt in enumerate(model_points_world_transformed_ground_truth):
            vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
            vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_transformed_ground_truth[i], color=(0, 0, 0), size=2.,
                             scale=0.03)

    # start at a point on the surface of the bottle
    randseed = rand.seed(0)
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "{}_{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                                      randseed))
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
    else:
        cache = {name: {}}

    x = torch.tensor(closest_point_on_surface(objId, np.random.rand(3))[ContactInfo.POS_A])
    dd.draw_point('x', x, height=x[2])
    n = -torch.tensor(surface_normal_at_point(objId, x))
    dd.draw_2d_line('n', x, n, (0, 0.5, 0), scale=0.2)

    xs = [x]
    df = [n]
    # augmented x and df
    aug_xs = None
    aug_df = None
    likelihood = None
    model = None
    meshId = None
    # error data
    error_t = []
    error_at_model_points = []
    error_at_gp_surface = []
    # ICP data
    best_tsf_guess = None

    for t in range(timesteps):
        train_xs = xs
        train_df = df
        if aug_xs is not None:
            train_xs = torch.cat([torch.stack(xs), aug_xs.view(-1, 3)], dim=0)
            train_df = torch.cat([torch.stack(df), aug_df.view(-1, 3)], dim=0)
        likelihood, model = fit_gpis(train_xs, train_df, threedimensional=True, use_df=True, scale=gpscale,
                                     training_iter=training_iter, likelihood=likelihood, model=model)

        def gp(cx):
            pred = likelihood(model(cx * gpscale))
            return pred

        def gp_var(cx):
            return gp(cx).variance

        if verify_numerical_gradients:
            with rand.SavedRNG():
                # try numerically computing the gradient along valid directions only
                dx_samples = sample_dx_on_tange_plane(n, alpha)
                new_x_samples = x + dx_samples
                with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False,
                                                                          covar_root_decomposition=False):
                    pred_samples = gp(new_x_samples)
                    # select dx corresponding to largest var in first dimension
                    sample_ordering = torch.argsort(pred_samples.variance[:, 0], descending=True)
                    # sample_ordering = torch.argsort(pred_samples.variance, descending=True)
                dx_numerical = dx_samples[sample_ordering[0]]
                for k in range(10):
                    strength = (10 - k) / 10
                    dd.draw_2d_line(f'dx{k}', x, dx_samples[sample_ordering[k]], (strength, 1 - strength, 1 - strength),
                                    scale=3)

        if t > 5 and approximate_uncertainty_with_icp_error_variance:
            # query for next place to go based on approximated uncertainty using ICP error variance
            with rand.SavedRNG():
                N = 100
                B = 30
                dx_samples = sample_dx_on_tange_plane(n, alpha, num_samples=N)
                new_x_samples = x + dx_samples
                # do ICP
                this_pts = torch.stack(xs).reshape(-1, 3)
                T, distances, _ = icp.icp_3(this_pts, model_points,
                                            given_init_pose=best_tsf_guess, batch=B)
                # T = T.inverse()
                # link_to_current_tf = tf.Transform3d(matrix=T)
                # all_normals = link_to_current_tf.transform_normals(model_normals)
                point_tf_to_link = tf.Transform3d(matrix=T)
                all_points = point_tf_to_link.transform_points(new_x_samples)

                # compute ICP error for new sampled points
                query_icp_error = torch.zeros(B, N)
                # TODO consider using cached SDF
                for b in range(B):
                    for i in range(N):
                        closest = closest_point_on_surface(objId, all_points[b, i])
                        query_icp_error[b, i] = closest[ContactInfo.DISTANCE]
                # find error variance for each sampled dx
                icp_error_var = query_icp_error.var(dim=1)
                most_informative_idx = icp_error_var.argmax()
                # choose gradient based on this (basically throw out GP calculations)
                dx = dx_samples[most_informative_idx]
        else:
            # query for next place to go based on gradient of variance
            gp_var_jacobian = jacobian(gp_var, x)
            gp_var_grad = gp_var_jacobian[0]  # first dimension corresponds to SDF, other 2 are for the SDF gradient
            # choose random direction if it's the 0 vector
            if torch.allclose(gp_var_grad, torch.zeros_like(gp_var_grad)):
                gp_var_grad = torch.rand_like(gp_var_grad)
            dx = project_to_plane(n, gp_var_grad)

        # normalize to be magnitude alpha
        dx = dx / dx.norm() * alpha
        dd.draw_2d_line('a', x, dx, (1, 0., 0), scale=1)

        new_x = x + dx
        # project onto object (via low level controller applying a force)
        new_x = torch.tensor(closest_point_on_surface(objId, new_x)[ContactInfo.POS_A])

        dd.draw_transition(x, new_x)
        x = new_x
        dd.draw_point('x', x, height=x[2])
        n = -torch.tensor(surface_normal_at_point(objId, x))
        dd.draw_2d_line('n', x, n, (0, 0.5, 0), scale=0.2)

        xs.append(x)
        df.append(n)

        # augment data with ICP shape pseudo-observations
        if t > 0 and t % icp_period == 0:
            print('conditioning exploration on ICP results')
            # perform ICP and visualize the transformed points
            this_pts = torch.stack(xs).reshape(-1, 3)
            T, distances, _ = icp.icp_3(this_pts, model_points,
                                        given_init_pose=best_tsf_guess, batch=30)
            T = T.inverse()
            link_to_current_tf = tf.Transform3d(matrix=T)
            all_points = link_to_current_tf.transform_points(model_points)
            all_normals = link_to_current_tf.transform_normals(model_normals)

            # TODO can score on surface normals being consistent with robot path? Or switch to ICP with normal support
            score = score_icp(all_points, all_normals, distances).numpy()
            best_tsf_index = np.argmin(score)
            best_tsf_guess = T[best_tsf_index].inverse()

            if plot_point_type is PlotPointType.ICP_MODEL_POINTS:
                color = (1, 0, 0)
                for i, pt in enumerate(all_points[best_tsf_index]):
                    vis.draw_point(f"best.{i}", pt, color=color, length=0.003)

            # for j in range(len(T)):
            #     if j == best_tsf_index:
            #         continue
            #     color = (0, 1, 0)
            #     for i, pt in enumerate(all_points[j]):
            #         if i % 2 == 0:
            #             continue
            #         vis.draw_point(f"tmpt.{j}.{i}", pt, color=color, length=0.003)

            # # consider variance across the closest point (how consistent is this point with any contact point
            # B, N, d = all_points.shape
            # ap = all_points.reshape(B * N, d)
            # dists = torch.cdist(ap, ap)
            # # from every point of every batch to every point of every batch
            # dists = dists.reshape(B, N, B, N)
            # brange = torch.arange(B)
            # # nrange = torch.arange(N)
            # # avoid considering intercluster distance
            # # dists[b][n][b] should all be max dist (all distance inside cluster treated as max)
            # # min distance of point b,n to bth sample; the index will allow us to query the actual min points
            # # idx [b][n][i] will get us the index into all_points[b]
            # # to return the closest point to point b,n from sample i
            # min_dist_for_each_sample, min_dist_idx = dists.min(dim=-1)
            #
            # min_dist_idx = min_dist_idx.unsqueeze(2).repeat(1, 1, B, 1)
            #
            # # ind = np.indices(min_dist_idx.shape)
            # # ind[-1] = min_dist_idx.numpy()
            # # min_pts_for_each_sample = all_points.unsqueeze(2).repeat(1, 1, N, 1)[tuple(ind)]
            #
            # # for each b,n point, retrieve the min points of every other b sample
            # # need to insert the index into all_points[b] since it wouldn't know that otherwise
            # # batch_idx = torch.arange(B).reshape(B, 1, 1).repeat(1, N, B)
            # # batch_idx = torch.arange(B).unsqueeze(-1).unsqueeze(-1).repeat_interleave(N, dim=-2).repeat_interleave(B, dim=-1)
            # batch_idx = torch.arange(B).repeat(B, N, 1)
            # min_pts_for_each_sample = all_points[batch_idx, min_dist_idx]
            #
            # # each point b,n will have 1 that is equal to mask weight, so need to remove that
            # ignore_self_mask = torch.ones(B, N, B, dtype=torch.bool, device=dists.device)
            # ignore_self_mask[brange, :, brange] = False
            # min_dist_idx = min_dist_idx[ignore_self_mask].reshape(B, N, B - 1)
            #
            # # for each b,n point, evaluate variance across the B - 1 non-self batch closest points
            # vars = torch.var(min_pts_for_each_sample, dim=-2)
            # # sum variance across dimensions of point
            # vars = vars.sum(dim=1)
            #
            # # vars_cutoff = torch.quantile(vars, .25)
            # vars_cutoff = 0.03
            # ok_to_aug = vars < vars_cutoff
            # print(
            #     f"vars avg {vars.mean()} 25th percentile {torch.quantile(vars, .25)} cutoff {vars_cutoff} num passed {ok_to_aug.sum()}")
            #
            # # augment points and normals
            # if torch.any(ok_to_aug):
            #     transformed_model_points = all_points[best_tsf_index][ok_to_aug]
            #     aug_xs = transformed_model_points
            #     aug_df = all_normals[best_tsf_index][ok_to_aug]
            #     # visualize points that are below some variance threshold to be added
            #     for i, pt in enumerate(transformed_model_points):
            #         vis.draw_point(f"aug.{i}", pt, color=(1, 1, 0), length=0.015)
            #         vis.draw_2d_line(f"augn.{i}", pt, aug_df[i], color=(0, 0, 0), scale=0.05)
            #     vis.clear_visualization_after("aug", i + 1)
            #     vis.clear_visualization_after("augn", i + 1)
            # else:
            #     aug_xs = None
            #     aug_df = None
            #     vis.clear_visualization_after("aug", 0)
            #     vis.clear_visualization_after("augn", 0)

            # TODO consider evaluating the fit GP without augmented data (so only use augmented data to guide sliding)

        # after a period of time evaluate current level set
        if t > 0 and t % eval_period == 0:
            print('evaluating shape ' + str(t))
            # see what's the range of values we've actually traversed
            xx = torch.stack(xs)
            print(f'ranges: {torch.min(xx, dim=0)} - {torch.max(xx, dim=0)}')
            # n1, n2, n3 = 80, 80, 50
            num = [40, 40, 30]
            xv, yv, zv = torch.meshgrid(
                [torch.linspace(*ranges[0], num[0]), torch.linspace(*ranges[1], num[1]),
                 torch.linspace(*ranges[2], num[2])])

            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
                test_x = torch.stack(
                    [xv.reshape(np.prod(num), 1), yv.reshape(np.prod(num), 1), zv.reshape(np.prod(num), 1)],
                    -1).squeeze(1)
                PER_BATCH = 256
                vars = []
                us = []
                for i in range(0, test_x.shape[0], PER_BATCH):
                    predictions = gp(test_x[i:i + PER_BATCH])
                    mean = predictions.mean
                    var = predictions.variance

                    vars.append(var[:, 0])
                    us.append(mean[:, 0])

                var = torch.cat(vars).contiguous()
                imprint_norm = matplotlib.colors.Normalize(vmin=0, vmax=torch.quantile(var, .90))
                color_map = matplotlib.cm.ScalarMappable(norm=imprint_norm)

                u = torch.cat(us).reshape(*num).contiguous()

                verts, faces = marching_cubes(u, 0.0)

                # re-get the colour at the vertices instead of grid interpolation since that introduces artifacts
                # output of vertices need to be converted back to original space
                verts_xyz = verts.clone()
                verts_xyz[:, 0] = verts[:, 2]
                verts_xyz[:, 2] = verts[:, 0]
                for dim in range(ranges.shape[0]):
                    verts_xyz[:, dim] /= num[dim] - 1
                    verts_xyz[:, dim] = verts_xyz[:, dim] * (ranges[dim][1] - ranges[dim][0]) + ranges[dim][0]

                # plot mesh
                if t > 0 and t % (eval_period * plot_per_eval_period) == 0:
                    vars = []
                    for i in range(0, verts.shape[0], PER_BATCH):
                        predictions = gp(verts_xyz[i:i + PER_BATCH])
                        var = predictions.variance
                        vars.append(var[:, 0])
                    var = torch.cat(vars).contiguous()

                    faces = faces.cpu().numpy()
                    colrs = color_map.to_rgba(var.reshape(-1))
                    # note, can control alpha on last column here
                    colrs[:, -1] = mesh_surface_alpha

                    # create and save mesh
                    m = pymeshlab.Mesh(verts_xyz, faces, v_color_matrix=colrs)
                    ms = pymeshlab.MeshSet()
                    ms.add_mesh(m, "level_set")
                    # UV map and turn vertex coloring into a texture
                    base_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{t}"
                    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
                    ms.compute_texmap_from_color(textname=f"tex_{base_name}")

                    fn = os.path.join(cfg.DATA_DIR, "shape_explore", f"mesh_{base_name}.obj")
                    ms.save_current_mesh(fn)

                    print('plotting mesh')

                    if meshId is not None:
                        p.removeBody(meshId)
                    visId = p.createVisualShape(p.GEOM_MESH, fileName=fn)
                    meshId = p.createMultiBody(0, baseVisualShapeIndex=visId, basePosition=[0, 0, 0])

                    # input('enter to clear visuals')
                    dd.clear_visualization_after('grad', 0)

                # evaluate surface error
                error_norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
                color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
                with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False,
                                                                          covar_root_decomposition=False):
                    predictions = gp(model_points_world_transformed_ground_truth)
                mean = predictions.mean
                err = torch.abs(mean[:, 0])
                if plot_point_type is PlotPointType.ERROR_AT_MODEL_POINTS:
                    rgb = color_map.to_rgba(err.reshape(-1))
                    rgb = torch.from_numpy(rgb[:, :-1]).to(dtype=u.dtype, device=u.device)
                    for i in range(model_points_world_transformed_ground_truth.shape[0]):
                        dd.draw_point(f'pt.{i}', model_points_world_transformed_ground_truth[i], rgb[i], length=0.002)
                    dd.clear_visualization_after("pt", verts.shape[0])
                error_at_model_points.append(err.mean())

                # evaluation of estimated SDF surface points on ground truth
                dists = []
                for vert in verts_xyz:
                    closest = closest_point_on_surface(objId, vert)
                    dists.append(closest[ContactInfo.DISTANCE])
                dists = torch.tensor(dists).abs()
                if plot_point_type is PlotPointType.ERROR_AT_GP_SURFACE:
                    rgb = color_map.to_rgba(dists.reshape(-1))
                    rgb = torch.from_numpy(rgb[:, :-1]).to(dtype=u.dtype, device=u.device)
                    for i in range(verts_xyz.shape[0]):
                        dd.draw_point(f'pt.{i}', verts_xyz[i], rgb[i], length=0.002)
                    dd.clear_visualization_after("pt", verts.shape[0])
                error_at_gp_surface.append(dists.mean())
                error_t.append(t)
                # save every time in case we break somewhere in between
                cache[name][randseed] = {'t': error_t,
                                         'error_at_model_points': error_at_model_points,
                                         'error_at_gp_surface': error_at_gp_surface}
                torch.save(cache, fullname)

    fig, axs = plt.subplots(3, 1, sharex="col", figsize=(8, 8), constrained_layout=True)
    axs[0].plot(error_t, error_at_model_points)
    axs[1].plot(error_t, error_at_gp_surface)
    avg_err = torch.stack([torch.tensor(error_at_gp_surface), torch.tensor(error_at_model_points)])

    # arithmetic mean
    # avg_err = avg_err.mean(dim=0)
    # harmonic mean
    avg_err = 1 / avg_err
    avg_err = 2 / (avg_err.sum(dim=0))

    axs[2].plot(error_t, avg_err)
    axs[0].set_ylabel('error at model points')
    axs[1].set_ylabel('error at gp surface')
    axs[2].set_ylabel('average error')
    axs[-1].set_xlabel('step')
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=0)
    axs[2].set_ylim(bottom=0)
    plt.show()

    return error_at_model_points


def score_icp(all_points, all_normals, distances):
    distance_score = torch.mean(distances, dim=1)
    # min z should be close to 0
    physics_score = all_points[:, :, 2].min(dim=1).values.abs()
    # TODO can score on surface normals being consistent with robot path? Or switch to ICP with normal support

    return distance_score + physics_score


def fit_gpis(x, df, threedimensional=True, training_iter=50, use_df=True, scale=5, likelihood=None, model=None):
    if not torch.is_tensor(x):
        x = torch.stack(x)
        df = torch.stack(df)
    f = torch.zeros(x.shape[0], dtype=x.dtype)

    # scale to be about the same magnitude
    train_x = x * scale
    if use_df:
        train_y = torch.cat((f.view(-1, 1), df), dim=1)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=4 if threedimensional else 3)  # Value + x-derivative + y-derivative
            model = GPModelWithDerivatives(train_x, train_y, likelihood)
        else:
            model.set_train_data(train_x, train_y, strict=False)
    else:
        train_y = f
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y, likelihood)
        else:
            model.set_train_data(train_x, train_y, strict=False)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        try:
            loss = -mll(output, train_y)
        except RuntimeError as e:
            print(e)
            continue
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.likelihood.noise.item()
        # ))
        # print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.squeeze()[0],
        #     model.covar_module.base_kernel.lengthscale.squeeze()[1],
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()
    # Set into eval mode
    model.eval()
    likelihood.eval()

    return likelihood, model


def direct_fit_2d():
    x = torch.tensor([
        [0.05, 0.01],
        [0.1, 0.02],
        [0.12, 0.025],
        [0.15, 0.05],
        [0.15, 0.08],
        [0.15, 0.10],
    ])
    f = torch.tensor([
        0, 0, 0, 0,
        0, 0
    ])
    df = torch.tensor([
        [0, 1],
        [-0.3, 0.8],
        [-0.1, 0.9],
        [-0.7, 0.2],
        [-0.8, 0.],
        [-0.8, 0.],
    ])
    train_x = x * 10
    train_y = torch.cat((f.view(-1, 1), df), dim=1)

    # official sample code
    # xv, yv = torch.meshgrid([torch.linspace(0, 1, 10), torch.linspace(0, 1, 10)])
    # train_x = torch.cat((
    #     xv.contiguous().view(xv.numel(), 1),
    #     yv.contiguous().view(yv.numel(), 1)),
    #     dim=1
    # )

    # f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
    # train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)  # Value + x-derivative + y-derivative
    model = GPModelWithDerivatives(train_x, train_y, likelihood)

    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.squeeze()[0],
            model.covar_module.base_kernel.lengthscale.squeeze()[1],
            model.likelihood.noise.item()
        ))
        optimizer.step()
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    # fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Test points
    n1, n2 = 50, 50
    xv, yv = torch.meshgrid([torch.linspace(-1, 3, n1), torch.linspace(-1, 3, n2)])
    # f, dfx, dfy = franke(xv, yv)

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        test_x = torch.stack([xv.reshape(n1 * n2, 1), yv.reshape(n1 * n2, 1)], -1).squeeze(1)
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    def gp_var(cx):
        pred = likelihood(model(cx))
        return pred.variance

    # query for next place to go based on gradient of variance
    gp_var_jacobian = jacobian(gp_var, x[-1])
    gp_var_grad = gp_var_jacobian[0]  # first dimension corresponds to SDF, other 2 are for the SDF gradient

    # project gradient onto tangent plane

    extent = (xv.min(), xv.max(), yv.min(), yv.max())
    # ax[0, 0].imshow(f, extent=extent, cmap=cm.jet)
    # ax[0, 0].set_title('True values')
    # ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
    # ax[0, 1].set_title('True x-derivatives')
    # ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
    # ax[0, 2].set_title('True y-derivatives')

    # ax = ax[1,0]
    imprint_norm = matplotlib.colors.Normalize(vmin=-5, vmax=5)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=imprint_norm), ax=ax)
    ax.imshow(mean[:, 0].detach().numpy().reshape(n1, n2).T, extent=extent, origin='lower', cmap=cm.jet)
    ax.set_title('Predicted values')

    # lower, upper = predictions.confidence_region()
    # std = (upper - lower) / 2
    # imprint_norm = matplotlib.colors.Normalize(vmin=0, vmax=torch.round(std.max()))
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=imprint_norm), ax=ax)
    # ax.imshow(std[:, 0].detach().numpy().reshape(n1, n2).T, extent=extent, origin='lower', cmap=cm.jet)
    # ax.set_title('Uncertainty (standard deviation)')

    ax.quiver(train_x[:, 0], train_x[:, 1], df[:, 0], df[:, 1], scale=10)

    # highlight the object boundary
    threshold = 0.01
    boundary = torch.abs(mean[:, 0]) < threshold
    boundary_x = test_x[boundary, :]
    ax.scatter(boundary_x[:, 0], boundary_x[:, 1], marker='.', c='k')

    # ax[1, 1].imshow(mean[:, 1].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    # ax[1, 1].set_title('Predicted x-derivatives')
    # ax[1, 2].imshow(mean[:, 2].detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
    # ax[1, 2].set_title('Predicted y-derivatives')
    plt.show()
    print('done')


def keyboard_control(env):
    print("waiting for arrow keys to be pressed to command a movement")
    contact_params = RetrievalGetter.contact_parameters(env)
    pt_to_config = arm.ArmPointToConfig(env)
    contact_set = tracking.ContactSetSoft(pt_to_config, contact_params)
    ctrl = KeyboardController(env.contact_detector, contact_set, nu=2)

    obs = env._obs()
    info = None
    while not ctrl.done():
        try:
            env.visualize_contact_set(contact_set)
            u = ctrl.command(obs, info)
            obs, _, done, info = env.step(u)
        except:
            pass
        time.sleep(0.05)
    print(ctrl.u_history)
    cleaned_u = [u for u in ctrl.u_history if u != (0, 0)]
    print(cleaned_u)


parser = argparse.ArgumentParser(description='Downstream task of blind object retrieval')
parser.add_argument('method',
                    choices=['ours', 'online-birch', 'online-dbscan', 'online-kmeans', 'gmphd'],
                    help='which method to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
# run parameters
task_map = {"FB": Levels.FLAT_BOX, "BC": Levels.BEHIND_CAN, "IB": Levels.IN_BETWEEN, "SC": Levels.SIMPLE_CLUTTER,
            "TC": Levels.TOMATO_CAN}
parser.add_argument('--task', default="IB", choices=task_map.keys(), help='what task to run')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    method_name = args.method

    # env = RetrievalGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI)

    # direct_fit_2d()
    test_existing_method_3d()
