import abc
import argparse
import enum
import time
import math

import matplotlib
import pytorch_kinematics
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
    ERROR_AT_REP_SURFACE = 3
    ICP_ERROR_POINTS = 4


def make_mustard_bottle(z):
    objId = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"),
                       [0., 0., z * 3],
                       p.getQuaternionFromEuler([0, 0, -1]), globalScaling=2.5)
    ranges = np.array([[-.2, .2], [-.2, .2], [0, .5]])
    return objId, ranges


def make_sphere_preconfig(z):
    objId = make_sphere(z, [0., 0, z])
    ranges = np.array([[-.2, .2], [-.2, .2], [-.1, .4]])
    return objId, ranges


class ShapeExplorationExperiment(abc.ABC):
    LINK_FRAME_POS = [0, 0, 0]
    LINK_FRAME_ORIENTATION = [0, 0, 0, 1]

    def __init__(self, make_obj=make_mustard_bottle, eval_period=10, plot_per_eval_period=1,
                 plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS):
        self.make_obj = make_obj
        self.plot_point_type = plot_point_type
        self.plot_per_eval_period = plot_per_eval_period
        self.eval_period = eval_period

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.dd = DebugDrawer(0.8, 0.8)
        self.dd.toggle_3d(True)

        self.z = 0.1
        self.objId, self.ranges = self.make_obj(self.z)
        # purely visual object to allow us to see estimated pose
        self.visId, _ = self.make_obj(self.z)
        p.resetBasePositionAndOrientation(self.visId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.visId, -1, mass=0)
        p.changeVisualShape(self.visId, -1, rgbaColor=[0.2, 0.8, 1.0, 0.5])
        p.setCollisionFilterPair(self.objId, self.visId, -1, -1, 0)

        self.model_points = None
        self.model_normals = None
        self.model_points_world_transformed_ground_truth = None
        self.model_normals_world_transformed_ground_truth = None

        self.best_tsf_guess = None

        self.has_run = False

    @abc.abstractmethod
    def _start_step(self, xs, df):
        """Bookkeeping at the start of a step"""

    @abc.abstractmethod
    def _end_step(self, xs, df, t):
        """Bookkeeping at the end of a step, with new x and d appended"""

    @abc.abstractmethod
    def _get_next_dx(self, xs, df, t):
        """Get control for deciding which state to visit next"""

    @abc.abstractmethod
    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        """Evaluate errors and append to given lists"""

    def run(self, seed=0, timesteps=202, build_model=False, clean_cache=False,
            model_name="mustard_normal", run_name=""):
        if self.has_run:
            return RuntimeError("Experiment can only be run once for now; missing reset function")

        # wait for it to settle
        for _ in range(1000):
            p.stepSimulation()

        target_obj_id = self.objId
        vis = self.dd
        name = f"{model_name} {run_name}".strip()

        # these are in object frame (aligned with [0,0,0], [0,0,0,1]
        self.model_points, self.model_normals, _ = sample_model_points(target_obj_id, num_points=100, force_z=None,
                                                                       mid_z=0.05,
                                                                       seed=0, clean_cache=build_model,
                                                                       random_sample_sigma=0.2,
                                                                       name=model_name, vis=vis, restricted_points=(
                [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))

        self.device, self.dtype = self.model_points.device, self.model_points.dtype
        pose = p.getBasePositionAndOrientation(target_obj_id)
        link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        self.model_points_world_transformed_ground_truth = link_to_current_tf.transform_points(self.model_points)
        self.model_normals_world_transformed_ground_truth = link_to_current_tf.transform_normals(self.model_normals)
        if build_model:
            for i, pt in enumerate(self.model_points_world_transformed_ground_truth):
                vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
                vis.draw_2d_line(f"mn.{i}", pt, -self.model_normals_world_transformed_ground_truth[i], color=(0, 0, 0),
                                 size=2.,
                                 scale=0.03)

        # start at a point on the surface of the bottle
        randseed = rand.seed(seed)
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "{}_{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                                          randseed))
        fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
        if os.path.exists(fullname):
            cache = torch.load(fullname)
            if name not in cache or clean_cache:
                cache[name] = {}
        else:
            cache = {name: {}}

        x = torch.tensor(closest_point_on_surface(self.objId, np.random.rand(3))[ContactInfo.POS_A])
        self.dd.draw_point('x', x, height=x[2])
        n = -torch.tensor(surface_normal_at_point(self.objId, x))
        self.dd.draw_2d_line('n', x, n, (0, 0.5, 0), scale=0.2)

        xs = [x]
        df = [n]

        # error data
        error_t = []
        error_at_model_points = []
        error_at_rep_surface = []

        for t in range(timesteps):
            self._start_step(xs, df)
            dx = self._get_next_dx(xs, df, t)

            self.dd.draw_2d_line('a', x, dx, (1, 0., 0), scale=1)

            new_x = x + dx
            # project onto object (via low level controller applying a force)
            new_x = torch.tensor(closest_point_on_surface(self.objId, new_x)[ContactInfo.POS_A])

            self.dd.draw_transition(x, new_x)
            x = new_x
            self.dd.draw_point('x', x, height=x[2])
            n = -torch.tensor(surface_normal_at_point(self.objId, x))
            self.dd.draw_2d_line('n', x, n, (0, 0.5, 0), scale=0.2)

            xs.append(x)
            df.append(n)

            self._end_step(xs, df, t)

            # after a period of time evaluate current level set
            if t > 0 and t % self.eval_period == 0:
                print('evaluating shape ' + str(t))
                self._eval(xs, df, error_at_model_points, error_at_rep_surface, t)

                error_t.append(t)
                # save every time in case we break somewhere in between
                cache[name][randseed] = {'t': error_t,
                                         'error_at_model_points': error_at_model_points,
                                         'error_at_rep_surface': error_at_rep_surface}
                torch.save(cache, fullname)

        self.has_run = True
        return error_at_model_points


class ICPErrorVarianceExploration(ShapeExplorationExperiment):
    def __init__(self, alpha=0.01, alpha_evaluate=0.05, verify_icp_error=False, **kwargs):
        super(ICPErrorVarianceExploration, self).__init__(**kwargs)
        self.alpha = alpha
        self.alpha_evaluate = alpha_evaluate
        self.verify_icp_error = verify_icp_error

        self.best_tsf_guess = None
        self.T = None

        self.testObjId, _ = self.make_obj(self.z)
        p.resetBasePositionAndOrientation(self.testObjId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.testObjId, -1, mass=0)
        p.changeVisualShape(self.testObjId, -1, rgbaColor=[0, 0, 0, 0])
        p.setCollisionFilterPair(self.objId, self.testObjId, -1, -1, 0)

    def _start_step(self, xs, df):
        pass

    def _end_step(self, xs, df, t):
        pass

    def _get_next_dx(self, xs, df, t):
        x = xs[-1]
        n = df[-1]
        if t > 5:
            # query for next place to go based on approximated uncertainty using ICP error variance
            with rand.SavedRNG():
                N = 100
                B = 30
                dx_samples = sample_dx_on_tange_plane(n, self.alpha_evaluate, num_samples=N)
                new_x_samples = x + dx_samples
                # do ICP
                this_pts = torch.stack(xs).reshape(-1, 3)
                self.T, distances, _ = icp.icp_3(this_pts, self.model_points, given_init_pose=self.best_tsf_guess,
                                                 batch=B)
                # model points are given in link frame; new_x_sample points are in world frame

                # T = T.inverse()
                point_tf_to_link = tf.Transform3d(matrix=self.T)
                all_points = point_tf_to_link.transform_points(new_x_samples)

                # compute ICP error for new sampled points
                query_icp_error = torch.zeros(B, N)
                # points are transformed to link frame, thus it needs to compare against the object in link frame
                # objId is not in link frame and shouldn't be moved
                for b in range(B):
                    for i in range(N):
                        closest = closest_point_on_surface(self.testObjId, all_points[b, i])
                        query_icp_error[b, i] = closest[ContactInfo.DISTANCE]
                        if self.plot_point_type == PlotPointType.ICP_ERROR_POINTS:
                            self.dd.draw_point("test_point", all_points[b, i], color=(1, 0, 0), length=0.005)
                            self.dd.draw_point("test_point_surf", closest[ContactInfo.POS_A], color=(0, 1, 0),
                                               length=0.005,
                                               label=f'{closest[ContactInfo.DISTANCE]:.5f}')

                # don't care about sign of penetration or separation
                query_icp_error = query_icp_error.abs()

                if self.verify_icp_error:
                    # compare our method of transforming all points to link frame with transforming all objects
                    query_icp_error_ground_truth = torch.zeros(B, N)
                    link_to_world = tf.Transform3d(matrix=self.T.inverse())
                    m = link_to_world.get_matrix()
                    for b in range(B):
                        pos = m[b, :3, 3]
                        rot = pytorch_kinematics.matrix_to_quaternion(m[b, :3, :3])
                        rot = tf.wxyz_to_xyzw(rot)
                        p.resetBasePositionAndOrientation(self.visId, pos, rot)

                        # transform our visual object to the pose
                        for i in range(N):
                            closest = closest_point_on_surface(self.visId, new_x_samples[i])
                            query_icp_error_ground_truth[b, i] = closest[ContactInfo.DISTANCE]
                            if self.plot_point_type == PlotPointType.ICP_ERROR_POINTS:
                                self.dd.draw_point("test_point", new_x_samples[i], color=(1, 0, 0), length=0.005)
                                self.dd.draw_point("test_point_surf", closest[ContactInfo.POS_A], color=(0, 1, 0),
                                                   length=0.005,
                                                   label=f'{closest[ContactInfo.DISTANCE]:.5f}')
                    query_icp_error_ground_truth = query_icp_error_ground_truth.abs()
                    assert (query_icp_error_ground_truth - query_icp_error).sum() < 1e-4

                # find error variance for each sampled dx
                icp_error_var = query_icp_error.var(dim=0)
                most_informative_idx = icp_error_var.argmax()
                # choose gradient based on this (basically throw out GP calculations)
                dx = dx_samples[most_informative_idx]
                print(f"chose action {most_informative_idx.item()} / {N} {dx} "
                      f"error var {icp_error_var[most_informative_idx].item():.5f} "
                      f"avg error var {icp_error_var.mean().item():.5f}")

                # score ICP and save best one to initialize for next step
                link_to_current_tf = tf.Transform3d(matrix=self.T.inverse())
                all_points = link_to_current_tf.transform_points(self.model_points)
                all_normals = link_to_current_tf.transform_normals(self.model_normals)
                # TODO can score on surface normals being consistent with robot path? Or switch to ICP with normal support
                score = score_icp(all_points, all_normals, distances).numpy()
                best_tsf_index = np.argmin(score)
                self.best_tsf_guess = self.T[best_tsf_index].inverse()
        else:
            dx = torch.randn(3, dtype=self.dtype, device=self.device)
            dx = dx / dx.norm() * self.alpha
        return dx

    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        # get points after transforming with best ICP pose estimate

        link_to_world = tf.Transform3d(matrix=self.best_tsf_guess)
        m = link_to_world.get_matrix()
        pos = m[:, :3, 3]
        rot = pytorch_kinematics.matrix_to_quaternion(m[:, :3, :3])
        rot = tf.wxyz_to_xyzw(rot)
        p.resetBasePositionAndOrientation(self.visId, pos[0], rot[0])

        model_pts_to_compare = self.model_points_world_transformed_ground_truth

        # transform our visual object to the pose
        dists = []
        for i in range(model_pts_to_compare.shape[0]):
            closest = closest_point_on_surface(self.visId, model_pts_to_compare[i])
            dists.append(closest[ContactInfo.DISTANCE])
            # if self.plot_point_type == PlotPointType.ERROR_AT_MODEL_POINTS:
            #     self.dd.draw_point("test_point", self.model_points[i], color=(1, 0, 0), length=0.005)
            #     self.dd.draw_point("test_point_surf", closest[ContactInfo.POS_A], color=(0, 1, 0),
            #                        length=0.005,
            #                        label=f'{closest[ContactInfo.DISTANCE]:.5f}')
        dists = torch.tensor(dists).abs()
        err = dists.mean()
        error_at_model_points.append(err)
        error_at_rep_surface.append(err)

        if self.plot_point_type is PlotPointType.ERROR_AT_MODEL_POINTS:
            error_norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
            color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
            rgb = color_map.to_rgba(dists.reshape(-1))
            rgb = rgb[:, :-1]
            for i in range(self.model_points.shape[0]):
                self.dd.draw_point(f'pt.{i}', model_pts_to_compare[i], rgb[i], length=0.002)
            self.dd.clear_visualization_after("pt", model_pts_to_compare.shape[0])


class GPVarianceExploration(ShapeExplorationExperiment):
    def __init__(self, alpha=0.01, training_iter=50, icp_period=200, gpscale=5, verify_numerical_gradients=False,
                 mesh_surface_alpha=1., **kwargs):
        super(GPVarianceExploration, self).__init__(**kwargs)
        self.alpha = alpha
        self.training_iter = training_iter
        self.gpscale = gpscale
        self.verify_numerical_gradients = verify_numerical_gradients
        self.icp_period = icp_period
        self.mesh_surface_alpha = mesh_surface_alpha

        self.likelihood = None
        self.model = None
        self.aug_xs = None
        self.aug_df = None
        self.meshId = None

    def gp(self, cx):
        pred = self.likelihood(self.model(cx * self.gpscale))
        return pred

    def gp_var(self, cx):
        return self.gp(cx).variance

    def _start_step(self, xs, df):
        train_xs = xs
        train_df = df
        if self.aug_xs is not None:
            train_xs = torch.cat([torch.stack(xs), self.aug_xs.view(-1, 3)], dim=0)
            train_df = torch.cat([torch.stack(df), self.aug_df.view(-1, 3)], dim=0)
        self.likelihood, self.model = fit_gpis(train_xs, train_df, threedimensional=True, use_df=True,
                                               scale=self.gpscale,
                                               training_iter=self.training_iter, likelihood=self.likelihood,
                                               model=self.model)

        if self.verify_numerical_gradients:
            x = xs[-1]
            n = df[-1]
            with rand.SavedRNG():
                # try numerically computing the gradient along valid directions only
                dx_samples = sample_dx_on_tange_plane(n, self.alpha)
                new_x_samples = x + dx_samples
                with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False,
                                                                          covar_root_decomposition=False):
                    pred_samples = self.gp(new_x_samples)
                    # select dx corresponding to largest var in first dimension
                    sample_ordering = torch.argsort(pred_samples.variance[:, 0], descending=True)
                    # sample_ordering = torch.argsort(pred_samples.variance, descending=True)
                dx_numerical = dx_samples[sample_ordering[0]]
                for k in range(10):
                    strength = (10 - k) / 10
                    self.dd.draw_2d_line(f'dx{k}', x, dx_samples[sample_ordering[k]],
                                         (strength, 1 - strength, 1 - strength),
                                         scale=3)

    def _end_step(self, xs, df, t):
        # augment data with ICP shape pseudo-observations
        if t > 0 and t % self.icp_period == 0:
            print('conditioning exploration on ICP results')
            # perform ICP and visualize the transformed points
            this_pts = torch.stack(xs).reshape(-1, 3)
            T, distances, _ = icp.icp_3(this_pts, self.model_points,
                                        given_init_pose=self.best_tsf_guess, batch=30)
            T = T.inverse()
            link_to_current_tf = tf.Transform3d(matrix=T)
            all_points = link_to_current_tf.transform_points(self.model_points)
            all_normals = link_to_current_tf.transform_normals(self.model_normals)

            # TODO can score on surface normals being consistent with robot path? Or switch to ICP with normal support
            score = score_icp(all_points, all_normals, distances).numpy()
            best_tsf_index = np.argmin(score)
            self.best_tsf_guess = T[best_tsf_index].inverse()

            if self.plot_point_type is PlotPointType.ICP_MODEL_POINTS:
                color = (1, 0, 0)
                for i, pt in enumerate(all_points[best_tsf_index]):
                    self.dd.draw_point(f"best.{i}", pt, color=color, length=0.003)

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

    def _get_next_dx(self, xs, df, t):
        x = xs[-1]
        n = df[-1]

        # query for next place to go based on gradient of variance
        gp_var_jacobian = jacobian(self.gp_var, x)
        gp_var_grad = gp_var_jacobian[0]  # first dimension corresponds to SDF, other 2 are for the SDF gradient
        # choose random direction if it's the 0 vector
        if torch.allclose(gp_var_grad, torch.zeros_like(gp_var_grad)):
            gp_var_grad = torch.rand_like(gp_var_grad)
        dx = project_to_plane(n, gp_var_grad)

        # normalize to be magnitude alpha
        dx = dx / dx.norm() * self.alpha
        return dx

    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        # see what's the range of values we've actually traversed
        xx = torch.stack(xs)
        print(f'ranges: {torch.min(xx, dim=0)} - {torch.max(xx, dim=0)}')
        # n1, n2, n3 = 80, 80, 50
        num = [40, 40, 30]
        xv, yv, zv = torch.meshgrid(
            [torch.linspace(*self.ranges[0], num[0]), torch.linspace(*self.ranges[1], num[1]),
             torch.linspace(*self.ranges[2], num[2])])
        # Make GP predictions
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False,
                                                                  covar_root_decomposition=False):
            test_x = torch.stack(
                [xv.reshape(np.prod(num), 1), yv.reshape(np.prod(num), 1), zv.reshape(np.prod(num), 1)],
                -1).squeeze(1)
            PER_BATCH = 256
            vars = []
            us = []
            for i in range(0, test_x.shape[0], PER_BATCH):
                predictions = self.gp(test_x[i:i + PER_BATCH])
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
            for dim in range(self.ranges.shape[0]):
                verts_xyz[:, dim] /= num[dim] - 1
                verts_xyz[:, dim] = verts_xyz[:, dim] * (self.ranges[dim][1] - self.ranges[dim][0]) + self.ranges[dim][
                    0]

            # plot mesh
            if t > 0 and t % (self.eval_period * self.plot_per_eval_period) == 0:
                vars = []
                for i in range(0, verts.shape[0], PER_BATCH):
                    predictions = self.gp(verts_xyz[i:i + PER_BATCH])
                    var = predictions.variance
                    vars.append(var[:, 0])
                var = torch.cat(vars).contiguous()

                faces = faces.cpu().numpy()
                colrs = color_map.to_rgba(var.reshape(-1))
                # note, can control alpha on last column here
                colrs[:, -1] = self.mesh_surface_alpha

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

                if self.meshId is not None:
                    p.removeBody(self.meshId)
                visId = p.createVisualShape(p.GEOM_MESH, fileName=fn)
                self.meshId = p.createMultiBody(0, baseVisualShapeIndex=visId, basePosition=[0, 0, 0])

                # input('enter to clear visuals')
                self.dd.clear_visualization_after('grad', 0)

            # evaluate surface error
            error_norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)
            color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
            with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False,
                                                                      covar_root_decomposition=False):
                predictions = self.gp(self.model_points_world_transformed_ground_truth)
            mean = predictions.mean
            err = torch.abs(mean[:, 0])
            if self.plot_point_type is PlotPointType.ERROR_AT_MODEL_POINTS:
                rgb = color_map.to_rgba(err.reshape(-1))
                rgb = torch.from_numpy(rgb[:, :-1]).to(dtype=u.dtype, device=u.device)
                for i in range(self.model_points_world_transformed_ground_truth.shape[0]):
                    self.dd.draw_point(f'pt.{i}', self.model_points_world_transformed_ground_truth[i], rgb[i],
                                       length=0.002)
                self.dd.clear_visualization_after("pt", verts.shape[0])
            error_at_model_points.append(err.mean())

            # evaluation of estimated SDF surface points on ground truth
            dists = []
            for vert in verts_xyz:
                closest = closest_point_on_surface(self.objId, vert)
                dists.append(closest[ContactInfo.DISTANCE])
            dists = torch.tensor(dists).abs()
            if self.plot_point_type is PlotPointType.ERROR_AT_REP_SURFACE:
                rgb = color_map.to_rgba(dists.reshape(-1))
                rgb = torch.from_numpy(rgb[:, :-1]).to(dtype=u.dtype, device=u.device)
                for i in range(verts_xyz.shape[0]):
                    self.dd.draw_point(f'pt.{i}', verts_xyz[i], rgb[i], length=0.002)
                self.dd.clear_visualization_after("pt", verts.shape[0])
            error_at_rep_surface.append(dists.mean())


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
    experiment = ICPErrorVarianceExploration()
    experiment.run(run_name="icp_var_debug_3")
    # experiment = GPVarianceExploration()
