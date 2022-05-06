import abc
import argparse
import time
import typing

import matplotlib
import numpy as np
import pybullet_data
import pymeshlab
import pytorch_kinematics
import torch
import pybullet as p
import logging
import os
from datetime import datetime

import gpytorch
from matplotlib import cm
from matplotlib import pyplot as plt

from arm_pytorch_utilities import tensor_utils, rand
from arm_pytorch_utilities.grad import jacobian
from pybullet_object_models import ycb_objects
from pytorch_kinematics import transforms as tf
from torchmcubes import marching_cubes

from stucco import cfg, icp
from stucco import tracking
from stucco.env import arm
from stucco.env.arm import Levels
from stucco.env.env import Visualizer
from stucco.env.pybullet_env import make_sphere, DebugDrawer, closest_point_on_surface, ContactInfo, \
    surface_normal_at_point
from stucco.env_getters.arm import RetrievalGetter
from stucco import exploration
from stucco.exploration import PlotPointType, ShapeExplorationPolicy, ICPEVExplorationPolicy, GPVarianceExploration, \
    score_icp

from stucco.retrieval_controller import sample_model_points, KeyboardController

import pytorch_kinematics.transforms as tf

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def test_icp(target_obj_id, vis_obj_id, vis: Visualizer, seed=0, name="", clean_cache=False, viewing_delay=0.3,
             register_num_points=500, eval_num_points=200, num_points_list=(5, 10, 20, 30, 40, 50, 100),
             save_best_tsf=True,
             model_name="mustard_normal"):
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
        if seed not in cache[name] or clean_cache:
            cache[name][seed] = {}
    else:
        cache = {name: {seed: {}}}

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(target_obj_id, num_points=eval_num_points,
                                                                   force_z=None,
                                                                   mid_z=0.05,
                                                                   seed=0, clean_cache=False,
                                                                   random_sample_sigma=0.2,
                                                                   name=model_name, vis=None,
                                                                   restricted_points=(
                                                                       [(0.01897749298212774,
                                                                         -0.008559855822130511,
                                                                         0.001455972652355926)]))

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(target_obj_id,
                                                                           num_points=register_num_points,
                                                                           force_z=None,
                                                                           mid_z=0.05,
                                                                           seed=seed, clean_cache=False,
                                                                           random_sample_sigma=0.2,
                                                                           name=model_name, vis=None,
                                                                           restricted_points=(
                                                                               [(0.01897749298212774,
                                                                                 -0.008559855822130511,
                                                                                 0.001455972652355926)]))
    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    best_tsf_guess = None
    for num_points in num_points_list:
        # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
        model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=num_points, force_z=None,
                                                             mid_z=0.05,
                                                             seed=seed, clean_cache=False, random_sample_sigma=0.2,
                                                             name=model_name, vis=None, restricted_points=(
                [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))

        device, dtype = model_points.device, model_points.dtype
        pose = p.getBasePositionAndOrientation(target_obj_id)
        link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
        model_points_world_frame = link_to_current_tf_gt.transform_points(model_points)
        model_normals_world_frame = link_to_current_tf_gt.transform_normals(model_normals)
        model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)

        for i, pt in enumerate(model_points_world_frame):
            vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
            vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
        vis.clear_visualization_after("mpt", i + 1)
        vis.clear_visualization_after("mn", i + 1)

        rand.seed(seed)
        # perform ICP and visualize the transformed points
        # reverse engineer the transform
        # compare not against current model points (which may be few), but against the maximum number of model points
        B = 30
        T, distances, _ = icp.icp_3(model_points_world_frame, model_points_register, given_init_pose=best_tsf_guess,
                                    batch=B)

        # link_to_current_tf = tf.Transform3d(matrix=T.inverse())
        # all_points = link_to_current_tf.transform_points(model_points)
        # all_normals = link_to_current_tf.transform_normals(model_normals)

        # score ICP and save best one to initialize for next step
        if save_best_tsf:
            link_to_current_tf = tf.Transform3d(matrix=T.inverse())
            all_points = link_to_current_tf.transform_points(model_points)
            all_normals = link_to_current_tf.transform_normals(model_normals)
            score = score_icp(all_points, all_normals, distances).numpy()
            best_tsf_index = np.argmin(score)
            best_tsf_guess = T[best_tsf_index].inverse()

        # due to inherent symmetry, can't just use the known correspondence to measure error, since it's ok to mirror
        # we're essentially measuring the chamfer distance (acts on 2 point clouds), where one point cloud is the
        # evaluation model points on the ground truth object surface, and the surface points of the object transformed
        # by our estimated pose (which is infinitely dense)
        # this is the unidirectional chamfer distance since we're only measuring dist of eval points to surface
        chamfer_distance = torch.zeros(B, eval_num_points)
        link_to_world = tf.Transform3d(matrix=T.inverse())
        m = link_to_world.get_matrix()
        for b in range(B):
            pos = m[b, :3, 3]
            rot = pytorch_kinematics.matrix_to_quaternion(m[b, :3, :3])
            rot = tf.wxyz_to_xyzw(rot)
            p.resetBasePositionAndOrientation(vis_obj_id, pos, rot)

            # transform our visual object to the pose
            for i in range(eval_num_points):
                closest = closest_point_on_surface(vis_obj_id, model_points_world_frame_eval[i])
                chamfer_distance[b, i] = (1000 * closest[ContactInfo.DISTANCE]) ** 2  # convert m^2 to mm^2

            vis.draw_point("err", (0, 0, 0.1), (1, 0, 0),
                           label=f"err: {chamfer_distance[b].abs().mean().item():.5f}")
            vis.draw_point("dist", (0, 0, 0.2), (1, 0, 0), label=f"dist: {distances[b].mean().item():.5f}")
            time.sleep(viewing_delay)

        errors_per_transform = chamfer_distance.mean(dim=-1)
        errors.append(errors_per_transform.mean())

    for num, err in zip(num_points_list, errors):
        cache[name][seed][num] = err
    torch.save(cache, fullname)
    for i in range(len(num_points_list)):
        print(f"num {num_points_list[i]} err {errors[i]}")


def plot_icp_results(names_to_include=None, logy=True):
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison.pkl')
    cache = torch.load(fullname)

    fig, axs = plt.subplots(1, 1, sharex="col", figsize=(8, 8), constrained_layout=True)
    if logy:
        axs.set_yscale('log')

    for name in cache.keys():
        if names_to_include is not None and name not in names_to_include:
            print(f"ignored {name}")
            continue
        num_points = []
        errors = []
        for seed in cache[name]:
            data = cache[name][seed]
            # short by num points
            a, b = zip(*sorted(data.items(), key=lambda e: e[0]))
            num_points.append(a)
            errors.append(b)

        # assume all the num errors are the same
        # convert to cm^2 (saved as mm^2, so divide by 10^2
        errors = np.stack(errors) / 100
        mean = errors.mean(axis=0)
        std = errors.std(axis=0)
        x = num_points[0]
        axs.plot(x, mean, label=name)
        # axs.errorbar(x, mean, std, label=name)
        axs.fill_between(x, mean - std, mean + std, alpha=0.2)

        # print each numerically
        for i in range(len(mean)):
            print(f"{name} {x[i]:>4} : {mean[i]:.2f} ({std[i]:.2f})")
        print()

    axs.set_ylabel('unidirectional chamfer dist (UCD [cm^2])')
    axs.set_xlabel('num test points')
    if not logy:
        axs.set_ylim(bottom=0)
    axs.legend()
    plt.show()


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
    model = exploration.GPModelWithDerivatives(train_x, train_y, likelihood)

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


def make_mustard_bottle(z):
    objId = p.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbMustardBottle', "model.urdf"),
                       [0., 0., z * 3],
                       p.getQuaternionFromEuler([0, 0, -1]), globalScaling=2.5)
    ranges = np.array([[-.3, .3], [-.3, .3], [-0.1, .8]])
    return objId, ranges


def make_sphere_preconfig(z):
    objId = make_sphere(z, [0., 0, z])
    ranges = np.array([[-.2, .2], [-.2, .2], [-.1, .4]])
    return objId, ranges


class ShapeExplorationExperiment(abc.ABC):
    LINK_FRAME_POS = [0, 0, 0]
    LINK_FRAME_ORIENTATION = [0, 0, 0, 1]

    def __init__(self, make_obj=make_mustard_bottle, eval_period=10,
                 plot_per_eval_period=1,
                 plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS):
        self.policy: typing.Optional[ShapeExplorationPolicy] = None
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

        # log video
        self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                              "{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        self.z = 0.1
        self.objId, self.ranges = self.make_obj(self.z)
        # purely visual object to allow us to see estimated pose
        self.visId, _ = self.make_obj(self.z)
        p.resetBasePositionAndOrientation(self.visId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.visId, -1, mass=0)
        p.changeVisualShape(self.visId, -1, rgbaColor=[0.2, 0.8, 1.0, 0.5])
        p.setCollisionFilterPair(self.objId, self.visId, -1, -1, 0)

        self.has_run = False

    def set_policy(self, policy: ShapeExplorationPolicy):
        self.policy = policy

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
        model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=100, force_z=None,
                                                             mid_z=0.05,
                                                             seed=0, clean_cache=build_model,
                                                             random_sample_sigma=0.2,
                                                             name=model_name, vis=vis, restricted_points=(
                [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926)]))
        pose = p.getBasePositionAndOrientation(target_obj_id)

        self.policy.save_model_points(model_points, model_normals, pose)

        if build_model:
            for i, pt in enumerate(self.policy.model_points_world_transformed_ground_truth):
                vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
                vis.draw_2d_line(f"mn.{i}", pt, -self.policy.model_normals_world_transformed_ground_truth[i],
                                 color=(0, 0, 0),
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
            self.policy.start_step(xs, df)
            dx = self.policy.get_next_dx(xs, df, t)

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

            self.policy.end_step(xs, df, t)

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


class ICPEVExperiment(ShapeExplorationExperiment):
    def __init__(self, policy_factory=ICPEVExplorationPolicy, policy_args=None, **kwargs):
        if policy_args is None:
            policy_args = {}
        super(ICPEVExperiment, self).__init__(**kwargs)

        self.testObjId, _ = self.make_obj(self.z)
        p.resetBasePositionAndOrientation(self.testObjId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.testObjId, -1, mass=0)
        p.changeVisualShape(self.testObjId, -1, rgbaColor=[0, 0, 0, 0])
        p.setCollisionFilterPair(self.objId, self.testObjId, -1, -1, 0)

        self.set_policy(policy_factory(test_obj_id=self.testObjId, **policy_args))

    def _start_step(self, xs, df):
        pass

    def _end_step(self, xs, df, t):
        pass

    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        # get points after transforming with best ICP pose estimate

        link_to_world = tf.Transform3d(matrix=self.policy.best_tsf_guess)
        m = link_to_world.get_matrix()
        pos = m[:, :3, 3]
        rot = pytorch_kinematics.matrix_to_quaternion(m[:, :3, :3])
        rot = tf.wxyz_to_xyzw(rot)
        p.resetBasePositionAndOrientation(self.visId, pos[0], rot[0])

        model_pts_to_compare = self.policy.model_points_world_transformed_ground_truth

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
            for i in range(self.policy.model_points.shape[0]):
                self.dd.draw_point(f'pt.{i}', model_pts_to_compare[i], rgb[i], length=0.002)
            self.dd.clear_visualization_after("pt", model_pts_to_compare.shape[0])


class GPVarianceExperiment(ShapeExplorationExperiment):
    def __init__(self, gp_exploration_policy: GPVarianceExploration, **kwargs):
        super(GPVarianceExperiment, self).__init__(**kwargs)
        self.set_policy(gp_exploration_policy)

    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        if not isinstance(self.policy, GPVarianceExploration):
            raise RuntimeError("This experiment requires GP Exploration Policy")
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
                predictions = self.policy.gp(test_x[i:i + PER_BATCH])
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
                    predictions = self.policy.gp(verts_xyz[i:i + PER_BATCH])
                    var = predictions.variance
                    vars.append(var[:, 0])
                var = torch.cat(vars).contiguous()

                faces = faces.cpu().numpy()
                colrs = color_map.to_rgba(var.reshape(-1))
                # note, can control alpha on last column here
                colrs[:, -1] = self.policy.mesh_surface_alpha

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
                predictions = self.policy.gp(self.policy.model_points_world_transformed_ground_truth)
            mean = predictions.mean
            err = torch.abs(mean[:, 0])
            if self.plot_point_type is PlotPointType.ERROR_AT_MODEL_POINTS:
                rgb = color_map.to_rgba(err.reshape(-1))
                rgb = torch.from_numpy(rgb[:, :-1]).to(dtype=u.dtype, device=u.device)
                for i in range(self.policy.model_points_world_transformed_ground_truth.shape[0]):
                    self.dd.draw_point(f'pt.{i}', self.policy.model_points_world_transformed_ground_truth[i], rgb[i],
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

    experiment = ICPEVExperiment()
    experiment.dd.set_camera_position([0., 0.3], yaw=0, pitch=-30)
    for gt_num in [30, 50, 80, 100, 200, 500]:
        for seed in range(10):
            test_icp(experiment.objId, experiment.visId, experiment.dd, seed=seed, register_num_points=gt_num,
                     name=f"save tsf {gt_num} mp", viewing_delay=0)

    plot_icp_results()

    # experiment.run(run_name="icp_var_debug_3")
    # experiment = ICPEVExperiment(exploration.ICPEVSampleModelPointsPolicy)
    # experiment.run(run_name="icp_var_sample_points_overstep")
    # experiment = GPVarianceExploration()
    # experiment.run(run_name="gp_var")