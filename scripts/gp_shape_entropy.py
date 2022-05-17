import abc
import argparse
import time
import typing
import re

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
import matplotlib.colors, matplotlib.cm
from matplotlib import cm
from matplotlib import pyplot as plt

from arm_pytorch_utilities import tensor_utils, rand
from arm_pytorch_utilities.grad import jacobian
from pybullet_object_models import ycb_objects
from torchmcubes import marching_cubes

from stucco import cfg, icp
from stucco.env.arm import Levels
from stucco.env.env import Visualizer
from stucco.env.pybullet_env import make_sphere, DebugDrawer, closest_point_on_surface, ContactInfo, \
    surface_normal_at_point
from stucco import exploration
from stucco.exploration import PlotPointType, ShapeExplorationPolicy, ICPEVExplorationPolicy, GPVarianceExploration, \
    PybulletObjectFactory

from stucco.retrieval_controller import sample_model_points

import pytorch_kinematics.transforms as tf

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

restricted_pts = {
    'mustard_normal': [(0.01897749298212774, -0.008559855822130511, 0.001455972652355926), (0.0314, -0.0126, 0.2169),
                       (-0.0348, -0.0616, -0.0007), (0.0450, 0.0021, 0.2208), (-0.0177, -0.0202, 0.2220),
                       (-0.0413, 0.0119, -0.0006), (0.0126, 0.0265, 0.0018), (-0.0090, 0.0158, 0.2203),
                       (-0.0114, -0.0462, -0.0009), (0.0103, -0.0085, 0.2200), (0.0096, -0.0249, 0.2201)]
}

reject_model_pts = {
    'mustard_normal': lambda pt, normal: abs(normal[2]) > 0.99 and abs(pt[2]) < 0.01
}


def build_model(target_obj_id, vis, model_name, seed, num_points, pause_at_end=False):
    points, normals, _ = sample_model_points(target_obj_id, reject_too_close=0.006,
                                             num_points=num_points,
                                             force_z=None,
                                             mid_z=0.05,
                                             seed=seed, clean_cache=True,
                                             random_sample_sigma=0.2,
                                             name=model_name, vis=None,
                                             other_rejection_criteria=reject_model_pts[model_name],
                                             restricted_points=restricted_pts[model_name])
    for i, pt in enumerate(points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -normals[i], color=(0, 0, 0), size=2., scale=0.03)

    print(f"finished building {model_name} {seed} {num_points}")
    if pause_at_end:
        input("paused for inspection")
    vis.clear_visualizations()


def test_icp(target_obj_id, vis_obj_id, vis: Visualizer, seed=0, name="", clean_cache=False, viewing_delay=0.3,
             register_num_points=500, eval_num_points=200, num_points_list=(5, 10, 20, 30, 40, 50, 100),
             save_best_tsf=False, save_best_only_on_improvement=False, normal_scale=0.05,
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
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)
    for num_points in num_points_list:
        # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
        model_points, model_normals, _ = sample_model_points(num_points=num_points, name=model_name, seed=seed)

        pose = p.getBasePositionAndOrientation(target_obj_id)
        link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
        model_points_world_frame = link_to_current_tf_gt.transform_points(model_points)
        model_normals_world_frame = link_to_current_tf_gt.transform_normals(model_normals)
        model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)

        i = 0
        for i, pt in enumerate(model_points_world_frame):
            vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
            vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
        vis.clear_visualization_after("mpt", i + 1)
        vis.clear_visualization_after("mn", i + 1)

        rand.seed(seed)
        # perform ICP and visualize the transformed points
        # reverse engineer the transform
        # compare not against current model points (which may be few), but against the maximum number of model points
        # T, distances, _ = icp.icp_3(model_points_world_frame, model_points_register, given_init_pose=best_tsf_guess,
        #                             batch=B, normal_scale=normal_scale, A_normals=model_normals_world_frame,
        #                             B_normals=model_normals_register)
        T, distances = icp.icp_pytorch3d(model_points_world_frame, model_points_register,
                                         given_init_pose=best_tsf_guess, batch=B)

        errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, vis_obj_id, distances,
                                                     viewing_delay)
        errors.append(np.mean(errors_per_batch))

    for num, err in zip(num_points_list, errors):
        cache[name][seed][num] = err
    torch.save(cache, fullname)
    for i in range(len(num_points_list)):
        print(f"num {num_points_list[i]} err {errors[i]}")


def test_icp_on_experiment_run(target_obj_id, vis_obj_id, vis: Visualizer, seed=0, viewing_delay=0.1,
                               register_num_points=500, eval_num_points=200,
                               normal_scale=0.05, upto_index=-1, upright_bias=0.1,
                               model_name="mustard_normal", run_name=""):
    name = f"{model_name} {run_name}".strip()
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    cache = torch.load(fullname)
    data = cache[name][seed]

    for _ in range(1000):
        p.stepSimulation()

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    best_tsf_guess = None if upright_bias == 0 else exploration.random_upright_transforms(B, dtype, device)

    # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
    pose = p.getBasePositionAndOrientation(target_obj_id)
    link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points_world_frame = data['xs'][:upto_index]
    model_normals_world_frame = data['df'][:upto_index]
    model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)

    current_to_link_tf = link_to_current_tf_gt.inverse()
    model_points = current_to_link_tf.transform_points(model_points_world_frame)
    model_normals = current_to_link_tf.transform_normals(model_normals_world_frame)
    i = 0
    for i, pt in enumerate(model_points_world_frame):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
    vis.clear_visualization_after("mpt", i + 1)
    vis.clear_visualization_after("mn", i + 1)

    rand.seed(seed)
    # perform ICP and visualize the transformed points
    # -- try out pytorch3d
    T, distances = icp.icp_pytorch3d(model_points_world_frame, model_points_register, given_init_pose=best_tsf_guess,
                                     batch=B)

    errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, vis_obj_id, distances,
                                                 viewing_delay)


def evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, vis_obj_id, distances, viewing_delay):
    # due to inherent symmetry, can't just use the known correspondence to measure error, since it's ok to mirror
    # we're essentially measuring the chamfer distance (acts on 2 point clouds), where one point cloud is the
    # evaluation model points on the ground truth object surface, and the surface points of the object transformed
    # by our estimated pose (which is infinitely dense)
    # this is the unidirectional chamfer distance since we're only measuring dist of eval points to surface
    B = T.shape[0]
    eval_num_points = model_points_world_frame_eval.shape[0]

    chamfer_distance = torch.zeros(B, eval_num_points)
    link_to_world = tf.Transform3d(matrix=T.inverse())
    m = link_to_world.get_matrix()
    errors_per_batch = []
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
        if distances is not None:
            vis.draw_point("dist", (0, 0, 0.2), (1, 0, 0), label=f"dist: {distances[b].mean().item():.5f}")
        time.sleep(viewing_delay)

        errors_per_transform = chamfer_distance.mean(dim=-1)
        errors_per_batch.append(errors_per_transform.mean())
    return errors_per_batch


def marginalize_over_suffix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[:suffix_start_idx - 1] if suffix_start_idx > 0 else "base"


def marginalize_over_prefix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[suffix_start_idx:] if suffix_start_idx > 0 else name


def marginalize_over_registration_num(name):
    registration_num = re.search(r"\d+", name)
    return f"{registration_num[0]} registered points" if registration_num is not None else name


def plot_icp_results(names_to_include=None, logy=True, marginalize_over_name=None):
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison.pkl')
    cache = torch.load(fullname)

    fig, axs = plt.subplots(1, 1, sharex="col", figsize=(8, 8), constrained_layout=True)
    if logy:
        axs.set_yscale('log')

    to_plot = {}
    for name in cache.keys():
        to_plot_name = marginalize_over_name(name) if marginalize_over_name is not None else name
        if names_to_include is not None and not names_to_include(name):
            print(f"ignored {name}")
            continue

        for seed in cache[name]:
            data = cache[name][seed]
            # short by num points
            a, b = zip(*sorted(data.items(), key=lambda e: e[0]))

            if to_plot_name not in to_plot:
                to_plot[to_plot_name] = a, []
            to_plot[to_plot_name][1].append(b)

    # sort by name
    to_plot = dict(sorted(to_plot.items()))
    for name, data in to_plot.items():
        x = data[0]
        errors = data[1]
        # assume all the num errors are the same
        # convert to cm^2 (saved as mm^2, so divide by 10^2
        errors = np.stack(errors) / 100
        mean = errors.mean(axis=0)
        low = np.percentile(errors, 20, axis=0)
        high = np.percentile(errors, 80, axis=0)
        std = errors.std(axis=0)
        axs.plot(x, mean, label=name)
        # axs.errorbar(x, mean, std, label=name)
        axs.fill_between(x, low, high, alpha=0.2)

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


def make_sphere_preconfig(z):
    objId = make_sphere(z, [0., 0, z])
    ranges = np.array([[-.2, .2], [-.2, .2], [-.1, .4]])
    return objId, ranges


class YCBObjectFactory(PybulletObjectFactory):
    def __init__(self, name, ycb_name, vis_frame_pos=(0, 0, 0), vis_frame_rot=(0, 0, 0, 1), **kwargs):
        super(YCBObjectFactory, self).__init__(name, **kwargs)
        self.ycb_name = ycb_name
        self.vis_frame_pos = vis_frame_pos
        self.vis_frame_rot = vis_frame_rot

    def make_collision_obj(self, z, rgba=None):
        obj_id = p.loadURDF(os.path.join(ycb_objects.getDataPath(), self.ycb_name, "model.urdf"),
                            [0., 0., z * 3],
                            p.getQuaternionFromEuler([0, 0, -1]), globalScaling=self.scale)
        ranges = np.array([[-.15, .15], [-.15, .15], [-0.05, .5]]) * self.scale
        if rgba is not None:
            p.changeVisualShape(obj_id, -1, rgbaColor=rgba)
        return obj_id, ranges

    def make_visual_obj(self, visual_shape_id=None, pos=(0, 0, 0), rgba=(0, 0.8, 0.2, 0.2)):
        if visual_shape_id is None:
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                  fileName=os.path.join(ycb_objects.getDataPath(), self.ycb_name,
                                                                        "textured_simple_reoriented.obj"),
                                                  rgbaColor=rgba, meshScale=[self.scale, self.scale, self.scale],
                                                  visualFrameOrientation=self.vis_frame_rot,
                                                  visualFramePosition=self.vis_frame_pos)
        obj_id = p.createMultiBody(baseMass=0, basePosition=pos, baseVisualShapeIndex=visual_shape_id)
        return visual_shape_id, obj_id


class ShapeExplorationExperiment(abc.ABC):
    LINK_FRAME_POS = [0, 0, 0]
    LINK_FRAME_ORIENTATION = [0, 0, 0, 1]

    def __init__(self, obj_factory=YCBObjectFactory("mustard_normal", "YcbMustardBottle",
                                                    vis_frame_rot=p.getQuaternionFromEuler([0, 0, 1.57 - 0.1]),
                                                    vis_frame_pos=[-0.014, -0.0125, 0.04]),
                 eval_period=10,
                 plot_per_eval_period=1,
                 plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS):
        self.policy: typing.Optional[ShapeExplorationPolicy] = None
        self.obj_factory = obj_factory
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
        self.dd.set_camera_position([0., 0.3], yaw=0, pitch=-30)

        # log video
        self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                              "{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        self.z = 0.1
        self.objId, self.ranges = self.obj_factory.make_collision_obj(self.z)
        # also needs to be collision since we will test collision against it to get distance
        self.visId, _ = self.obj_factory.make_collision_obj(self.z, rgba=[0.2, 0.2, 1.0, 0.5])
        p.resetBasePositionAndOrientation(self.visId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.visId, -1, mass=0)
        p.setCollisionFilterPair(self.objId, self.visId, -1, -1, 0)

        self.has_run = False

    def set_policy(self, policy: ShapeExplorationPolicy):
        self.policy = policy

    @abc.abstractmethod
    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        """Evaluate errors and append to given lists"""

    def run(self, seed=0, timesteps=202, build_model=False, clean_cache=False,
            model_name="mustard_normal", run_name=""):
        target_obj_id = self.objId
        vis = self.dd
        name = f"{model_name} {run_name}".strip()

        # wait for it to settle
        for _ in range(1000):
            p.stepSimulation()

        # these are in object frame (aligned with [0,0,0], [0,0,0,1]
        model_points, model_normals, _ = sample_model_points(target_obj_id, num_points=500, force_z=None,
                                                             mid_z=0.05,
                                                             seed=0, clean_cache=build_model,
                                                             random_sample_sigma=0.2,
                                                             name=model_name, vis=vis,
                                                             restricted_points=restricted_pts[model_name])
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
        n = torch.tensor(surface_normal_at_point(self.objId, x))
        self.dd.draw_2d_line('n', x, -n, (0, 0.5, 0), scale=0.2)

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
            n = torch.tensor(surface_normal_at_point(self.objId, x))
            self.dd.draw_2d_line('n', x, -n, (0, 0.5, 0), scale=0.2)

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
                                         'xs': torch.stack(xs),
                                         'df': torch.stack(df),
                                         'error_at_model_points': error_at_model_points,
                                         'error_at_rep_surface': error_at_rep_surface}
                torch.save(cache, fullname)

        torch.save(cache, fullname)
        self.has_run = True
        self.dd.clear_visualizations()
        return error_at_model_points


class ICPEVExperiment(ShapeExplorationExperiment):
    def __init__(self, policy_factory=ICPEVExplorationPolicy, policy_args=None, **kwargs):
        if policy_args is None:
            policy_args = {}
        super(ICPEVExperiment, self).__init__(**kwargs)

        # test object needs collision shape to test against, so we can't use visual only object
        self.testObjId, _ = self.obj_factory.make_collision_obj(self.z)
        p.resetBasePositionAndOrientation(self.testObjId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.testObjId, -1, mass=0)
        p.changeVisualShape(self.testObjId, -1, rgbaColor=[0, 0, 0, 0])
        p.setCollisionFilterPair(self.objId, self.testObjId, -1, -1, 0)

        obj_frame_sdf = exploration.PyBulletNaiveSDF(self.testObjId, vis=self.dd)
        self.set_policy(policy_factory(obj_frame_sdf, vis=self.dd, debug_obj_factory=self.obj_factory, **policy_args))

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
        self.meshId = None
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
            imprint_norm = matplotlib.colors.Normalize(vmin=0, vmax=torch.quantile(var, .90).item())
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


class Means:
    HARMONIC = 0
    ARITHMETIC = 1


def plot_exploration_results(names_to_include=None, logy=False, marginalize_over_name=None, used_mean=Means.ARITHMETIC):
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    cache = torch.load(fullname)

    fig, axs = plt.subplots(3, 1, sharex="col", figsize=(8, 8), constrained_layout=True)

    if logy:
        for ax in axs:
            ax.set_yscale('log')

    to_plot = {}
    for name in cache.keys():
        to_plot_name = marginalize_over_name(name) if marginalize_over_name is not None else name
        if names_to_include is not None and not names_to_include(name):
            print(f"ignored {name}")
            continue

        for seed in cache[name]:
            data = cache[name][seed]
            # data = sorted(data.items(), key=lambda e: e[0])

            error_t = data['t']
            error_at_model_points = data['error_at_model_points']
            # for backward compatibility
            error_at_rep_surface = data.get('error_at_gp_surface', None)
            if error_at_rep_surface is None:
                error_at_rep_surface = data.get('error_at_rep_surface', None)

            if to_plot_name not in to_plot:
                to_plot[to_plot_name] = error_t, [], []
            to_plot[to_plot_name][1].append(error_at_model_points)
            to_plot[to_plot_name][2].append(error_at_rep_surface)

    # sort by name
    to_plot = dict(sorted(to_plot.items()))
    for name, data in to_plot.items():
        x = data[0]
        errors_at_model = data[1]
        errors_at_rep = data[2]

        try:
            avg_err = torch.stack([torch.tensor(errors_at_rep), torch.tensor(errors_at_model)])
        except RuntimeError as e:
            print(f"Skipping {name} due to {e}")
            continue

        if used_mean is Means.HARMONIC:
            avg_err = 1 / avg_err
            avg_err = 2 / (avg_err.sum(dim=0))
        elif used_mean is Means.ARITHMETIC:
            avg_err = avg_err.mean(dim=0)

        for i, errors in enumerate([errors_at_model, errors_at_rep, avg_err]):
            # assume all the num errors are the same
            # convert to cm^2 (saved as m^2, so multiply by 100^2)
            errors = np.stack(errors)
            mean = errors.mean(axis=0)
            median = np.median(errors, axis=0)
            std = errors.std(axis=0)
            axs[i].plot(x, median, label=name)

            low = np.percentile(errors, 20, axis=0)
            high = np.percentile(errors, 80, axis=0)
            # axs.errorbar(x, mean, std, label=name)
            axs[i].fill_between(x, low, high, alpha=0.2)

            # print each numerically
            for i in range(len(mean)):
                print(f"{name} {x[i]:>4} : {mean[i]:.2f} ({std[i]:.2f})")
            print()

    axs[0].set_ylabel('error at model points')
    axs[1].set_ylabel('error at gp surface')
    axs[2].set_ylabel('average error')
    axs[-1].set_xlabel('step')
    axs[-1].legend()
    if not logy:
        axs[0].set_ylim(bottom=0)
        axs[1].set_ylim(bottom=0)
        axs[2].set_ylim(bottom=0)
    plt.show()


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

    # -- Build object models (sample points from their surface)
    # experiment = ICPEVExperiment()
    # for num_points in (5, 10, 20, 30, 40, 50, 100):
    #     for seed in range(10):
    #         build_model(experiment.objId, experiment.dd, "mustard_normal", seed=seed, num_points=num_points,
    #                     pause_at_end=False)

    # -- ICP experiment
    # experiment = ICPEVExperiment()
    # for normal_weight in [0.05]:
    #     for gt_num in [500]:
    #         for seed in range(10):
    #             test_icp(experiment.objId, experiment.visId, experiment.dd, seed=seed, register_num_points=gt_num,
    #                      name=f"pytorch3d icp {gt_num} mp", viewing_delay=0, num_points_list=[100],
    #                      normal_scale=normal_weight)
    # plot_icp_results(names_to_include=lambda name: not name.startswith("point2plane"))

    # -- exploration experiment
    policy_args = {"upright_bias": 0.1, "debug": True, "num_samples_each_action": 200}
    exp_name = "gp_var"
    # experiment = ICPEVExperiment()
    # test_icp_on_experiment_run(experiment.objId, experiment.visId, experiment.dd, seed=2, upto_index=50,
    #                            register_num_points=500,
    #                            run_name=exp_name, viewing_delay=0.5)
    # experiment = ICPEVExperiment(exploration.ICPEVExplorationPolicy, plot_point_type=PlotPointType.NONE,
    #                              policy_args=policy_args)
    # experiment = ICPEVExperiment(exploration.ICPEVSampleModelPointsPolicy, plot_point_type=PlotPointType.NONE,
    #                              policy_args=policy_args)
    # experiment = ICPEVExperiment(exploration.ICPEVSampleRandomPointsPolicy, plot_point_type=PlotPointType.NONE,
    #                              policy_args=policy_args)
    experiment = GPVarianceExperiment(GPVarianceExploration(), plot_point_type=PlotPointType.NONE)
    for seed in range(10):
        experiment.run(run_name=exp_name, seed=seed)
    # plot_exploration_results(names_to_include=lambda
    #     name: "no_upright_prior" in name or "var_upright_prior_sample" in name or "reachability" in name or "no_normal" in name or "pytorch3d" in name)
    # plot_exploration_results(names_to_include=lambda name: "pytorch3d" in name and "sample" in name or "gp_var" in name, logy=False)
    # experiment.run(run_name="gp_var")
