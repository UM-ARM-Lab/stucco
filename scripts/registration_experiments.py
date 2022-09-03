import abc
import typing
import re
import copy

import argparse
import matplotlib
import numpy as np
import pybullet_data
import pymeshlab
import pytorch_kinematics
from pytorch_kinematics import transforms as tf
from sklearn.cluster import Birch, DBSCAN, KMeans

import stucco.exploration
import stucco.util
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
from torchmcubes import marching_cubes

from stucco import cfg, icp, exploration, util
from stucco.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from stucco.defines import NO_CONTACT_ID
from stucco.env import poke
from stucco.env.env import InfoKeys
from stucco.env.poke import YCBObjectFactory
from stucco.env.pybullet_env import make_sphere, closest_point_on_surface, ContactInfo, \
    surface_normal_at_point
from stucco import exploration
from stucco.env.real_env import CombinedVisualizer
from stucco.env_getters.poke import PokeGetter
from stucco.evaluation import evaluate_chamfer_distance, clustering_metrics, compute_contact_error
from stucco.exploration import PlotPointType, ShapeExplorationPolicy, ICPEVExplorationPolicy, GPVarianceExploration
from stucco.icp import costs as icp_costs
from stucco import util

from stucco.retrieval_controller import sample_model_points, TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, PHDFilterTrackingMethod

# try:
#     import rospy
#
#     rospy.init_node("info_retrieval", log_level=rospy.INFO)
#     # without this we get not logging from the library
#     import importlib
#     importlib.reload(logging)
# except RuntimeError as e:
#     print("Proceeding without ROS: {}".format(e))

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


def build_model(target_obj_id, vis, model_name, seed, num_points, pause_at_end=False, device="cpu"):
    points, normals, _ = sample_model_points(target_obj_id, reject_too_close=0.006,
                                             num_points=num_points,
                                             force_z=None,
                                             mid_z=0.05,
                                             seed=seed, clean_cache=True,
                                             random_sample_sigma=0.2,
                                             name=model_name, vis=None,
                                             device=device,
                                             other_rejection_criteria=reject_model_pts[model_name],
                                             restricted_points=restricted_pts[model_name])
    for i, pt in enumerate(points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -normals[i], color=(0, 0, 0), size=2., scale=0.03)

    print(f"finished building {model_name} {seed} {num_points}")
    if pause_at_end:
        input("paused for inspection")
    vis.clear_visualizations()


def build_model_poke(env: poke.PokeEnv, seed, num_points, pause_at_end=False, device="cpu"):
    return build_model(env.target_object_id, env.vis, env.obj_factory.name, seed, num_points, pause_at_end=pause_at_end,
                       device=device)


def do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B, volumetric_cost, reg_method):
    # perform ICP and visualize the transformed points
    # compare not against current model points (which may be few), but against the maximum number of model points
    if reg_method == icp.ICPMethod.ICP:
        T, distances = icp.icp_pytorch3d(model_points_world_frame, model_points_register,
                                         given_init_pose=best_tsf_guess, batch=B)
    elif reg_method == icp.ICPMethod.ICP_SGD:
        T, distances = icp.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                             given_init_pose=best_tsf_guess, batch=B, learn_translation=True,
                                             use_matching_loss=True)
    # use only volumetric loss
    elif reg_method == icp.ICPMethod.ICP_SGD_VOLUMETRIC_NO_ALIGNMENT:
        T, distances = icp.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                             given_init_pose=best_tsf_guess, batch=B, pose_cost=volumetric_cost,
                                             max_iterations=20, lr=0.01,
                                             learn_translation=True,
                                             use_matching_loss=False)
    elif reg_method in [icp.ICPMethod.VOLUMETRIC, icp.ICPMethod.VOLUMETRIC_NO_FREESPACE]:
        if reg_method == icp.ICPMethod.VOLUMETRIC_NO_FREESPACE:
            volumetric_cost = copy.copy(volumetric_cost)
            volumetric_cost.scale_known_freespace = 0
        T, distances = icp.icp_volumetric(volumetric_cost, model_points_world_frame,
                                          given_init_pose=best_tsf_guess.inverse(),
                                          batch=B, max_iterations=20, lr=0.01)
    else:
        raise RuntimeError(f"Unsupported ICP method {reg_method}")
    # T, distances = icp.icp_mpc(model_points_world_frame, model_points_register,
    #                            icp_costs.ICPPoseCostMatrixInputWrapper(volumetric_cost),
    #                            given_init_pose=best_tsf_guess, batch=B, draw_mesh=exp.draw_mesh)

    # T, distances = icp.icp_stein(model_points_world_frame, model_points_register, given_init_pose=T.inverse(),
    #                              batch=B)
    return T, distances


def test_icp(exp, seed=0, name="", clean_cache=False, viewing_delay=0.3,
             register_num_points=500, eval_num_points=200, num_points_list=(5, 10, 20, 30, 40, 50, 100),
             num_freespace=0,
             freespace_on_one_side=True,
             surface_delta=0.025,
             freespace_cost_scale=1,
             ground_truth_initialization=False,
             icp_method=icp.ICPMethod.VOLUMETRIC,
             debug=False,
             model_name="mustard_normal"):
    obj_name = exp.obj_factory.name
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison_{obj_name}.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
        if seed not in cache[name] or clean_cache:
            cache[name][seed] = {}
    else:
        cache = {name: {seed: {}}}

    target_obj_id = exp.objId
    vis_obj_id = exp.visId
    vis = exp.dd
    freespace_ranges = exp.ranges

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                   device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0, device=exp.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)

    for num_points in num_points_list:
        # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
        model_points, model_normals, _ = sample_model_points(num_points=num_points, name=model_name, seed=seed,
                                                             device=exp.device)

        pose = p.getBasePositionAndOrientation(target_obj_id)
        link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
        model_points_world_frame = link_to_current_tf_gt.transform_points(model_points)
        model_normals_world_frame = link_to_current_tf_gt.transform_normals(model_normals)
        model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)
        model_normals_world_frame_eval = link_to_current_tf_gt.transform_points(model_normals_eval)

        i = 0
        for i, pt in enumerate(model_points_world_frame):
            vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
            vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
        vis.clear_visualization_after("mpt", i + 1)
        vis.clear_visualization_after("mn", i + 1)

        free_voxels = util.VoxelGrid(0.025, freespace_ranges, dtype=dtype, device=device)
        known_sdf = util.VoxelSet(model_points_world_frame,
                                  torch.zeros(model_points_world_frame.shape[0], dtype=dtype, device=device))
        volumetric_cost = icp_costs.VolumetricCost(free_voxels, known_sdf, exp.sdf, scale=1,
                                                   scale_known_freespace=freespace_cost_scale,
                                                   vis=vis, debug=debug)

        # sample points in freespace and plot them
        # sample only on one side
        if freespace_on_one_side:
            used_model_points = model_points_eval[:, 0] > 0
        else:
            used_model_points = model_points_eval[:, 0] > -10
        # extrude model points that are on the surface of the object along their normal vector
        free_space_world_frame_points = model_points_world_frame_eval[used_model_points][:num_freespace] - \
                                        model_normals_world_frame_eval[used_model_points][
                                        :num_freespace] * surface_delta
        free_voxels[free_space_world_frame_points] = 1

        i = 0
        for i, pt in enumerate(free_space_world_frame_points):
            vis.draw_point(f"fspt.{i}", pt, color=(1, 0, 1), scale=2, length=0.003)
        vis.clear_visualization_after("fspt", i + 1)

        rand.seed(seed)

        # initialize at the actual transform to compare global minima
        if ground_truth_initialization:
            best_tsf_guess = link_to_current_tf_gt.get_matrix().repeat(B, 1, 1)
        # perform ICP and visualize the transformed points
        # compare not against current model points (which may be few), but against the maximum number of model points
        T, distances = do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B,
                                       volumetric_cost,
                                       icp_method)

        errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, vis_obj_id, distances,
                                                     viewing_delay)
        errors.append(errors_per_batch)

    for num, err in zip(num_points_list, errors):
        cache[name][seed][num] = err
    torch.save(cache, fullname)
    for i in range(len(num_points_list)):
        print(f"num {num_points_list[i]} err {errors[i]}")


def test_icp_freespace(exp,
                       seed=0, name="", clean_cache=False,
                       viewing_delay=0.3,
                       register_num_points=500, eval_num_points=200, num_points=10,
                       # number of known contact points
                       num_freespace_points_list=(0, 10, 20, 30, 40, 50, 100),
                       freespace_on_one_side=True,
                       surface_delta=0.025,
                       freespace_cost_scale=1,
                       ground_truth_initialization=False,
                       icp_method=icp.ICPMethod.VOLUMETRIC,
                       model_name="mustard_normal"):
    obj_name = exp.obj_factory.name
    fullname = os.path.join(cfg.DATA_DIR, f'icp_freespace_{obj_name}.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
        if seed not in cache[name] or clean_cache:
            cache[name][seed] = {}
    else:
        cache = {name: {seed: {}}}

    target_obj_id = exp.objId
    vis_obj_id = exp.visId
    vis = exp.dd
    freespace_ranges = exp.ranges

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                   device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0, device=exp.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
    model_points, model_normals, _ = sample_model_points(num_points=num_points, name=model_name, seed=seed,
                                                         device=exp.device)

    pose = p.getBasePositionAndOrientation(target_obj_id)
    link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points_world_frame = link_to_current_tf_gt.transform_points(model_points)
    model_normals_world_frame = link_to_current_tf_gt.transform_normals(model_normals)
    model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)
    model_normals_world_frame_eval = link_to_current_tf_gt.transform_normals(model_normals_eval)

    i = 0
    for i, pt in enumerate(model_points_world_frame):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
    vis.clear_visualization_after("mpt", i + 1)
    vis.clear_visualization_after("mn", i + 1)

    rand.seed(seed)
    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)
    # best_tsf_guess = None
    for num_freespace in num_freespace_points_list:
        free_voxels = util.VoxelGrid(0.015, freespace_ranges, dtype=dtype, device=device)
        known_sdf = util.VoxelSet(model_points_world_frame,
                                  torch.zeros(model_points_world_frame.shape[0], dtype=dtype, device=device))
        volumetric_cost = icp_costs.VolumetricCost(free_voxels, known_sdf, exp.sdf,
                                                   scale_known_freespace=freespace_cost_scale, vis=vis,
                                                   debug=False, debug_freespace=True)

        # sample points in freespace and plot them
        # sample only on one side
        if freespace_on_one_side:
            used_model_points = model_points_eval[:, 0] > 0
        else:
            used_model_points = model_points_eval[:, 0] > -10
        # extrude model points that are on the surface of the object along their normal vector
        free_space_world_frame_points = model_points_world_frame_eval[used_model_points][:num_freespace] - \
                                        model_normals_world_frame_eval[used_model_points][
                                        :num_freespace] * surface_delta
        free_voxels[free_space_world_frame_points] = 1

        i = 0
        for i, pt in enumerate(free_space_world_frame_points):
            vis.draw_point(f"fspt.{i}", pt, color=(1, 0, 1), scale=2, length=0.003)
        vis.clear_visualization_after("fspt", i + 1)

        rand.seed(seed)
        # perform ICP and visualize the transformed points
        # reverse engineer the transform
        # compare not against current model points (which may be few), but against the maximum number of model points
        if ground_truth_initialization:
            best_tsf_guess = link_to_current_tf_gt.inverse().get_matrix().repeat(B, 1, 1)
        T, distances = do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B,
                                       volumetric_cost,
                                       icp_method)

        # draw all ICP's sample meshes
        exp.policy._clear_cached_tf()
        exp.policy.register_transforms(T, distances)
        exp.policy._debug_icp_distribution(None, None)

        errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, vis_obj_id,
                                                     distances,
                                                     viewing_delay)
        errors.append(errors_per_batch)

    for num, err in zip(num_freespace_points_list, errors):
        cache[name][seed][num] = err
    torch.save(cache, fullname)
    for i in range(len(num_freespace_points_list)):
        print(f"num {num_freespace_points_list[i]} err {errors[i]}")


def test_icp_on_experiment_run(exp, seed=0, viewing_delay=0.1,
                               register_num_points=500, eval_num_points=200,
                               normal_scale=0.05, upto_index=-1, upright_bias=0.1,
                               model_name="mustard_normal", run_name=""):
    name = f"{model_name} {run_name}".strip()
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    cache = torch.load(fullname)
    data = cache[name][seed]

    target_obj_id = exp.objId
    vis_obj_id = exp.visId
    vis = exp.dd
    freespace_ranges = exp.ranges

    for _ in range(1000):
        p.stepSimulation()

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                   device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0, device=exp.device)

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


def marginalize_over_suffix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[:suffix_start_idx - 1] if suffix_start_idx > 0 else "base"


def marginalize_over_prefix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[suffix_start_idx:] if suffix_start_idx > 0 else name


def marginalize_over_registration_num(name):
    registration_num = re.search(r"\d+", name)
    return f"{registration_num[0]} registered points" if registration_num is not None else name


def plot_icp_results(names_to_include=None, logy=True, plot_median=True, marginalize_over_name=None, x_filter=None,
                     leave_out_percentile=20, icp_res_file='icp_comparison.pkl'):
    fullname = os.path.join(cfg.DATA_DIR, icp_res_file)
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
        x = np.array(data[0])
        errors = data[1]
        # assume all the num errors are the same
        # convert to cm^2 (saved as mm^2, so divide by 10^2
        errors = np.stack(errors) / 100

        # expect [seed, x, batch]
        if len(errors.shape) < 3:
            logger.warning("data for %s is less than 3 dimensional; probably outdated", name)

        if x_filter is not None:
            to_keep = x_filter(x)
            x = x[to_keep]
            errors = errors[:, to_keep]

        if leave_out_percentile > 0:
            remove_threshold = np.percentile(errors, 100 - leave_out_percentile, axis=-1)
            to_keep = errors < remove_threshold[:, :, None]
            # this is equivalent to the below; can uncomment to check
            errors = errors[to_keep].reshape(errors.shape[0], errors.shape[1], -1)
            # t1 = []
            # for i in range(to_keep.shape[0]):
            #     t2 = []
            #     for j in range(to_keep.shape[1]):
            #         t2.append(errors[i, j, to_keep[i, j]])
            #     t1.append(np.stack(t2))
            # errors = np.stack(t1)

        # transpose to get [x, seed, ...]
        errors = errors.transpose(1, 0, 2)
        # flatten the other dimensions for plotting
        errors = errors.reshape(errors.shape[0], -1)

        mean = errors.mean(axis=1)
        median = np.median(errors, axis=1)
        low = np.percentile(errors, 20, axis=1)
        high = np.percentile(errors, 80, axis=1)
        std = errors.std(axis=1)

        if plot_median:
            axs.plot(x, median, label=name)
        else:
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


class ShapeExplorationExperiment(abc.ABC):
    LINK_FRAME_POS = [0, 0, 0]
    LINK_FRAME_ORIENTATION = [0, 0, 0, 1]

    def __init__(self, obj_factory=YCBObjectFactory("mustard_normal", "YcbMustardBottle",
                                                    vis_frame_rot=p.getQuaternionFromEuler([0, 0, 1.57 - 0.1]),
                                                    vis_frame_pos=[-0.014, -0.0125, 0.04]),
                 device="cpu",
                 gui=True,
                 eval_period=10,
                 plot_per_eval_period=1,
                 plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS):
        self.device = device
        self.policy: typing.Optional[ShapeExplorationPolicy] = None
        self.obj_factory = obj_factory
        self.plot_point_type = plot_point_type
        self.plot_per_eval_period = plot_per_eval_period
        self.eval_period = eval_period

        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -10)
        # p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        self.dd = CombinedVisualizer()
        self.dd.init_sim(0.8, 0.8)
        self.dd.sim.toggle_3d(True)
        self.dd.sim.set_camera_position([0.1, 0.35, 0.15], yaw=-20, pitch=-25)
        # init rviz visualization
        # self.dd.init_ros(world_frame="world")

        # log video
        self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                              "{}.mp4".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))

        self.z = 0.1
        self.objId, self.ranges = self.obj_factory.make_collision_obj(self.z)

        # draw base object (in pybullet will already be there since we loaded the collision shape)
        pose = p.getBasePositionAndOrientation(self.objId)
        self.draw_mesh("base_object", pose, (1.0, 1.0, 0., 0.5))

        # also needs to be collision since we will test collision against it to get distance
        self.visId, _ = self.obj_factory.make_collision_obj(self.z, rgba=[0.2, 0.2, 1.0, 0.5])
        p.resetBasePositionAndOrientation(self.visId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.visId, -1, mass=0)
        p.setCollisionFilterPair(self.objId, self.visId, -1, -1, 0)

        self.has_run = False

    def close(self):
        p.disconnect(self.physics_client)

    def draw_mesh(self, *args, **kwargs):
        return self.obj_factory.draw_mesh(self.dd, *args, **kwargs)

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
                                                             device=self.device,
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
    def __init__(self, policy_factory=ICPEVExplorationPolicy, policy_args=None, sdf_resolution=0.025, **kwargs):
        if policy_args is None:
            policy_args = {}
        super(ICPEVExperiment, self).__init__(**kwargs)

        # test object needs collision shape to test against, so we can't use visual only object
        self.testObjId, range_per_dim = self.obj_factory.make_collision_obj(self.z)
        p.resetBasePositionAndOrientation(self.testObjId, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.changeDynamics(self.testObjId, -1, mass=0)
        p.changeVisualShape(self.testObjId, -1, rgbaColor=[0, 0, 0, 0])
        p.setCollisionFilterPair(self.objId, self.testObjId, -1, -1, 0)

        obj_frame_sdf = stucco.exploration.PyBulletNaiveSDF(self.testObjId, vis=self.dd)
        # inflate the range_per_dim to allow for pose estimates around a single point
        range_per_dim *= 2
        # fix the z dimension since there shouldn't be that much variance across it
        range_per_dim[2] = [-range_per_dim[2, 1] * 0.4, range_per_dim[2, 1] * 0.6]
        self.sdf = stucco.exploration.CachedSDF(self.obj_factory.name, sdf_resolution, range_per_dim,
                                                obj_frame_sdf, device=self.device)
        self.set_policy(
            policy_factory(self.sdf, vis=self.dd, debug_obj_factory=self.obj_factory, **policy_args))

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


def plot_exploration_results(names_to_include=None, logy=False, marginalize_over_name=None, plot_just_avg=True,
                             used_mean=Means.ARITHMETIC):
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    cache = torch.load(fullname)

    fig, axs = plt.subplots(1 if plot_just_avg else 3, 1, sharex="col", figsize=(8, 8), constrained_layout=True)
    if plot_just_avg:
        axs = [axs]

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

        # try stripping out runs - sometimes the experiment terminated prematurely
        try:
            expected_len = len(errors_at_rep[0])
            errors_at_rep = [run for run in errors_at_rep if len(run) == expected_len]
            errors_at_model = [run for run in errors_at_model if len(run) == expected_len]
            avg_err = torch.stack([torch.tensor(errors_at_rep), torch.tensor(errors_at_model)])
        except RuntimeError as e:
            print(f"Skipping {name} due to {e}")
            continue

        if used_mean is Means.HARMONIC:
            avg_err = 1 / avg_err
            avg_err = 2 / (avg_err.sum(dim=0))
        elif used_mean is Means.ARITHMETIC:
            avg_err = avg_err.mean(dim=0)

        for i, errors in enumerate([avg_err] if plot_just_avg else [errors_at_model, errors_at_rep, avg_err]):
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

    if plot_just_avg:
        axs[0].set_ylabel('average error')
    else:
        axs[0].set_ylabel('error at model points')
        axs[1].set_ylabel('error at rep surface')
        axs[2].set_ylabel('average error')
    axs[-1].set_xlabel('step')
    axs[-1].legend()
    if not logy:
        for ax in axs:
            ax.set_ylim(bottom=0)
    plt.show()


def ignore_beyond_distance(threshold):
    def filter(distances):
        d = distances.clone()
        d[d > threshold] = threshold
        return d

    return filter


def run_poke(env: poke.PokeEnv, method: TrackingMethod, reg_method, name="", seed=0, clean_cache=False,
             register_num_points=500,
             eval_num_points=200, ctrl_noise_max=0.005):
    name = reg_method + name
    # [name][seed] to access
    # chamfer_err: T x B number of steps by batch chamfer error
    fullname = os.path.join(cfg.DATA_DIR, f'poking.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
        if seed not in cache[name] or clean_cache:
            cache[name][seed] = {}
    else:
        cache = {name: {seed: {}}}

    predetermined_control = {}

    ctrl = [[1., 0., 0]] * 3
    ctrl += [[-1., 0., 1]] * 2
    ctrl += [[1., 0., 0]] * 3
    ctrl += [[-1., 0., 1]] * 2
    ctrl += [[1., 0., 0]] * 3
    # ctrl += [[0.4, 0.4], [.5, -1]] * 6
    # ctrl += [[-0.2, 1]] * 4
    # ctrl += [[0.3, -0.3], [0.4, 1]] * 4
    # ctrl += [[1., -1]] * 3
    # ctrl += [[1., 0.6], [-0.7, 0.5]] * 4
    # ctrl += [[0., 1]] * 5
    # ctrl += [[1., 0]] * 4
    # ctrl += [[0.4, -1.], [0.4, 0.5]] * 4
    rand.seed(0)
    # noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    # ctrl = np.add(ctrl, noise)
    predetermined_control[poke.Levels.MUSTARD] = ctrl

    ctrl = method.create_controller(predetermined_control[env.level])

    obs = env.reset()

    model_name = env.target_model_name
    # sample_in_order = env.level in [poke.Levels.COFFEE_CAN]
    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                   device=env.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0, device=env.device)

    pose = p.getBasePositionAndOrientation(env.target_object_id)
    link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)
    model_normals_world_frame_eval = link_to_current_tf_gt.transform_points(model_normals_eval)

    info = None
    simTime = 0

    B = 30
    device = env.device
    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)
    best_T = None
    guess_pose = None
    chamfer_err = []

    pt_to_config = poke.ArmPointToConfig(env)

    contact_id = []

    # placeholder for now
    empty_sdf = util.VoxelSet(torch.empty(0), torch.empty(0))
    volumetric_cost = icp_costs.VolumetricCost(env.free_voxels, empty_sdf, env.target_sdf, scale=1,
                                               scale_known_freespace=20,
                                               vis=env.vis, debug=False)

    rand.seed(seed)
    while not ctrl.done():
        best_distance = None
        simTime += 1
        env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

        action = ctrl.command(obs, info)
        method.visualize_contact_points(env)

        if env.contact_detector.in_contact():
            contact_id.append(info[InfoKeys.CONTACT_ID])
        else:
            contact_id.append(NO_CONTACT_ID)

        # note that we update our registration regardless if we're in contact or not
        all_configs = torch.tensor(np.array(ctrl.x_history), dtype=dtype, device=device).view(-1, env.nx)
        dist_per_est_obj = []
        transforms_per_object = []
        rmse_per_object = []
        best_segment_idx = None
        for k, this_pts in enumerate(method):
            if len(this_pts) < 4:
                continue
            # this_pts corresponds to tracked contact points that are segmented together
            this_pts = tensor_utils.ensure_tensor(device, dtype, this_pts)
            volumetric_cost.sdf_voxels = util.VoxelSet(this_pts,
                                                       torch.zeros(this_pts.shape[0], dtype=dtype, device=device))

            T, distances = do_registration(this_pts, model_points_register, best_tsf_guess, B, volumetric_cost,
                                           reg_method)

            transforms_per_object.append(T)
            T = T.inverse()
            score = distances
            best_tsf_index = np.argmin(score)

            # pick object with lowest variance in its translation estimate
            translations = T[:, :2, 2]
            best_tsf_distances = (translations.var(dim=0).sum()).item()

            dist_per_est_obj.append(best_tsf_distances)
            rmse_per_object.append(distances)
            if best_distance is None or best_tsf_distances < best_distance:
                best_distance = best_tsf_distances
                best_tsf_guess = T[best_tsf_index].inverse()
                best_segment_idx = k

        # has at least one contact segment
        if best_segment_idx is not None:
            method.register_transforms(transforms_per_object[best_segment_idx], best_tsf_guess)
            logger.debug(f"err each obj {np.round(dist_per_est_obj, 4)}")
            best_T = best_tsf_guess.inverse()

            # evaluate with chamfer distance
            errors_per_batch = evaluate_chamfer_distance(transforms_per_object[best_segment_idx],
                                                         model_points_world_frame_eval, env.vis, env.testObjId,
                                                         rmse_per_object[best_segment_idx], 0)
            # errors.append(np.mean(errors_per_batch))
            chamfer_err.append(errors_per_batch)
            logger.debug(f"chamfer distance {simTime}: {np.mean(errors_per_batch)}")

            # draw mesh at where our best guess is
            # TODO check if we need to invert this best guess
            guess_pose = util.matrix_to_pos_rot(best_T)
            env.draw_mesh("base_object", guess_pose, (0.0, 1.0, 0., 0.5))
            # TODO save current pose and contact point for playback

        if torch.is_tensor(action):
            action = action.cpu()

        action = np.array(action).flatten()
        obs, rew, done, info = env.step(action)

        if len(chamfer_err) > 0:
            cache[name][seed] = {'chamfer_err': np.stack(chamfer_err), }
            torch.save(cache, fullname)

    # evaluate FMI and contact error here
    labels, moved_points = method.get_labelled_moved_points(np.ones(len(contact_id)) * NO_CONTACT_ID)
    contact_id = np.array(contact_id)

    in_label_contact = contact_id != NO_CONTACT_ID

    m = clustering_metrics(contact_id[in_label_contact], labels[in_label_contact])
    contact_error = compute_contact_error(None, moved_points, env=env, visualize=False)
    cme = np.mean(np.abs(contact_error))

    # grasp_at_pose(env, guess_pose)

    return m, cme


def main_poke(env, method_name, registration_method, seed=0, name=""):
    methods_to_run = {
        'ours': OurSoftTrackingMethod(env, PokeGetter.contact_parameters(env), poke.ArmPointToConfig(env), dim=3),
        'online-birch': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                              inertia_ratio=0.2,
                                              threshold=0.08),
        'online-dbscan': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1),
        'online-kmeans': SklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2, n_clusters=1,
                                               random_state=0),
        'gmphd': PHDFilterTrackingMethod(env, fp_fn_bias=4, q_mag=0.00005, r_mag=0.00005, birth=0.001, detection=0.3)
    }
    env.draw_user_text(f"{method_name} seed {seed}", xy=[-0.1, 0.28, -0.5])
    return run_poke(env, methods_to_run[method_name], registration_method, seed=seed, name=name)


def experiment_ground_truth_initialization_for_global_minima_comparison(obj_factory, plot_only=False, gui=True):
    # -- Ground truth initialization experiment
    if not plot_only:
        experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
        for seed in range(10):
            test_icp(experiment, seed=seed, register_num_points=500,
                     num_freespace=0,
                     name=f"gt init pytorch3d",
                     icp_method=icp.ICPMethod.ICP,
                     ground_truth_initialization=True,
                     viewing_delay=0)
        experiment.close()

        for sdf_resolution in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, sdf_resolution=sdf_resolution, gui=gui)
            for seed in range(10):
                test_icp(experiment, seed=seed, register_num_points=500,
                         num_freespace=0,
                         name=f"gt init volumetric sdf res {sdf_resolution}",
                         icp_method=icp.ICPMethod.VOLUMETRIC,
                         ground_truth_initialization=True,
                         viewing_delay=0)
            experiment.close()
        for sdf_resolution in [0.025]:
            experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, sdf_resolution=sdf_resolution, gui=gui)
            for seed in range(10):
                test_icp(experiment, seed=seed, register_num_points=500,
                         num_freespace=100,
                         surface_delta=0.01,
                         freespace_on_one_side=False,
                         freespace_cost_scale=20,
                         name=f"gt init volumetric freespace sdf res {sdf_resolution}",
                         icp_method=icp.ICPMethod.VOLUMETRIC,
                         ground_truth_initialization=True,
                         viewing_delay=0)
            experiment.close()
    file = f"icp_comparison_{obj_factory.name}.pkl"
    plot_icp_results(icp_res_file=file, names_to_include=lambda name: name.startswith("gt init") and "0.025" in name)
    plot_icp_results(icp_res_file=file, names_to_include=lambda name: name.startswith("gt init") and "0.025" in name,
                     x_filter=lambda x: x < 40)


def experiment_vary_num_points_and_num_freespace(obj_factory, plot_only=False, gui=True):
    # -- Differing number of freespace experiment while varying number of known points
    if not plot_only:
        for surface_delta in [0.025, 0.05]:
            for num_freespace in (0, 10, 20, 30, 40, 50, 100):
                experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
                for seed in range(10):
                    test_icp(experiment, seed=seed, register_num_points=500,
                             # num_points_list=(50,),
                             num_freespace=num_freespace,
                             freespace_cost_scale=20,
                             surface_delta=surface_delta,
                             name=f"volumetric fixed sdf free pts {num_freespace} delta {surface_delta}",
                             icp_method=icp.ICPMethod.VOLUMETRIC,
                             viewing_delay=0)
                experiment.close()
    file = f"icp_comparison_{obj_factory.name}.pkl"
    # TODO adjust the plotter here
    plot_icp_results(icp_res_file=file, names_to_include=lambda
        name: "volumetric fixed sdf free pts 30" in name or name == "volumetric fixed sdf free pts 0 delta 0.025")
    plot_icp_results(icp_res_file=file, names_to_include=lambda
        name: name == "volumetric fixed sdf free pts 0 delta 0.025" or "rerun" in name)


def experiment_vary_num_freespace(obj_factory, plot_only=False, gui=True):
    # -- Freespace ICP experiment
    if not plot_only:
        # test_gradients(experiment)
        for surface_delta in [0.01, 0.025, 0.05]:
            for freespace_cost_scale in [1, 5, 20]:
                experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
                for num_points in [5]:
                    for seed in range(10):
                        test_icp_freespace(experiment, seed=seed, num_points=num_points,
                                           # num_freespace_points_list=(0, 50, 100),
                                           register_num_points=500,
                                           surface_delta=surface_delta,
                                           freespace_cost_scale=freespace_cost_scale,
                                           name=f"volumetric {num_points}np delta {surface_delta} scale {freespace_cost_scale}",
                                           viewing_delay=0)
                experiment.close()
        for surface_delta in [0.01, 0.025, 0.05]:
            for freespace_cost_scale in [1, 5, 20]:
                experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
                for num_points in [5]:
                    for seed in range(10):
                        test_icp_freespace(experiment, seed=seed, num_points=num_points,
                                           # num_freespace_points_list=(0, 50, 100),
                                           register_num_points=500,
                                           surface_delta=surface_delta,
                                           freespace_cost_scale=freespace_cost_scale,
                                           freespace_on_one_side=False,
                                           name=f"volumetric {num_points}np all sides delta {surface_delta} scale {freespace_cost_scale}",
                                           viewing_delay=0)
                experiment.close()
    file = f"icp_freespace_{obj_factory.name}.pkl"
    plot_icp_results(icp_res_file=file,
                     names_to_include=lambda name: "volumetric 5np delta 0.01" in name)


parser = argparse.ArgumentParser(description='Object registration from contact')
parser.add_argument('experiment',
                    choices=['build', 'globalmin', 'random-sample', 'freespace', 'poke', 'debug'],
                    help='which experiment to run')
registration_map = {
    "volumetric": icp.ICPMethod.VOLUMETRIC,
    "volumetric-no-freespace": icp.ICPMethod.VOLUMETRIC_NO_FREESPACE,
    "icp": icp.ICPMethod.ICP,
    "icp-sgd": icp.ICPMethod.ICP_SGD,
    "icp-sgd-no-alignment": icp.ICPMethod.ICP_SGD_VOLUMETRIC_NO_ALIGNMENT
}
parser.add_argument('registration',
                    choices=registration_map.keys(),
                    help='which registration method to run')
parser.add_argument('--tracking',
                    choices=['ours', 'online-birch', 'online-dbscan', 'online-kmeans', 'gmphd'],
                    default='ours',
                    help='which tracking method to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
# run parameters
task_map = {"mustard_normal": poke.Levels.MUSTARD, "coffee": poke.Levels.COFFEE_CAN, "cracker": poke.Levels.CRACKER}
parser.add_argument('--task', default="mustard_normal", choices=task_map.keys(), help='what task to run')
parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
parser.add_argument('--plot_only', action='store_true',
                    help='plot only (previous) results without running any experiments')

obj_factory_map = {
    "mustard_normal": YCBObjectFactory("mustard_normal", "YcbMustardBottle",
                                       vis_frame_rot=p.getQuaternionFromEuler([0, 0, 1.57 - 0.1]),
                                       vis_frame_pos=[-0.014, -0.0125, 0.04]),
    # TODO create the other object factories
}

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    tracking_method_name = args.tracking
    registration_method = registration_map[args.registration]
    obj_factory = obj_factory_map[args.task]

    # -- Build object models (sample points from their surface)
    if args.experiment == "build":
        experiment = ICPEVExperiment(obj_factory=obj_factory)
        # for num_points in (5, 10, 20, 30, 40, 50, 100):
        for num_points in (300, 400, 500):
            for seed in range(10):
                build_model(experiment.objId, experiment.dd, args.task, seed=seed, num_points=num_points,
                            pause_at_end=False)

    elif args.experiment == "globalmin":
        experiment_ground_truth_initialization_for_global_minima_comparison(obj_factory, plot_only=args.plot_only,
                                                                            gui=not args.no_gui)
    elif args.experiment == "random-sample":
        experiment_vary_num_points_and_num_freespace(obj_factory, plot_only=args.plot_only, gui=not args.no_gui)
    elif args.experiment == "freespace":
        experiment_vary_num_freespace(obj_factory, plot_only=args.plot_only, gui=not args.no_gui)
    elif args.experiment == "poke":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=False)
        fmis = []
        cmes = []
        # backup video logging in case ffmpeg and nvidia driver are not compatible
        # with WindowRecorder(window_names=("Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build",),
        #                     name_suffix="sim", frame_rate=30.0, save_dir=cfg.VIDEO_DIR):
        for seed in args.seed:
            m, cme = main_poke(env, tracking_method_name, registration_method, seed=seed, name=args.name)
            fmi = m[0]
            fmis.append(fmi)
            cmes.append(cme)
            logger.info(f"{tracking_method_name} fmi {fmi} cme {cme}")
            env.vis.clear_visualizations()
            env.reset()

        logger.info(
            f"{tracking_method_name} mean fmi {np.mean(fmis)} median fmi {np.median(fmis)} std fmi {np.std(fmis)} {fmis}\n"
            f"mean cme {np.mean(cmes)} median cme {np.median(cmes)} std cme {np.std(cmes)} {cmes}")
        env.close()
    elif args.experiment == "debug":
        # -- ICP experiment
        for gt_num in [500]:
            experiment = ICPEVExperiment(device="cuda")
            for seed in range(10):
                test_icp(experiment, seed=seed, register_num_points=gt_num,
                         # num_points_list=(30, 40, 50, 100),
                         num_freespace=0,
                         freespace_cost_scale=20,
                         icp_method=icp.ICPMethod.ICP_SGD,
                         name=f"pytorch3d sgd rerun", viewing_delay=0)
            experiment.close()
        # plot_icp_results(names_to_include=lambda name: "rerun" in name or name == "volumetric free pts 0")

        # -- exploration experiment
        # exp_name = "tukey voxel 0.1"
        # policy_args = {"upright_bias": 0.1, "debug": True, "num_samples_each_action": 200,
        #                "evaluate_icpev_correlation": False, "debug_name": exp_name,
        #                "distance_filter": ignore_beyond_distance(0.}
        # experiment = ICPEVExperiment()
        # test_icp_on_experiment_run(experiment, seed=2, upto_index=50,
        #                            register_num_points=500,
        #                            run_name=exp_name, viewing_delay=0.5)
        # experiment = ICPEVExperiment(exploration.ICPEVExplorationPolicy, plot_point_type=PlotPointType.NONE,
        #                              policy_args=policy_args)
        # experiment = ICPEVExperiment(exploration.ICPEVSampleModelPointsPolicy, plot_point_type=PlotPointType.NONE,
        #                              policy_args=policy_args)
        # experiment = ICPEVExperiment(exploration.ICPEVSampleRandomPointsPolicy, plot_point_type=PlotPointType.NONE,
        #                              policy_args=policy_args)
        # experiment = GPVarianceExperiment(GPVarianceExploration(alpha=0.05), plot_point_type=PlotPointType.NONE)
        # experiment = ICPEVExperiment(exploration.ICPEVVoxelizedPolicy, plot_point_type=PlotPointType.NONE,
        #                              policy_args=policy_args)
        # for seed in range(10):
        #     experiment.run(run_name=exp_name, seed=seed)
        # plot_exploration_results(logy=True, names_to_include=lambda
        #     name: "voxel" in name and "cached" not in name and "icpev" not in name or "tukey" in name)
        # plot_exploration_results(names_to_include=lambda name: "temp" in name or "cache" in name, logy=True)
        # experiment.run(run_name="gp_var")