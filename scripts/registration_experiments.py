import abc
import enum
import itertools
import typing
import re
import copy
import time
import pandas as pd

import argparse
import matplotlib
import numpy as np
import seaborn as sns
import pybullet_data
import pymeshlab
from pytorch_kinematics import transforms as tf
from sklearn.cluster import Birch, DBSCAN, KMeans

import stucco.exploration
import stucco.sdf
import stucco.util
import torch
import pybullet as p
import logging
import os
from datetime import datetime

import gpytorch
import matplotlib.colors, matplotlib.cm
from matplotlib import pyplot as plt
from arm_pytorch_utilities import tensor_utils, rand
from torchmcubes import marching_cubes

from stucco import cfg, icp
from stucco.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from stucco.defines import NO_CONTACT_ID
from stucco.env import poke
from stucco.env.env import InfoKeys
from stucco.env.poke import obj_factory_map, level_to_obj_map
from stucco.env.pybullet_env import closest_point_on_surface, ContactInfo, \
    surface_normal_at_point, draw_AABB
from stucco import exploration
from stucco.env.real_env import CombinedVisualizer
from stucco.env_getters.poke import PokeGetter
from stucco.evaluation import evaluate_chamfer_distance
from stucco.exploration import PlotPointType, ShapeExplorationPolicy, ICPEVExplorationPolicy, GPVarianceExploration
from stucco.icp import costs as icp_costs
from stucco.icp import volumetric
from stucco import util
from stucco.sdf import ObjectFactory

from arm_pytorch_utilities.controller import Controller
from stucco import tracking, detection
from stucco.retrieval_controller import sample_mesh_points, TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, PHDFilterTrackingMethod

plt.switch_backend('Qt5Agg')

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

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def predetermined_poke_range():
    # y,z order of poking
    return {
        poke.Levels.DRILL: ((0, 0.2, 0.3, -0.2, -0.3), (0.05, 0.15, 0.25, 0.325, 0.4, 0.5)),
        poke.Levels.DRILL_OPPOSITE: ((0, 0.2, 0.3, -0.2, -0.3), (0.05, 0.15, 0.25, 0.4, 0.51)),
        poke.Levels.DRILL_SLANTED: ((0, 0.2, 0.3, -0.2, -0.3), (0.05, 0.15, 0.25, 0.4, 0.51)),
        poke.Levels.DRILL_FALLEN: ((0, 0.2, 0.3, -0.2, -0.3), (0.05, 0.18, 0.25, 0.4)),
        poke.Levels.MUSTARD: ((0, 0.18, 0.24, -0.25), (0.05, 0.2, 0.35, 0.52)),
        poke.Levels.MUSTARD_SIDEWAYS: ((0, 0.2, -0.2), (0.05, 0.2, 0.35, 0.52)),
        poke.Levels.MUSTARD_FALLEN: ((0, 0.3, -0.15, -0.36), (0.05, 0.2, 0.35)),
        poke.Levels.MUSTARD_FALLEN_SIDEWAYS: ((0, 0.2, 0.35, -0.2, -0.35), (0.05, 0.12, 0.2)),
        poke.Levels.HAMMER: ((0, -0.2, 0.2, 0.4), (0.05, 0.15, 0.25, 0.4)),
        poke.Levels.HAMMER_1: ((0, 0.15, -0.15), (0.05, 0.1, 0.2, 0.4)),
        poke.Levels.HAMMER_2: ((0, 0.15, 0.4, -0.15), (0.05, 0.15, 0.25)),
    }


def build_model(obj_factory: stucco.sdf.ObjectFactory, vis, model_name, seed, num_points, pause_at_end=False,
                device="cpu"):
    points, normals, _ = sample_mesh_points(obj_factory, num_points=num_points,
                                            seed=seed, clean_cache=True,
                                            name=model_name,
                                            device=device)
    for i, pt in enumerate(points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, normals[i], color=(0, 0, 0), size=2., scale=0.03)

    print(f"finished building {model_name} {seed} {num_points}")
    if pause_at_end:
        input("paused for inspection")
    vis.clear_visualization_after("mpt", 0)
    vis.clear_visualization_after("mn", 0)


def build_model_poke(env: poke.PokeEnv, seed, num_points, pause_at_end=False, device="cpu"):
    return build_model(env.obj_factory, env.vis, env.obj_factory.name, seed, num_points, pause_at_end=pause_at_end,
                       device=device)


def registration_method_uses_only_contact_points(reg_method: icp.ICPMethod):
    if reg_method in [icp.ICPMethod.ICP, icp.ICPMethod.ICP_SGD, icp.ICPMethod.ICP_REVERSE,
                      icp.ICPMethod.ICP_SGD_REVERSE, icp.ICPMethod.VOLUMETRIC_NO_FREESPACE]:
        return True
    return False


def do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B,
                    volumetric_cost: icp_costs.VolumetricCost, reg_method: icp.ICPMethod):
    """Register a set of observed surface points in world frame to an object using some method

    :param model_points_world_frame:
    :param model_points_register:
    :param best_tsf_guess: initial estimate of object frame to world frame
    :param B:
    :param volumetric_cost:
    :param reg_method:
    :return: B x 4 x 4 transform from world frame to object frame, B RMSE for each of the batches
    """
    # perform ICP and visualize the transformed points
    # compare not against current model points (which may be few), but against the maximum number of model points
    if reg_method == icp.ICPMethod.ICP:
        T, distances = icp.icp_pytorch3d(model_points_world_frame, model_points_register,
                                         given_init_pose=best_tsf_guess.inverse(), batch=B)
    elif reg_method == icp.ICPMethod.ICP_REVERSE:
        T, distances = icp.icp_pytorch3d(model_points_register, model_points_world_frame,
                                         given_init_pose=best_tsf_guess, batch=B)
        T = T.inverse()
    elif reg_method == icp.ICPMethod.ICP_SGD:
        T, distances = icp.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                             given_init_pose=best_tsf_guess.inverse(), batch=B, learn_translation=True,
                                             use_matching_loss=True)
    elif reg_method == icp.ICPMethod.ICP_SGD_REVERSE:
        T, distances = icp.icp_pytorch3d_sgd(model_points_register, model_points_world_frame,
                                             given_init_pose=best_tsf_guess, batch=B, learn_translation=True,
                                             use_matching_loss=True)
        T = T.inverse()
    # use only volumetric loss
    elif reg_method == icp.ICPMethod.ICP_SGD_VOLUMETRIC_NO_ALIGNMENT:
        T, distances = icp.icp_pytorch3d_sgd(model_points_world_frame, model_points_register,
                                             given_init_pose=best_tsf_guess.inverse(), batch=B,
                                             pose_cost=volumetric_cost,
                                             max_iterations=20, lr=0.01,
                                             learn_translation=True,
                                             use_matching_loss=False)
    elif reg_method in [icp.ICPMethod.VOLUMETRIC, icp.ICPMethod.VOLUMETRIC_NO_FREESPACE,
                        icp.ICPMethod.VOLUMETRIC_ICP_INIT, icp.ICPMethod.VOLUMETRIC_LIMITED_REINIT,
                        icp.ICPMethod.VOLUMETRIC_LIMITED_REINIT_FULL,
                        icp.ICPMethod.VOLUMETRIC_CMAES, icp.ICPMethod.VOLUMETRIC_CMAME,
                        icp.ICPMethod.VOLUMETRIC_SVGD]:
        if reg_method == icp.ICPMethod.VOLUMETRIC_NO_FREESPACE:
            volumetric_cost = copy.copy(volumetric_cost)
            volumetric_cost.scale_known_freespace = 0
        if reg_method == icp.ICPMethod.VOLUMETRIC_ICP_INIT:
            # try always using the prior
            # best_tsf_guess = exploration.random_upright_transforms(B, model_points_register.dtype,
            #                                                        model_points_register.device)
            T, distances = icp.icp_pytorch3d(model_points_register, model_points_world_frame,
                                             given_init_pose=best_tsf_guess, batch=B)
            best_tsf_guess = T

        optimization = volumetric.Optimization.SGD
        if reg_method == icp.ICPMethod.VOLUMETRIC_CMAES:
            optimization = volumetric.Optimization.CMAES
        elif reg_method == icp.ICPMethod.VOLUMETRIC_CMAME:
            optimization = volumetric.Optimization.CMAME
        elif reg_method == icp.ICPMethod.VOLUMETRIC_SVGD:
            optimization = volumetric.Optimization.SVGD
        # so given_init_pose expects world frame to object frame
        T, distances = icp.icp_volumetric(volumetric_cost, model_points_world_frame, optimization=optimization,
                                          given_init_pose=best_tsf_guess.inverse(), save_loss_plot=False,
                                          batch=B)
    elif reg_method == icp.ICPMethod.MEDIAL_CONSTRAINT:
        T, distances = icp.icp_medial_constraints(volumetric_cost.sdf, volumetric_cost.free_voxels,
                                                  model_points_world_frame,
                                                  given_init_pose=best_tsf_guess,
                                                  batch=B, max_iterations=5, verbose=False,
                                                  # vis=None)
                                                  vis=volumetric_cost.vis)
        T = T.inverse()
    else:
        raise RuntimeError(f"Unsupported ICP method {reg_method}")
    # T, distances = icp.icp_mpc(model_points_world_frame, model_points_register,
    #                            icp_costs.ICPPoseCostMatrixInputWrapper(volumetric_cost),
    #                            given_init_pose=best_tsf_guess, batch=B, draw_mesh=exp.draw_mesh)

    # T, distances = icp.icp_stein(model_points_world_frame, model_points_register, given_init_pose=T.inverse(),
    #                              batch=B)
    return T, distances


def saved_traj_dir_for_method(reg_method: icp.ICPMethod):
    name = reg_method.name.lower().replace('_', '-')
    return os.path.join(cfg.DATA_DIR, f"poke/{name}")


def saved_traj_file(reg_method: icp.ICPMethod, level: poke.Levels, seed):
    return f"{saved_traj_dir_for_method(reg_method)}/{level.name}_{seed}.txt"


def read_offline_output(reg_method: icp.ICPMethod, level: poke.Levels, seed: int, pokes: int):
    filepath = saved_traj_file(reg_method, level, seed)
    if not os.path.isfile(filepath):
        raise RuntimeError(f"Missing path, should run offline method first: {filepath}")

    T = []
    distances = []
    with open(filepath) as f:
        data = f.readlines()
        i = 0
        while i < len(data):
            header = data[i].split()
            this_poke = int(header[0])
            if this_poke < pokes:
                # keep going forward
                i += 5
                continue
            elif this_poke > pokes:
                # assuming the pokes are ordered, if we're past then there won't be anymore of this poke later
                break

            transform = torch.tensor([[float(v) for v in line.strip().split()] for line in data[i + 1:i + 5]])
            T.append(transform)
            batch = int(header[1])
            # lower is better
            rmse = float(header[2])
            distances.append(rmse)
            i += 5

    # current_to_link transform (world to base frame)
    T = torch.stack(T)
    T = T.inverse()
    distances = torch.tensor(distances)
    return T, distances


def test_icp(exp, seed=0, name="", clean_cache=False, viewing_delay=0.1,
             register_num_points=500, eval_num_points=200,
             num_points_list=(2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100),
             num_freespace=0,
             freespace_x_filter_threshold=0.,  # 0 allows only positive, -10 allows any
             surface_delta=0.025,
             freespace_cost_scale=20,
             ground_truth_initialization=False,
             icp_method=icp.ICPMethod.VOLUMETRIC,
             debug=False):
    obj_name = exp.obj_factory.name
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison_{obj_name}.pkl')
    if os.path.exists(fullname) and not clean_cache:
        cache = pd.read_pickle(fullname)
    else:
        cache = pd.DataFrame()

    target_obj_id = exp.objId
    vis = exp.dd
    freespace_ranges = exp.ranges

    vis.draw_point("seed", (0, 0, 0.4), (1, 0, 0), label=f"seed {seed}")

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points, name=obj_name, seed=0,
                                                                  device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                          name=obj_name, seed=0, device=exp.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)

    for num_points in num_points_list:
        # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
        model_points, model_normals, _ = sample_mesh_points(num_points=num_points, name=obj_name, seed=seed,
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
            vis.draw_2d_line(f"mn.{i}", pt, model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
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
        used_model_points = model_points_eval[:, 0] > freespace_x_filter_threshold
        # extrude model points that are on the surface of the object along their normal vector
        free_space_world_frame_points = model_points_world_frame_eval[used_model_points][:num_freespace] + \
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

        errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis if exp.has_gui else None,
                                                     exp.obj_factory, viewing_delay)
        errors.append(errors_per_batch)

    for num, err in zip(num_points_list, errors):
        df = pd.DataFrame(
            {"date": datetime.today().date(), "method": icp_method.name, "name": name, "seed": seed, "points": num,
             "batch": np.arange(B),
             "chamfer_err": err.cpu().numpy()})
        cache = pd.concat([cache, df])

    cache.to_pickle(fullname)
    for i in range(len(num_points_list)):
        print(f"num {num_points_list[i]} err {errors[i]}")


def test_icp_freespace(exp,
                       seed=0, name="", clean_cache=False,
                       viewing_delay=0.3,
                       register_num_points=500, eval_num_points=200, num_points=10,
                       # number of known contact points
                       num_freespace_points_list=(0, 10, 20, 30, 40, 50, 100),
                       freespace_x_filter_threshold=0.,  # 0 allows only positive, -10 allows any
                       surface_delta=0.025,
                       freespace_cost_scale=1,
                       ground_truth_initialization=False,
                       icp_method=icp.ICPMethod.VOLUMETRIC):
    obj_name = exp.obj_factory.name
    fullname = os.path.join(cfg.DATA_DIR, f'icp_freespace_{obj_name}.pkl')
    if os.path.exists(fullname) and not clean_cache:
        cache = pd.read_pickle(fullname)
    else:
        cache = pd.DataFrame()

    target_obj_id = exp.objId
    vis = exp.dd
    freespace_ranges = exp.ranges

    vis.draw_point("seed", (0, 0, 0.4), (1, 0, 0), label=f"seed {seed}")

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points, name=obj_name, seed=0,
                                                                  device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                          name=obj_name, seed=0, device=exp.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    # for mustard bottle there's a hole in the model inside, we restrict it to avoid sampling points nearby
    model_points, model_normals, _ = sample_mesh_points(num_points=num_points, name=obj_name, seed=seed,
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
        vis.draw_2d_line(f"mn.{i}", pt, model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
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
        used_model_points = model_points_eval[:, 0] > freespace_x_filter_threshold
        # extrude model points that are on the surface of the object along their normal vector
        free_space_world_frame_points = model_points_world_frame_eval[used_model_points][:num_freespace] + \
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
            best_tsf_guess = link_to_current_tf_gt.get_matrix().repeat(B, 1, 1)
        T, distances = do_registration(model_points_world_frame, model_points_register, best_tsf_guess, B,
                                       volumetric_cost,
                                       icp_method)

        # draw all ICP's sample meshes
        exp.policy._clear_cached_tf()
        exp.policy.register_transforms(T, distances)
        exp.policy._debug_icp_distribution(None, None)

        errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, exp.obj_factory,
                                                     viewing_delay)
        errors.append(errors_per_batch)

    for num, err in zip(num_freespace_points_list, errors):
        df = pd.DataFrame(
            {"date": datetime.today().date(), "method": icp_method.name, "name": name, "seed": seed, "points": num,
             "batch": np.arange(B),
             "chamfer_err": err.cpu().numpy()})
        cache = pd.concat([cache, df])
    cache.to_pickle(fullname)
    for i in range(len(num_freespace_points_list)):
        print(f"num {num_freespace_points_list[i]} err {errors[i]}")


def test_icp_on_experiment_run(exp, seed=0, viewing_delay=0.1,
                               register_num_points=500, eval_num_points=200,
                               normal_scale=0.05, upto_index=-1, upright_bias=0.1,
                               model_name="mustard", run_name=""):
    name = f"{model_name} {run_name}".strip()
    fullname = os.path.join(cfg.DATA_DIR, f'exploration_res.pkl')
    cache = torch.load(fullname)
    data = cache[name][seed]

    target_obj_id = exp.objId
    vis = exp.dd
    freespace_ranges = exp.ranges

    for _ in range(1000):
        p.stepSimulation()

    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                  device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
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
        vis.draw_2d_line(f"mn.{i}", pt, model_normals_world_frame[i], color=(0, 0, 0), size=2., scale=0.03)
    vis.clear_visualization_after("mpt", i + 1)
    vis.clear_visualization_after("mn", i + 1)

    rand.seed(seed)
    # perform ICP and visualize the transformed points
    # -- try out pytorch3d
    T, distances = icp.icp_pytorch3d(model_points_world_frame, model_points_register, given_init_pose=best_tsf_guess,
                                     batch=B)

    errors_per_batch = evaluate_chamfer_distance(T, model_points_world_frame_eval, vis, exp.obj_factory, viewing_delay)


def marginalize_over_suffix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[:suffix_start_idx - 1] if suffix_start_idx > 0 else "base"


def marginalize_over_prefix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[suffix_start_idx:] if suffix_start_idx > 0 else name


def marginalize_over_registration_num(name):
    registration_num = re.search(r"\d+", name)
    return f"{registration_num[0]} registered points" if registration_num is not None else name


def plot_icp_results(filter=None, logy=True, plot_median=True, x='points', y='chamfer_err',
                     key_columns=("method", "name", "seed", "points", "batch"),
                     keep_lowest_y_quantile=0.5,
                     keep_lowest_y_wrt=None,
                     scatter=False,
                     save_path=None, show=True,
                     leave_out_percentile=50, icp_res_file='icp_comparison.pkl'):
    fullname = os.path.join(cfg.DATA_DIR, icp_res_file)
    df = pd.read_pickle(fullname)

    # clean up the database by removing duplicates (keeping latest results)
    df = df.drop_duplicates(subset=key_columns, keep='last')
    # save this version to keep the size small and not waste the filtering work we just did
    df.to_pickle(fullname)
    df.reset_index(inplace=True)

    if filter is not None:
        df = filter(df)

    group = [x, "method", "name", "seed"]
    if "level" in key_columns:
        group.append("level")
    if keep_lowest_y_wrt is None:
        keep_lowest_y_wrt = y
    df = df[df[keep_lowest_y_wrt] <= df.groupby(group)[keep_lowest_y_wrt].transform('quantile', keep_lowest_y_quantile)]
    df.loc[df["method"].str.contains("ICP"), "name"] = "non-freespace baseline"
    df.loc[df["method"].str.contains("VOLUMETRIC"), "name"] = "ours"
    df.loc[df["method"].str.contains("CVO"), "name"] = "freespace baseline"
    df.loc[df["method"].str.contains("MEDIAL"), "name"] = "freespace baseline"

    method_to_name = df.set_index("method")["name"].to_dict()
    # order the methods should be shown
    full_method_order = ["VOLUMETRIC",
                         # variants of our method
                         "VOLUMETRIC_ICP_INIT", "VOLUMETRIC_NO_FREESPACE",
                         "VOLUMETRIC_LIMITED_REINIT", "VOLUMETRIC_LIMITED_REINIT_FULL",
                         # variants with non-SGD optimization
                         "VOLUMETRIC_CMAES", "VOLUMETRIC_CMAME", "VOLUMETRIC_SVGD",
                         # baselines
                         "ICP", "ICP_REVERSE", "CVO", "MEDIAL"]
    # order the categories should be shown
    full_category_order = ["ours", "non-freespace baseline", "freespace baseline"]
    methods_order = [m for m in full_method_order if m in method_to_name]
    category_order = [m for m in full_category_order if m in method_to_name.values()]
    if scatter:
        res = sns.scatterplot(data=df, x=x, y=y, hue='method', style='name', alpha=0.5)
    else:
        res = sns.lineplot(data=df, x=x, y=y, hue='method', style='name',
                           estimator=np.median if plot_median else np.mean,
                           hue_order=methods_order, style_order=category_order,
                           errorbar=("pi", 100 - leave_out_percentile) if plot_median else ("ci", 95))
    if logy:
        res.set(yscale='log')
    else:
        res.set(ylim=(0, None))

    # combine hue and styles in the legend
    handles, labels = res.get_legend_handles_labels()
    next_title_index = labels.index('name')
    style_dict = {label: (handle.get_linestyle(), handle.get_marker(), handle._dashSeq)
                  for handle, label in zip(handles[next_title_index:], labels[next_title_index:])}

    for handle, label in zip(handles[1:next_title_index], labels[1:next_title_index]):
        handle.set_linestyle(style_dict[method_to_name[label]][0])
        handle.set_marker(style_dict[method_to_name[label]][1])
        dashes = style_dict[method_to_name[label]][2]
        if dashes is not None:
            handle.set_dashes(dashes)

    # create a legend only using the items
    res.legend(handles[1:next_title_index], labels[1:next_title_index], title='method', framealpha=0.4)
    # plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()


class ShapeExplorationExperiment(abc.ABC):
    LINK_FRAME_POS = [0, 0, 0]
    LINK_FRAME_ORIENTATION = [0, 0, 0, 1]

    def __init__(self, obj_factory=obj_factory_map("mustard"),
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

        self.has_gui = gui
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
            model_name="mustard", run_name=""):
        target_obj_id = self.objId
        vis = self.dd
        name = f"{model_name} {run_name}".strip()

        # wait for it to settle
        for _ in range(1000):
            p.stepSimulation()

        # these are in object frame (aligned with [0,0,0], [0,0,0,1]
        model_points, model_normals, _ = sample_mesh_points(self.obj_factory, num_points=500,
                                                            seed=0, clean_cache=build_model,
                                                            name=model_name,
                                                            device=self.device)
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
    def __init__(self, policy_factory=ICPEVExplorationPolicy, policy_args=None, sdf_resolution=0.025, clean_cache=False,
                 **kwargs):
        if policy_args is None:
            policy_args = {}
        super(ICPEVExperiment, self).__init__(**kwargs)

        # test object needs collision shape to test against, so we can't use visual only object
        obj_frame_sdf = stucco.sdf.MeshSDF(self.obj_factory)
        range_per_dim = copy.copy(self.obj_factory.ranges)
        if clean_cache:
            # draw the bounding box of the object frame SDF
            draw_AABB(self.dd, range_per_dim)

        self.sdf = stucco.sdf.CachedSDF(self.obj_factory.name, sdf_resolution, range_per_dim,
                                        obj_frame_sdf, device=self.device, clean_cache=clean_cache)
        self.set_policy(
            policy_factory(self.sdf, vis=self.dd, debug_obj_factory=self.obj_factory, **policy_args))

    def _start_step(self, xs, df):
        pass

    def _end_step(self, xs, df, t):
        pass

    def _eval(self, xs, df, error_at_model_points, error_at_rep_surface, t):
        # get points after transforming with best ICP pose estimate

        link_to_world = tf.Transform3d(matrix=self.policy.best_tsf_guess)
        world_to_link = link_to_world.inverse()

        model_pts_to_compare = self.policy.model_points_world_transformed_ground_truth
        # transform model points to object frame then use the object factory to get distances in object frame
        _, dists, _ = self.obj_factory.object_frame_closest_point(world_to_link.transform_points(model_pts_to_compare))

        dists = dists.abs()
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


def predetermined_controls():
    predetermined_control = {}

    ctrl = []
    # go up along one surface of the object
    for i in range(4):
        ctrl += [[1., 0., 0]] * 3
        ctrl += [[-1., 0., 1]] * 2
    # poke cap of mustard bottle, have to go in more
    # ctrl += [[1., 0., 0]] * 4
    ctrl += [[0., 0., 1]] * 2
    # poke past the top of the mustard bottle
    ctrl += [[1., 0., 0.1]] * 8
    ctrl += [[-1., 0., 0.]] * 2
    # move to the side then poke while going down
    ctrl += [[-1., -0.9, 0]] * 3
    ctrl += [[-1., 0., 0]] * 2
    ctrl += [[0., 0., -1]] * 6
    for i in range(4):
        ctrl += [[1., 0., 0]] * 3
        ctrl += [[-1., 0., 0]] * 2
        ctrl += [[0., 0., -1]] * 2

    # go directly down and sweep out
    ctrl += [[0., -1, 0]] * 5
    ctrl += [[1., 0., 0]] * 6
    ctrl += [[0., 0., 1]] * 10

    # go back to the left side
    ctrl += [[-1., 0., 0]] * 4
    ctrl += [[0., 1., 0]] * 10

    ctrl += [[0., 0.9, 0]] * 6
    ctrl += [[1., 0., 0]] * 5
    ctrl += [[0., 0., -1]] * 13

    rand.seed(0)
    # noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    # ctrl = np.add(ctrl, noise)
    predetermined_control[poke.Levels.MUSTARD] = ctrl

    ctrl = []
    # go up along one surface of the object
    for i in range(3):
        ctrl += [[1., 0., 0]] * 3
        ctrl += [[-1., 0., 1]] * 2
        ctrl.append(None)

    ctrl += [[1., 0., 0]] * 2
    ctrl += [[-1., 0., 0]] * 1
    ctrl.append(None)
    ctrl += [[0., 0., 1]] * 2
    ctrl += [[0., 0., 1]] * 2
    ctrl += [[1., 0., 0]] * 6
    ctrl.append(None)

    ctrl += [[-0.4, 1., 0]] * 3
    ctrl += [[-0.4, 1., -0.5]] * 4

    ctrl += [[1.0, -0.2, 0]] * 2
    ctrl += [[1., 0., -0.4]] * 4
    ctrl += [[-1., 0., -0.4]] * 4
    ctrl.append(None)

    # poke the side inwards once
    ctrl += [[0., -1., 0]] * 2
    ctrl += [[0., 1., 0]] * 1

    # # try poking while going down
    for _ in range(2):
        ctrl += [[1., 0., -0.5]] * 4
        ctrl += [[-1, 0., -0.5]] * 4
        ctrl.append(None)

    ctrl += [[-1., 0., 0]] * 5
    ctrl += [[0., -.99, 0]] * 12
    #
    ctrl += [[1., 0., 0]] * 3
    for _ in range(3):
        ctrl += [[1., 0., 0.5]] * 4
        ctrl += [[-1, 0., 0.5]] * 4
        ctrl.append(None)

    ctrl += [[-1, 0., 0]] * 2
    predetermined_control[poke.Levels.DRILL] = ctrl

    return predetermined_control


def draw_pose_distribution(link_to_world_tf_matrix, obj_id_map, dd, obj_factory: ObjectFactory, sequential_delay=None,
                           show_only_latest=False):
    m = link_to_world_tf_matrix
    for b in range(len(m)):
        mm = m[b]
        pos, rot = util.matrix_to_pos_rot(mm)
        # if we're given a sequential delay, then instead of drawing the distribution simultaneously, we render them
        # sequentially
        if show_only_latest:
            b = 0
            if sequential_delay is not None and sequential_delay > 0:
                time.sleep(sequential_delay)

        object_id = obj_id_map.get(b, None)
        object_id = obj_factory.draw_mesh(dd, "icp_distribution", (pos, rot), (0, 0.8, 0.2, 0.2), object_id=object_id)
        obj_id_map[b] = object_id


class PokingController(Controller):
    class Mode(enum.Enum):
        GO_TO_NEXT_TARGET = 0
        PUSH_FORWARD = 1
        RETURN_BACKWARD = 2
        DONE = 3

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet,
                 y_order=(0, 0.2, 0.3, -0.2, -0.3), z_order=(0.05, 0.15, 0.25, 0.325, 0.4, 0.5), x_rest=-0.05,
                 Kp=30,
                 push_forward_count=10, nu=3, dim=3, goal_tolerance=3e-4):
        super().__init__()

        self.x_rest = x_rest
        self._all_targets = list(itertools.product(y_order, z_order))
        self.target_yz = self._all_targets
        self.push_forward_count = push_forward_count
        self.mode = self.Mode.GO_TO_NEXT_TARGET
        self.kp = Kp

        self.goal_tolerance = goal_tolerance

        # primitive state machine where we push forward for push_forward_count, then go back to x_rest
        self.push_i = 0
        self.i = 0
        self.current_target = None

        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.dim = dim
        self.contact_indices = []

        self.nu = nu

        self.x_history = []
        self.u_history = []

    def reset(self):
        self.target_yz = self._all_targets
        self.mode = self.Mode.GO_TO_NEXT_TARGET
        self.contact_indices = []
        self.x_history = []
        self.u_history = []
        self.push_i = 0
        self.i = 0
        self.current_target = None

    def update(self, obs, info, visualizer=None):
        if self.contact_detector.in_contact():
            self.contact_indices.append(self.i)

        x = self.x_history[-1][:self.dim]
        pt, dx = self.contact_detector.get_last_contact_location(visualizer=visualizer)

        info['u'] = torch.tensor(self.u_history[-1])
        self.contact_set.update(x, dx, pt, info=info)

    def done(self):
        return len(self.target_yz) == 0

    def command(self, obs, info=None, visualizer=None):
        self.x_history.append(obs)

        if len(self.x_history) > 1:
            self.update(obs, info, visualizer=visualizer)

        u = [0 for _ in range(self.nu)]
        if self.done():
            self.mode = self.Mode.DONE
        else:
            if self.mode == self.Mode.GO_TO_NEXT_TARGET:
                # go to next target proportionally
                target = self.target_yz[0]
                diff = np.array(target) - np.array(obs[1:])
                # TODO clamp?
                u[1] = diff[0] * self.kp
                u[2] = diff[1] * self.kp

                if np.linalg.norm(diff) < self.goal_tolerance:
                    self.mode = self.Mode.PUSH_FORWARD
                    self.push_i = 0

            # if we don't have a current target, find the next
            elif self.mode == self.Mode.PUSH_FORWARD:
                u[0] = 1.
                self.push_i += 1
                if self.push_i >= self.push_forward_count or np.linalg.norm(info['reaction']) > 5:
                    self.mode = self.Mode.RETURN_BACKWARD
            elif self.mode == self.Mode.RETURN_BACKWARD:
                diff = self.x_rest - obs[0]
                u[0] = diff * self.kp

                if abs(diff) < self.goal_tolerance:
                    self.mode = self.Mode.GO_TO_NEXT_TARGET
                    self.target_yz = self.target_yz[1:]
                    u = None

            self.i += 1

        if u is not None:
            self.u_history.append(u)
        return u


def _export_pcs(f, pc_free, pc_occ):
    pc_free_serialized = [f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} 0" for pt in pc_free]
    pc_occ_serialized = [f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} 1" for pt in pc_occ]
    f.write("\n".join(pc_free_serialized))
    f.write("\n")
    f.write("\n".join(pc_occ_serialized))
    f.write("\n")


def _export_transform(f, T):
    T_serialized = [f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {t[3]:.4f}" for t in T]
    f.write("\n".join(T_serialized))
    f.write("\n")


def export_pc_register_against(point_cloud_file: str, env: poke.PokeEnv):
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'w') as f:
        surface_thresh = 0.01
        pc_surface = env.target_sdf.get_filtered_points(
            lambda voxel_sdf: (voxel_sdf < surface_thresh) & (voxel_sdf > -surface_thresh))
        if len(pc_surface) == 0:
            raise RuntimeError(f"Surface threshold of {surface_thresh} leads to no surface points")
        pc_free = env.target_sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf >= surface_thresh)

        total_pts = len(pc_free) + len(pc_surface)
        f.write(f"{total_pts}\n")
        _export_pcs(f, pc_free, pc_surface)


def export_pc_to_register(point_cloud_file: str, pokes: int, env: poke.PokeEnv, method: TrackingMethod):
    # append to the end and create it if it doesn't exist
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'a') as f:
        # write out the poke index and the size of the point cloud
        pc_free, _ = env.free_voxels.get_known_pos_and_values()
        _, pc_occ = method.get_labelled_moved_points()
        total_pts = len(pc_free) + len(pc_occ)
        f.write(f"{pokes} {total_pts}\n")
        _export_pcs(f, pc_free, pc_occ)


def export_registration(stored_file: str, to_export):
    """Exports current_to_link (world frame to base frame) transforms to file"""
    os.makedirs(os.path.dirname(stored_file), exist_ok=True)
    with open(stored_file, 'w') as f:
        # sort to order by pokes
        for pokes, data in sorted(to_export.items()):
            T = data['T'].inverse()
            d = data['rmse']
            B = T.shape[0]
            assert B == d.shape[0]
            for b in range(B):
                f.write(f"{pokes} {b} {d[b]}\n")
                _export_transform(f, T[b])


def export_init_transform(transform_file: str, T: torch.tensor):
    os.makedirs(os.path.dirname(transform_file), exist_ok=True)
    B = len(T)
    with open(transform_file, 'w') as f:
        f.write(f"{B}\n")
        for b in range(len(T)):
            f.write(f"{b}\n")
            serialized = [f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {t[3]:.4f}" for t in T[b]]
            f.write("\n".join(serialized))
            f.write("\n")


def debug_volumetric_loss(env: poke.PokeEnv, seed=0, show_free_voxels=False, pokes=4):
    # load from file
    dtype = env.dtype
    device = env.device
    pc_register_against_file = os.path.join(cfg.DATA_DIR, f"poke/{env.level.name}_{seed}.txt")
    with open(pc_register_against_file) as f:
        lines = [[float(v) for v in line.strip().split()] for line in f.readlines()]
        i = 0
        while i < len(lines):
            this_pokes, num_points = lines[i]
            this_pokes = int(this_pokes)
            num_points = int(num_points)
            if this_pokes == pokes:
                all_pts = torch.tensor(lines[i + 1: i + 1 + num_points], device=device, dtype=dtype)
                freespace = all_pts[:, -1] == 0
                freespace_pts = all_pts[freespace, :-1]
                pts = all_pts[~freespace, :-1]
                env.free_voxels[freespace_pts] = 1

                if show_free_voxels:
                    env._debug_visualizations[poke.DebugVisualization.FREE_VOXELS] = True
                    env._occupy_current_config_as_freespace()
                break
            i += num_points + 1

    for i, pt in enumerate(pts):
        env.vis.draw_point(f"c.{i}", pt.cpu().numpy(), (1, 0, 0))

    empty_sdf = util.VoxelSet(torch.empty(0), torch.empty(0))
    volumetric_cost = icp_costs.VolumetricCost(env.free_voxels, empty_sdf, env.target_sdf, scale=1,
                                               scale_known_freespace=1,
                                               vis=env.vis, debug=True)

    this_pts = tensor_utils.ensure_tensor(device, dtype, pts)
    volumetric_cost.sdf_voxels = util.VoxelSet(this_pts,
                                               torch.zeros(this_pts.shape[0], dtype=dtype, device=device))

    pose = env.target_pose
    gt_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(tensor_utils.ensure_tensor(device, dtype, pose[1])),
                           dtype=dtype, device=device)
    Hgt = gt_tf.get_matrix()
    Hgt.requires_grad = True
    Hgtinv = Hgt.inverse()
    gt_cost = volumetric_cost(Hgtinv[:, :3, :3], Hgtinv[:, :3, -1], None)
    gt_cost.mean().backward()
    volumetric_cost.visualize(Hgtinv[:, :3, :3], Hgtinv[:, :3, -1], None)

    env.draw_user_text("{}".format(gt_cost.item()), xy=(0.5, 0.7, -0.3))
    pose_obj_map = {}
    while True:
        from torch import tensor  # prints will just have tensor rather than torch.tensor so this allows copy pasting
        H = Hgt.clone()
        # breakpoint here and change the transform H with respect to the ground truth one to evaluate the cost
        if H.dim() == 2:
            H = H.unsqueeze(0)
            H.requires_grad = True
        cost = volumetric_cost(H.inverse()[:, :3, :3], H.inverse()[:, :3, -1], None)
        cost.mean().backward()
        volumetric_cost.visualize(H.inverse()[:, :3, :3], H.inverse()[:, :3, -1], None)
        env.draw_user_text("{}".format(cost.item()), xy=(0.5, 0.7, -.5))
        draw_pose_distribution(H, pose_obj_map, env.vis, obj_factory)


class PokeRunner:
    KEY_COLUMNS = ("method", "name", "seed", "poke", "level", "batch")

    def __init__(self, env: poke.PokeEnv, tracking_method_name: str, reg_method, B=30,
                 read_stored=False, ground_truth_initialization=False, init_method=icp.InitMethod.RANDOM,
                 register_num_points=500, start_at_num_pts=4, eval_num_points=200):
        self.env = env
        self.B = B
        self.dbname = os.path.join(cfg.DATA_DIR, f'poking_{env.obj_factory.name}.pkl')
        self.tracking_method_name = tracking_method_name
        self.reg_method = reg_method
        self.start_at_num_pts = start_at_num_pts
        self.read_stored = read_stored
        self.ground_truth_initialization = ground_truth_initialization
        self.init_method = init_method

        model_name = self.env.target_model_name
        # get a fixed number of model points to evaluate against (this will be independent on points used to register)
        self.model_points_eval, self.model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points,
                                                                                name=model_name,
                                                                                seed=0,
                                                                                device=env.device)
        self.device, self.dtype = self.model_points_eval.device, self.model_points_eval.dtype

        # get a large number of model points to register to
        self.model_points_register, self.model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                                        name=model_name, seed=0,
                                                                                        device=env.device)

        pose = p.getBasePositionAndOrientation(self.env.target_object_id())
        self.link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        self.model_points_world_frame_eval = self.link_to_current_tf_gt.transform_points(self.model_points_eval)
        self.model_normals_world_frame_eval = self.link_to_current_tf_gt.transform_points(self.model_normals_eval)

        self.draw_pose_distribution_separately = True
        self.method: typing.Optional[TrackingMethod] = None
        self.ctrl: typing.Optional[Controller] = None
        self.volumetric_cost: typing.Optional[icp_costs.VolumetricCost] = None
        # intermediary data for bookkeeping during a run
        self.pokes = 0
        self.best_tsf_guess = None
        self.num_points_to_T_cache = {}
        self.best_distance = None
        self.dist_per_est_obj = []
        self.transforms_per_object = []
        self.rmse_per_object = []
        self.chamfer_err = []
        self.freespace_violations = []
        self.num_freespace_voxels = []
        # for debug rendering of object meshes and keeping track of their object IDs
        self.pose_obj_map = {}
        # for exporting out to file, maps poke # -> data
        self.to_export = {}
        self.cache = None

    def create_volumetric_cost(self):
        # placeholder for now; have to be filled manually
        empty_sdf = util.VoxelSet(torch.empty(0), torch.empty(0))
        self.volumetric_cost = icp_costs.VolumetricCost(self.env.free_voxels, empty_sdf, self.env.target_sdf, scale=1,
                                                        scale_known_freespace=20,
                                                        vis=self.env.vis, debug=False)

    def register_transforms_with_points(self):
        """Exports best_segment_idx, transforms_per_object, and rmse_per_object"""
        # note that we update our registration regardless if we're in contact or not
        self.best_distance = None
        self.dist_per_est_obj = []
        self.transforms_per_object = []
        self.rmse_per_object = []
        self.best_segment_idx = None
        for k, this_pts in enumerate(self.method):
            N = len(this_pts)
            if N < self.start_at_num_pts or self.pokes < self.start_at_num_pts:
                continue
            # this_pts corresponds to tracked contact points that are segmented together
            this_pts = tensor_utils.ensure_tensor(self.device, self.dtype, this_pts)
            self.volumetric_cost.sdf_voxels = util.VoxelSet(this_pts,
                                                            torch.zeros(this_pts.shape[0], dtype=self.dtype,
                                                                        device=self.device))

            if self.best_tsf_guess is None:
                self.best_tsf_guess = initialize_transform_estimates(self.B, env, self.init_method, self.method)
            if self.ground_truth_initialization:
                self.best_tsf_guess = self.link_to_current_tf_gt.get_matrix().repeat(self.B, 1, 1)

            # avoid giving methods that don't use freespace more training iterations
            if registration_method_uses_only_contact_points(self.reg_method) and N in self.num_points_to_T_cache:
                T, distances = self.num_points_to_T_cache[N]
            else:
                if self.read_stored or self.reg_method == icp.ICPMethod.CVO:
                    T, distances = read_offline_output(self.reg_method, level, seed, self.pokes)
                    T = T.to(device=self.device, dtype=self.dtype)
                    distances = distances.to(device=self.device, dtype=self.dtype)
                else:
                    T, distances = do_registration(this_pts, self.model_points_register, self.best_tsf_guess, self.B,
                                                   self.volumetric_cost,
                                                   self.reg_method)
                self.num_points_to_T_cache[N] = T, distances

            self.transforms_per_object.append(T)
            T = T.inverse()
            score = distances
            best_tsf_index = np.argmin(score.detach().cpu())

            # pick object with lowest variance in its translation estimate
            translations = T[:, :3, 3]
            best_tsf_distances = (translations.var(dim=0).sum()).item()

            self.dist_per_est_obj.append(best_tsf_distances)
            self.rmse_per_object.append(distances)
            if self.best_distance is None or best_tsf_distances < self.best_distance:
                self.best_distance = best_tsf_distances
                self.best_tsf_guess = T[best_tsf_index]
                self.best_segment_idx = k

    def reinitialize_best_tsf_guess(self):
        # sample rotation and translation around the previous best solution to reinitialize
        radian_sigma = 0.3
        translation_sigma = 0.05

        # sample delta rotations in axis angle form
        temp = torch.eye(4, dtype=self.dtype, device=self.device).repeat(self.B, 1, 1)

        delta_R = torch.randn((self.B, 3), dtype=self.dtype, device=self.device) * radian_sigma
        delta_R = tf.axis_angle_to_matrix(delta_R)
        temp[:, :3, :3] = delta_R @ self.best_tsf_guess[:3, :3]
        temp[:, :3, 3] = self.best_tsf_guess[:3, 3]

        delta_t = torch.randn((self.B, 3), dtype=self.dtype, device=self.device) * translation_sigma
        temp[:, :3, 3] += delta_t
        # ensure one of them (first of the batch) has the exact transform
        temp[0] = self.best_tsf_guess
        self.best_tsf_guess = temp
        return self.best_tsf_guess

    def evaluate_registrations(self):
        """Responsible for populating to_export"""
        self.method.register_transforms(self.transforms_per_object[self.best_segment_idx], self.best_tsf_guess)
        logger.debug(f"err each obj {np.round(self.dist_per_est_obj, 4)}")
        best_T = self.best_tsf_guess

        self.reinitialize_best_tsf_guess()

        T = self.transforms_per_object[self.best_segment_idx]

        # when evaluating, move the best guess pose far away to improve clarity
        self.env.draw_mesh("base_object", ([0, 0, 100], [0, 0, 0, 1]), (0.0, 0.0, 1., 0.5),
                           object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
        if self.draw_pose_distribution_separately:
            evaluate_chamfer_dist_extra_args = [self.env.vis if self.env.mode == p.GUI else None, self.env.obj_factory,
                                                0.05,
                                                False]
        else:
            draw_pose_distribution(T.inverse(), self.pose_obj_map, self.env.vis, obj_factory)
            evaluate_chamfer_dist_extra_args = [None, self.env.obj_factory, 0., False]

        # evaluate with chamfer distance
        errors_per_batch = evaluate_chamfer_distance(T, self.model_points_world_frame_eval,
                                                     *evaluate_chamfer_dist_extra_args)

        link_to_current_tf = tf.Transform3d(matrix=T)
        interior_pts = link_to_current_tf.transform_points(self.volumetric_cost.model_interior_points_orig)
        occupied = self.env.free_voxels[interior_pts]

        self.chamfer_err.append(errors_per_batch)
        self.num_freespace_voxels.append(self.env.free_voxels.get_known_pos_and_values()[0].shape[0])
        self.freespace_violations.append(occupied.sum(dim=-1).detach().cpu())
        logger.info(f"chamfer distance {self.pokes}: {torch.mean(errors_per_batch)}")

        # draw mesh at where our best guess is
        guess_pose = util.matrix_to_pos_rot(best_T)
        self.env.draw_mesh("base_object", guess_pose, (0.0, 0.0, 1., 0.5),
                           object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)

    def export_metrics(self, cache, name, seed):
        """Responsible for populating to_export and saving to database"""
        _c = np.array(self.chamfer_err[-1].cpu().numpy())
        _f = np.array(self.freespace_violations[-1])
        _n = self.num_freespace_voxels[-1]
        _r = _f / _n
        batch = np.arange(self.B)
        rmse = self.rmse_per_object[self.best_segment_idx]

        df = pd.DataFrame(
            {"date": datetime.today().date(), "method": self.reg_method.name, "level": self.env.level.name,
             "name": name,
             "seed": seed, "poke": self.pokes,
             "batch": batch,
             "chamfer_err": _c, 'freespace_violations': _f,
             'num_freespace_voxels': _n,
             "freespace_violation_percent": _r,
             "rmse": rmse.cpu().numpy(),
             })
        cache = pd.concat([cache, df])
        cache.to_pickle(self.dbname)
        # additional data to export fo file
        self.to_export[self.pokes] = {
            'T': self.transforms_per_object[self.best_segment_idx],
            'rmse': rmse,
        }
        return cache

    def run(self, name="", seed=0, ctrl_noise_max=0.005, draw_text=None):
        if os.path.exists(self.dbname):
            self.cache = pd.read_pickle(self.dbname)
        else:
            self.cache = pd.DataFrame()

        env = self.env
        if draw_text is None:
            self.env.draw_user_text(f"{self.reg_method.name}{name} seed {seed}", xy=[-0.3, 1., -0.5])
        else:
            self.env.draw_user_text(draw_text, xy=[-0.3, 1., -0.5])

        obs = self.env.reset()
        self.create_volumetric_cost()
        self.method = create_tracking_method(self.env, tracking_method_name)
        y_order, z_order = predetermined_poke_range().get(env.level,
                                                          ((0, 0.2, 0.3, -0.2, -0.3),
                                                           (0.05, 0.15, 0.25, 0.325, 0.4, 0.5)))
        assert isinstance(self.method, OurSoftTrackingMethod)
        self.ctrl = PokingController(env.contact_detector, self.method.contact_set, y_order=y_order, z_order=z_order)

        info = None
        simTime = 0
        self.pokes = 0
        self.best_tsf_guess = None
        self.chamfer_err = []
        self.freespace_violations = []
        self.num_freespace_voxels = []
        # for debug rendering of object meshes and keeping track of their object IDs
        self.pose_obj_map = {}
        # for exporting out to file, maps poke # -> data
        self.to_export = {}

        rand.seed(seed)
        # create the action noise before sampling to ensure they are the same across methods
        action_noise = np.random.randn(5000, 3) * ctrl_noise_max

        self.hook_before_first_poke(seed)
        while not self.ctrl.done():
            simTime += 1
            env.draw_user_text("{}".format(self.pokes), xy=(0.5, 0.4, -0.5))

            action = self.ctrl.command(obs, info)
            self.method.visualize_contact_points(env)

            if action is None:
                self.pokes += 1
                self.hook_after_poke(name, seed)

            if action is not None:
                if torch.is_tensor(action):
                    action = action.cpu()

                action = np.array(action).flatten()
                action += action_noise[simTime]
                obs, rew, done, info = env.step(action)

        self.env.vis.clear_visualizations()
        self.hook_after_last_poke(seed)

    # hooks for derived classes to add behavior at specific locations
    def hook_before_first_poke(self, seed):
        pass

    def hook_after_poke(self, name, seed):
        self.register_transforms_with_points()
        # has at least one contact segment
        if self.best_segment_idx is not None:
            self.evaluate_registrations()
            self.cache = self.export_metrics(self.cache, name, seed)

    def hook_after_last_poke(self, seed):
        if not self.read_stored:
            export_registration(saved_traj_file(self.reg_method, self.env.level, seed), self.to_export)


class ExportProblemRunner(PokeRunner):
    """If registration method is None, export the point clouds, initial transforms, and ground truth transforms"""

    def __init__(self, *args, **kwargs):
        super(ExportProblemRunner, self).__init__(*args, **kwargs)
        self.reg_method = icp.ICPMethod.NONE
        self.pc_to_register_file = None
        self.transform_file = None

    def hook_before_first_poke(self, seed):
        data_dir = cfg.DATA_DIR
        self.pc_to_register_file = os.path.join(data_dir, f"poke/{self.env.level.name}_{seed}.txt")
        pc_register_against_file = os.path.join(data_dir, f"poke/{self.env.level.name}.txt")
        self.transform_file = os.path.join(data_dir, f"poke/{self.env.level.name}_{seed}_trans.txt")
        transform_gt_file = os.path.join(data_dir, f"poke/{self.env.level.name}_{seed}_gt_trans.txt")
        # exporting data for offline baselines, remove the stale file
        export_pc_register_against(pc_register_against_file, self.env)
        export_init_transform(transform_gt_file, self.link_to_current_tf_gt.get_matrix().repeat(self.B, 1, 1))
        try:
            os.remove(self.pc_to_register_file)
        except OSError:
            pass

    def hook_after_poke(self, name, seed):
        if self.pokes >= self.start_at_num_pts:
            if self.best_tsf_guess is None:
                self.best_tsf_guess = initialize_transform_estimates(self.B, env, self.init_method, self.method)
                export_init_transform(self.transform_file, self.best_tsf_guess)
            export_pc_to_register(self.pc_to_register_file, self.pokes, env, self.method)


class PlausibleSetRunner(PokeRunner):
    def plausible_set_filename(self, seed):
        return os.path.join(cfg.DATA_DIR, f"poke/{self.env.level.name}_plausible_set_{seed}.pkl")


class GeneratePlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, plausible_suboptimality=0.001, gt_position_max_offset=0.2, position_steps=10, N_rot=10000,
                 **kwargs):
        super(GeneratePlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plausible_suboptimality = plausible_suboptimality
        # self.gt_position_max_offset = gt_position_max_offset
        # self.position_steps = position_steps
        self.N_rot = N_rot

        rand.seed(0)
        # maps poke number to a set of plausible transforms
        self.plausible_set = {}
        # assume that there can only be plausible completions close enough to the ground truth
        target_pos = self.env.target_pose[0]
        x = torch.linspace(target_pos[0] - gt_position_max_offset * 0.5, target_pos[0] + gt_position_max_offset * 1.5,
                           steps=position_steps, device=self.device)
        y = torch.linspace(target_pos[1] - gt_position_max_offset, target_pos[1] + gt_position_max_offset,
                           steps=position_steps, device=self.device)
        z = torch.linspace(target_pos[2], target_pos[2] + gt_position_max_offset,
                           steps=position_steps, device=self.device)
        self.pos = torch.cartesian_prod(x, y, z)
        self.N_pos = len(self.pos)
        # uniformly sample rotations
        self.rot = tf.random_rotations(N_rot, device=self.device)
        # we know most of the ground truth poses are actually upright, so let's add those in as hard coded
        N_upright = min(100, N_rot)
        axis_angle = torch.zeros((N_upright, 3), dtype=self.dtype, device=self.device)
        axis_angle[:, -1] = torch.linspace(0, 2 * np.pi, N_upright)
        self.rot[:N_upright] = tf.axis_angle_to_matrix(axis_angle)

        pose = self.env.target_pose
        gt_tf = tf.Transform3d(pos=pose[0],
                               rot=tf.xyzw_to_wxyz(tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])),
                               dtype=self.dtype, device=self.device)
        self.Hgt = gt_tf.get_matrix()
        self.Hgtinv = self.Hgt.inverse()

    def create_volumetric_cost(self):
        # TODO use exact SDF when evaluating costs rather than voxelized
        return super(GeneratePlausibleSetRunner, self).create_volumetric_cost()

    def hook_after_poke(self, name, seed):
        # assume all contact points belong to the object
        contact_pts = []
        for k, this_pts in enumerate(self.method):
            contact_pts.append(this_pts)
        contact_pts = tensor_utils.ensure_tensor(self.device, self.dtype, torch.cat(contact_pts))
        if len(contact_pts) < self.start_at_num_pts or self.pokes < self.start_at_num_pts:
            return

        self.volumetric_cost.sdf_voxels = util.VoxelSet(contact_pts, torch.zeros(contact_pts.shape[0], dtype=self.dtype,
                                                                                 device=self.device))
        self.evaluate_registrations()

    def evaluate_registrations(self):
        # evaluate all the transforms
        plausible_transforms = []
        gt_cost = self.volumetric_cost(self.Hgtinv[:, :3, :3], self.Hgtinv[:, :3, -1], None)
        # evaluate the pts in chunks since we can't load all points in memory at the same time
        rot_chunk = 1
        pos_chunk = 1000
        for i in range(0, self.N_rot, rot_chunk):
            logger.debug(f"chunked {i}/{self.N_rot} plausible: {sum(h.shape[0] for h in plausible_transforms)}")
            for j in range(0, self.N_pos, pos_chunk):
                R = self.rot[i:i + rot_chunk]
                T = self.pos[j:j + pos_chunk]
                r_chunk_actual = len(R)
                t_chunk_actual = len(T)
                T = T.repeat(r_chunk_actual, 1)
                R = R.repeat_interleave(t_chunk_actual, 0)
                H = torch.eye(4, device=self.device).repeat(len(R), 1, 1)
                H[:, :3, :3] = R
                H[:, :3, -1] = T
                Hinv = H.inverse()

                costs = self.volumetric_cost(Hinv[:, :3, :3], Hinv[:, :3, -1], None)
                plausible = costs < self.plausible_suboptimality + gt_cost

                if torch.any(plausible):
                    Hp = H[plausible]
                    plausible_transforms.append(Hp)

        all_plausible_transforms = torch.cat(plausible_transforms)
        self.plausible_set[self.pokes] = all_plausible_transforms
        logger.info("poke %d with %d plausible completions gt cost: %f allowable cost: %f", self.pokes,
                    all_plausible_transforms.shape[0], gt_cost, gt_cost + self.plausible_suboptimality)
        # plot plausible transforms
        to_plot = torch.randperm(len(all_plausible_transforms))[:200]
        draw_pose_distribution(all_plausible_transforms[to_plot], self.pose_obj_map, self.env.vis, self.env.obj_factory,
                               show_only_latest=True, sequential_delay=0.1)

    def hook_after_last_poke(self, seed):
        # export plausible set to file
        filename = self.plausible_set_filename(seed)
        logger.info("saving plausible set to %s", filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.plausible_set, filename)


class PlotPlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, show_only_latest=True, sequential_delay=0.1, **kwargs):
        super(PlotPlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plausible_set = {}
        self.show_only_latest = show_only_latest
        self.sequential_delay = sequential_delay

    def hook_before_first_poke(self, seed):
        filename = self.plausible_set_filename(seed)
        self.plausible_set = torch.load(filename)

    def hook_after_poke(self, name, seed):
        if self.pokes in self.plausible_set:
            ps = self.plausible_set[self.pokes]
            if not self.show_only_latest:
                self.env.vis.clear_visualizations()
                self.pose_obj_map = {}
            draw_pose_distribution(ps, self.pose_obj_map, self.env.vis, self.env.obj_factory,
                                   show_only_latest=self.show_only_latest, sequential_delay=self.sequential_delay)

    def hook_after_last_poke(self, seed):
        pass


class EvaluatePlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, plot_meshes=True, sleep_between_plots=0.1, **kwargs):
        super(EvaluatePlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plot_meshes = plot_meshes
        self.sleep_between_plots = sleep_between_plots
        # always read stored with the plausible set evaluation
        self.read_stored = True
        self.plausible_set = {}
        self.plausibility = None
        self.coverage = None

    def hook_before_first_poke(self, seed):
        # TODO actually pass seed in if we're generating different plausible sets per seed
        # they are different trajectories due to control noise, but they are almost the exact same
        filename = self.plausible_set_filename(0)
        self.plausible_set = torch.load(filename)
        self.cache = self.cache.drop_duplicates(subset=self.KEY_COLUMNS, keep='last')

    def hook_after_poke(self, name, seed):
        if self.pokes in self.plausible_set:
            self.register_transforms_with_points()
            # has at least one contact segment
            if self.best_segment_idx is not None:
                logger.warning("No sufficient contact segment on poke %d despite having data for the plausible set",
                               self.pokes)
                return
            self.evaluate_registrations()
            self.cache = self.export_metrics(self.cache, name, seed)

    def hook_after_last_poke(self, seed):
        pass

    def evaluate_registrations(self):
        """Responsible for populating to_export"""
        self.method.register_transforms(self.transforms_per_object[self.best_segment_idx], self.best_tsf_guess)
        logger.debug(f"err each obj {np.round(self.dist_per_est_obj, 4)}")

        # sampled transforms and all plausible transforms
        T = self.transforms_per_object[self.best_segment_idx]
        Tp = self.plausible_set[self.pokes]

        # NOTE that T is already inverted, so no need for T.inverse() * T
        # however, Tinv is useful for plotting purposes
        Tinv = T.inverse()

        # effectively can apply one transform then take the inverse using the other one; if they are the same, then
        # we should end up in the base frame if that T == Tp
        # want pairwise matrix multiplication |T| x |Tp| x 4 x 4 T[0]@Tp[0], T[0]@Tp[1]
        # Iapprox = torch.einsum("bij,pjk->bpik", T, Tp)
        # the einsum does the multiplication below and is about twice as fast
        Iapprox = T.view(-1, 1, 4, 4) @ Tp.view(1, -1, 4, 4)

        # evaluate_chamfer_distance()

        # find pairwise chamfer distance between the sampled transforms
        # when evaluating, move the best guess pose far away to improve clarity
        self.env.draw_mesh("base_object", ([0, 0, 100], [0, 0, 0, 1]), (0.0, 0.0, 1., 0.5),
                           object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
        B, P = Iapprox.shape[:2]
        errors_per_batch = evaluate_chamfer_distance(Iapprox.view(B * P, 4, 4), self.model_points_eval, None,
                                                     self.env.obj_factory, 0)
        errors_per_batch = errors_per_batch.view(B, P)

        best_per_sampled = errors_per_batch.min(dim=1)
        best_per_plausible = errors_per_batch.min(dim=0)

        bp_plausibility = best_per_sampled.values.sum() / B
        bp_coverage = best_per_plausible.values.sum() / P

        # sampled vs closest plausible transform
        if self.plot_meshes:
            self.env.draw_user_text("sampled (green) vs closest plausible (blue)", xy=[-0.1, -0.1, -0.5])
            for b in range(B):
                p = best_per_sampled.indices[b]
                self.env.draw_user_text(f"{best_per_sampled.values[b].item():.0f}", xy=[-0.1, -0.2, -0.5])
                self.env.draw_mesh("sampled", util.matrix_to_pos_rot(Tinv[b]), (0.0, 1.0, 0., 0.5),
                                   object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                self.env.draw_mesh("plausible", util.matrix_to_pos_rot(Tp[p]), (0.0, 0.0, 1., 0.5),
                                   object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                time.sleep(self.sleep_between_plots)

            self.env.draw_user_text("plausible (blue) vs closest sampled (blue)", xy=[-0.1, -0.1, -0.5])
            for p in range(P):
                b = best_per_plausible.indices[p]
                self.env.draw_user_text(f"{best_per_plausible.values[p].item():.0f}", xy=[-0.1, -0.2, -0.5])
                self.env.draw_mesh("sampled", util.matrix_to_pos_rot(Tinv[b]), (0.0, 1.0, 0., 0.5),
                                   object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                self.env.draw_mesh("plausible", util.matrix_to_pos_rot(Tp[p]), (0.0, 0.0, 1., 0.5),
                                   object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                time.sleep(self.sleep_between_plots)

        # NOTE different multiplication order Tp @ T is also valid
        Iapprox = Tp.view(-1, 1, 4, 4) @ T.view(1, -1, 4, 4)
        errors_per_batch = evaluate_chamfer_distance(Iapprox.view(B * P, 4, 4), self.model_points_eval, None,
                                                     self.env.obj_factory, 0)
        errors_per_batch = errors_per_batch.view(P, B)
        best_per_sampled = errors_per_batch.min(dim=0)
        best_per_plausible = errors_per_batch.min(dim=1)
        pb_plausibility = best_per_sampled.values.sum() / B
        pb_coverage = best_per_plausible.values.sum() / P

        logger.info(
            f"pokes {self.pokes} BP plausibility {bp_plausibility.item():.0f} coverage {bp_coverage.item():.0f} "
            f"PB plausibility {pb_plausibility.item():.0f} coverage {pb_coverage.item():.0f}")
        self.plausibility = bp_plausibility
        self.coverage = bp_coverage

    def export_metrics(self, cache, name, seed):
        """Responsible for populating to_export and saving to database"""
        batch = np.arange(self.B)
        df = pd.DataFrame(
            {"method": self.reg_method.name, "level": self.env.level.name,
             "name": name,
             "seed": seed, "poke": self.pokes,
             "batch": batch,
             "plausibility": self.plausibility.item(),
             "coverage": self.coverage.item(),
             "plausible_diversity": (self.plausibility + self.coverage).item(),
             })

        cols = list(cache.columns)
        cols_next = [c for c in df.columns if c not in cols]
        if "plausibility" not in cols:
            dd = pd.merge(cache, df, how="outer", suffixes=('', '_y'))
        else:
            # this will replace values that are shared between cache and df with those of df
            dd = df.combine_first(cache)

        # rearrange back in proper order
        dd = dd[cols + cols_next]
        dd.to_pickle(self.dbname)

        return dd


def initialize_transform_estimates(B, env: poke.PokeEnv, init_method: icp.InitMethod, tracker: TrackingMethod):
    dtype = env.dtype
    device = env.device
    # translation is 0,0,0
    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)
    if init_method == icp.InitMethod.ORIGIN:
        pass
    elif init_method == icp.InitMethod.CONTACT_CENTROID:
        _, points = tracker.get_labelled_moved_points()
        centroid = points.mean(dim=0).to(device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = centroid
    elif init_method == icp.InitMethod.RANDOM:
        trans = np.random.uniform(env.freespace_ranges[:, 0], env.freespace_ranges[:, 1], (B, 3))
        trans = torch.tensor(trans, device=device, dtype=dtype)
        best_tsf_guess[:, :3, 3] = trans
    else:
        raise RuntimeError(f"Unsupported initialization method: {init_method}")
    return best_tsf_guess


def create_tracking_method(env, method_name) -> TrackingMethod:
    if method_name == "ours":
        return OurSoftTrackingMethod(env, PokeGetter.contact_parameters(env), poke.ArmPointToConfig(env), dim=3)
    elif method_name == 'online-birch':
        return SklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                     inertia_ratio=0.2,
                                     threshold=0.08)
    elif method_name == 'online-dbscan':
        return SklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1)
    elif method_name == 'online-kmeans':
        return SklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2, n_clusters=1,
                                     random_state=0)
    elif method_name == 'gmphd':
        return PHDFilterTrackingMethod(env, fp_fn_bias=4, q_mag=0.00005, r_mag=0.00005, birth=0.001, detection=0.3)
    else:
        raise RuntimeError(f"Unsupported tracking method {method_name}")


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

        experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
        for seed in range(10):
            test_icp(experiment, seed=seed, register_num_points=500,
                     num_freespace=0,
                     name=f"gt init pytorch3d reverse",
                     icp_method=icp.ICPMethod.ICP_REVERSE,
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
                         freespace_x_filter_threshold=-10,
                         freespace_cost_scale=20,
                         name=f"gt init volumetric freespace sdf res {sdf_resolution}",
                         icp_method=icp.ICPMethod.VOLUMETRIC,
                         ground_truth_initialization=True,
                         viewing_delay=0)
            experiment.close()
    file = f"icp_comparison_{obj_factory.name}.pkl"
    plot_icp_results(icp_res_file=file, reduce_batch=np.mean, names_to_include=lambda name: name.startswith("gt init"))
    plot_icp_results(icp_res_file=file, reduce_batch=np.mean, names_to_include=lambda name: name.startswith("gt init"),
                     x_filter=lambda x: x < 40)


def experiment_vary_num_points_and_num_freespace(obj_factory, plot_only=False, gui=True):
    # -- Differing number of freespace experiment while varying number of known points
    if not plot_only:
        for surface_delta in [0.01, 0.025, 0.05]:
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
        name: "volumetric fixed sdf free pts 100" in name or name == "volumetric fixed sdf free pts 0 delta 0.025")
    plot_icp_results(icp_res_file=file, reduce_batch=np.median, names_to_include=lambda
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
        some_side_x_threshold_map = {"mustard": -0.04}
        # for surface_delta in [0.01, 0.025, 0.05]:
        for thres in [0.0, ]:
            for surface_delta in [0.01, ]:
                for freespace_cost_scale in [20]:
                    experiment = ICPEVExperiment(device="cuda", obj_factory=obj_factory, gui=gui)
                    # thres = some_side_x_threshold_map[obj_factory.name],
                    for num_points in [5]:
                        for seed in range(10):
                            test_icp_freespace(experiment, seed=seed, num_points=num_points,
                                               register_num_points=500,
                                               surface_delta=surface_delta,
                                               freespace_x_filter_threshold=thres,
                                               freespace_cost_scale=freespace_cost_scale,
                                               name=f"volumetric {num_points}np {thres} threshold delta {surface_delta} scale {freespace_cost_scale}",
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
                                           freespace_x_filter_threshold=-10,
                                           name=f"volumetric {num_points}np all sides delta {surface_delta} scale {freespace_cost_scale}",
                                           viewing_delay=0)
                experiment.close()
    file = f"icp_freespace_{obj_factory.name}.pkl"
    plot_icp_results(icp_res_file=file, reduce_batch=np.mean,
                     names_to_include=lambda
                         name: "volumetric 5np" in name and "scale 20" in name and "delta 0.01" in name and (
                             " 0.0 " in name or "-0.1 " in name or "-0.05 " in name or "-0.03 " in name))


def experiment_compare_basic_baseline(obj_factory, plot_only=False, gui=True):
    file = f"icp_comparison_{obj_factory.name}.pkl"
    if not plot_only:
        experiment = ICPEVExperiment(obj_factory=obj_factory, gui=gui)
        for seed in range(10):
            test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0, num_points_list=(10,),
                     icp_method=icp.ICPMethod.MEDIAL_CONSTRAINT,
                     name=f"freespace baseline")
        experiment.close()

        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=100,
        #              icp_method=icp.ICPMethod.VOLUMETRIC, freespace_x_filter_threshold=-10,
        #              name=f"comparison 100 free pts all around")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=100,
        #              icp_method=icp.ICPMethod.VOLUMETRIC, freespace_x_filter_threshold=0.,
        #              name=f"comparison 100 free pts")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.VOLUMETRIC_NO_FREESPACE,
        #              name=f"comparison")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.ICP_SGD,
        #              name=f"comparison")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.ICP,
        #              name=f"comparison")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.ICP_REVERSE,
        #              name=f"comparison")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.ICP_SGD_REVERSE,
        #              name=f"comparison")
        # experiment.close()

        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=0,
        #              icp_method=icp.ICPMethod.VOLUMETRIC_ICP_INIT,
        #              name=f"comparison")
        # experiment.close()
        # experiment = ICPEVExperiment(obj_factory=obj_factory, device="cuda", gui=gui)
        # for seed in range(10):
        #     test_icp(experiment, seed=seed, register_num_points=500, num_freespace=100,
        #              icp_method=icp.ICPMethod.VOLUMETRIC_ICP_INIT,
        #              name=f"comparison 100 free pts")
        # experiment.close()

    def filter_names(df):
        df = df[df["name"].str.contains("comparison")]
        return df

    def filter_names_and_x(df):
        df = filter_names(df)
        df = df[df["points"] < 40]
        return df

    plot_icp_results(filter=filter_names, icp_res_file=file)
    plot_icp_results(filter=filter_names_and_x, icp_res_file=file)


def plot_sdf(experiment: ICPEVExperiment, filter_pts=None):
    vis = experiment.dd
    experiment.obj_factory.draw_mesh(experiment.dd, "objframe", ([0, 0, 0], [0, 0, 0, 1]), (0.3, 0.3, 0.3, 0.5),
                                     object_id=experiment.dd.USE_DEFAULT_ID_FOR_NAME)
    # TODO figure out why we're getting out of bound issues when using the sdf range as the input
    coords, pts = util.get_coordinates_and_points_in_grid(experiment.sdf.resolution, experiment.sdf.ranges)
    if filter_pts is not None:
        pts = filter_pts(pts)
    sdf_val, sdf_grad = experiment.sdf(pts)

    # color code them
    error_norm = matplotlib.colors.Normalize(vmin=sdf_val.min(), vmax=sdf_val.max())
    color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
    rgb = color_map.to_rgba(sdf_val.reshape(-1))
    rgb = rgb[:, :-1]

    for i in range(len(pts)):
        vis.draw_point(f"sdf_pt.{i}", pts[i],
                       color=rgb[i], length=0.003)
        vis.draw_2d_line(f"sdf_n.{i}", pts[i], sdf_grad[i],
                         color=rgb[i], size=1., scale=0.01)
    input("finished")


def plot_exported_pcd(env: poke.PokeEnv, seed=0):
    target_file = os.path.join(cfg.DATA_DIR, f"poke/{env.level.name}.txt")
    source_file = os.path.join(cfg.DATA_DIR, f"poke/{env.level.name}_{seed}.txt")

    with open(target_file) as f:
        num = f.readline()
        points = f.readlines()
        points = np.array([[float(v) for v in line.strip().split()] for line in points])
        env.draw_user_text(f"target pcd", xy=[-0.3, 1., -0.5])
        for i, pt in enumerate(points):
            if pt[-1] == 0:
                c = (1, 0, 1)
            else:
                c = (0, 1, 1)
            env.vis.draw_point(f"target.{i}", pt[:3], color=c, scale=2, length=0.003)
        input()
    env.vis.clear_visualizations()

    with open(source_file) as f:
        data = f.readlines()
        j = 0
        while j < len(data):
            line = data[j]
            pokes, num = [int(v) for v in line.strip().split()]
            points = np.array([[float(v) for v in line.strip().split()] for line in data[j + 1:j + num + 1]])
            j += num + 1
            env.draw_user_text(f"source pcd {pokes}", xy=[-0.4, 1., -0.4])
            for i, pt in enumerate(points):
                if pt[-1] == 0:
                    c = (1, 0, 1)
                else:
                    c = (0, 1, 1)
                env.vis.draw_point(f"source.{i}", pt[:3], color=c, scale=2, length=0.003)
            input()


def plot_poke_results(args):
    def filter(df):
        # df = df[df["level"].str.contains(level.name) & (
        #         (df["method"] == "VOLUMETRIC") | (df["method"] == "VOLUMETRIC_SVGD") | (
        #         df["method"] == "VOLUMETRIC_CMAES"))]
        # df = df[(df["level"] == level.name) & (df["method"].str.contains("VOLUMETRIC"))]

        # show each level individually or marginalize over all of them
        # df = df[(df["level"] == level.name)]
        df = df[(df["level"].str.contains(level.name))]

        return df

    def filter_single(df):
        # df = df[(df["level"] == level.name) & (df["seed"] == 0) & (df["method"] == "VOLUMETRIC")]
        df = df[(df["level"] == level.name) & (df["seed"] == args.seed[0])]
        return df

    plot_icp_results(filter=filter, icp_res_file=f"poking_{obj_factory.name}.pkl",
                     key_columns=PokeRunner.KEY_COLUMNS,
                     logy=True, keep_lowest_y_wrt="rmse",
                     save_path=os.path.join(cfg.DATA_DIR, f"img/{level.name.lower()}.png"),
                     show=not args.no_gui,
                     plot_median=False, x='poke', y='chamfer_err')


parser = argparse.ArgumentParser(description='Object registration from contact')
parser.add_argument('experiment',
                    choices=['build', 'plot-sdf', 'globalmin', 'baseline', 'random-sample', 'freespace', 'poke',
                             'poke-visualize-sdf', 'poke-visualize-pcd', 'poke-results',
                             'generate-plausible-set', 'plot-plausible-set', 'evaluate-plausible-diversity',
                             'debug'],
                    help='which experiment to run')
registration_map = {m.name.lower().replace('_', '-'): m for m in icp.ICPMethod}
parser.add_argument('--registration',
                    choices=registration_map.keys(),
                    default='volumetric',
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
task_map = {level.name.lower(): level for level in poke.Levels}
parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
parser.add_argument('--plot_only', action='store_true',
                    help='plot only (previous) results without running any experiments')
parser.add_argument('--read_stored', action='store_true', help='read and process previously output results rather than'
                                                               ' rerunning where possible')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    tracking_method_name = args.tracking
    registration_method = registration_map[args.registration]
    obj_name = level_to_obj_map[level]
    obj_factory = obj_factory_map(obj_name)

    rand.seed(0)

    # -- Build object models (sample points from their surface)
    if args.experiment == "build":
        experiment = ICPEVExperiment(obj_factory=obj_factory, clean_cache=True, gui=not args.no_gui)
        experiment.draw_mesh(name='objframe', pose=([0, 0, 0], [0, 0, 0, 1]), rgba=(1, 1, 1, 0.5),
                             object_id=experiment.dd.USE_DEFAULT_ID_FOR_NAME)
        # for num_points in (5, 10, 20, 30, 40, 50, 100):
        for num_points in (2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500):
            for seed in range(10):
                build_model(obj_factory, experiment.dd, args.task, seed=seed, num_points=num_points,
                            pause_at_end=False)
    elif args.experiment == "plot-sdf":
        experiment = ICPEVExperiment(obj_factory=obj_factory)


        def filter(pts):
            c1 = (pts[:, 0] > -0.15) & (pts[:, 0] < 0.15)
            c2 = (pts[:, 1] > 0.) & (pts[:, 1] < 0.2)
            c3 = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.4)
            # c1 = (pts[:, 0] > -0.2) & (pts[:, 0] < 0.2)
            # c2 = (pts[:, 1] > 0.) & (pts[:, 1] < 0.2)
            # c3 = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.5)
            c = c1 & c2 & c3
            return pts[c][::2]


        plot_sdf(experiment, filter_pts=filter)

    elif args.experiment == "globalmin":
        experiment_ground_truth_initialization_for_global_minima_comparison(obj_factory, plot_only=args.plot_only,
                                                                            gui=not args.no_gui)
    elif args.experiment == "random-sample":
        experiment_vary_num_points_and_num_freespace(obj_factory, plot_only=args.plot_only, gui=not args.no_gui)
    elif args.experiment == "freespace":
        experiment_vary_num_freespace(obj_factory, plot_only=args.plot_only, gui=not args.no_gui)
    elif args.experiment == "baseline":
        experiment_compare_basic_baseline(obj_factory, plot_only=args.plot_only, gui=not args.no_gui)
    elif args.experiment == "poke-visualize-sdf":
        env = PokeGetter.env(level=level, mode=p.GUI, clean_cache=True)
        env.close()
    elif args.experiment == "poke-visualize-pcd":
        env = PokeGetter.env(level=level, mode=p.GUI)
        plot_exported_pcd(env, seed=args.seed[0])
        env.close()
    elif args.experiment == "poke":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=False, device="cuda")
        if registration_method == icp.ICPMethod.NONE:
            runner = ExportProblemRunner(env, tracking_method_name, registration_method, read_stored=False)
        else:
            runner = PokeRunner(env, tracking_method_name, registration_method, ground_truth_initialization=False,
                                read_stored=args.read_stored)
        # backup video logging in case ffmpeg and nvidia driver are not compatible
        # with WindowRecorder(window_names=("Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build",),
        #                     name_suffix="sim", frame_rate=30.0, save_dir=cfg.VIDEO_DIR):
        for seed in args.seed:
            runner.run(name=args.name, seed=seed)

        env.close()
    elif args.experiment == "generate-plausible-set":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=False, device="cuda")
        runner = GeneratePlausibleSetRunner(env, tracking_method_name, registration_method,
                                            ground_truth_initialization=False, read_stored=args.read_stored)
        for seed in args.seed:
            runner.run(name=args.name, seed=seed, draw_text=f"seed {seed} plausible set")

    elif args.experiment == "plot-plausible-set":
        env = PokeGetter.env(level=level, mode=p.GUI, device="cuda")
        runner = PlotPlausibleSetRunner(env, tracking_method_name, registration_method)
        runner.run(seed=0, draw_text=f"plausible set seed 0")

    elif args.experiment == "evaluate-plausible-diversity":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, device="cuda")
        runner = EvaluatePlausibleSetRunner(env, tracking_method_name, registration_method, read_stored=True)
        for seed in args.seed:
            runner.run(name=args.name, seed=seed, draw_text=f"seed {seed}")

    elif args.experiment == "poke-results":
        plot_poke_results(args)

    elif args.experiment == "debug":
        env = PokeGetter.env(level=level, mode=p.GUI, device="cuda")
        debug_volumetric_loss(env)

        # plot_icp_results(icp_res_file=f"poking_{obj_factory.name}.pkl",
        #                  key_columns=("method", "name", "seed", "poke", "batch"),
        #                  plot_median=False, x='poke', y='chamfer_err')

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
