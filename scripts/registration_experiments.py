import typing
import re
import time

import pytorch_kinematics.transforms.rotation_conversions

import pandas as pd
from timeit import default_timer as timer

import argparse
import numpy as np

from chsel import registration_util, costs as icp_costs, initialization, quality_diversity
from pytorch_kinematics import transforms as tf
from sklearn.cluster import Birch, DBSCAN, KMeans

from chsel_experiments.experiments.registration import do_registration, do_medial_constraint_registration
from chsel_experiments.experiments.registration_nopytorch3d import saved_traj_dir_base, saved_traj_file, read_offline_output, \
    build_model, plot_sdf, plot_poke_chamfer_err, plot_poke_plausible_diversity
# marching cubes free surface extraction

from chsel_experiments.registration import registration_method_uses_only_contact_points
from chsel.initialization import initialize_transform_estimates
from pytorch_volumetric import voxel
from chsel_experiments.poking_controller import PokingController

import torch
import pybullet as p
import logging
import os
from datetime import datetime

from matplotlib import pyplot as plt
from arm_pytorch_utilities import tensor_utils, rand

from chsel_experiments import registration
from base_experiments import cfg, serialization
from stucco_experiments.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from chsel_experiments.env import poke
from chsel_experiments.env import obj_factory_map, level_to_obj_map
from chsel_experiments.env_getters.poke import PokeGetter
from pytorch_volumetric.chamfer import batch_chamfer_dist
from base_experiments.sdf import draw_pose_distribution
from pytorch_volumetric.sdf import sample_mesh_points

from arm_pytorch_utilities.controller import Controller
from stucco_experiments.retrieval_controller import TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, PHDFilterTrackingMethod

plt.switch_backend('Qt5Agg')

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)

model_points_dbname = 'model_points_cache.pkl'


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
        poke.Levels.HAMMER_STRAIGHT: ((0, 0.15, -0.15), (0.05, 0.1, 0.2, 0.4)),
        poke.Levels.HAMMER_FALLEN: ((0, 0.15, 0.4, -0.15), (0.05, 0.15, 0.25)),
        poke.Levels.BOX: ((0, 0.18, -0.2), (0.05, 0.15, 0.25, 0.37)),
        poke.Levels.BOX_FALLEN: ((0, 0.15, 0.25, -0.1), (0.05, 0.15, 0.3)),
        poke.Levels.CAN: ((0, 0.2, -0.15), (0.05, 0.12, 0.2, 0.3)),
        poke.Levels.CAN_FALLEN: ((0, 0.15, 0.25, -0.1), (0.05, 0.12, 0.2)),
        poke.Levels.CLAMP: ((0, 0.18, -0.2), (0.05, 0.08, 0.15, 0.25)),
        poke.Levels.CLAMP_SIDEWAYS: ((0, 0.22, -0.23), (0.05, 0.18, 0.26, 0.32)),
    }


def build_model_poke(env: poke.PokeEnv, seed, num_points, pause_at_end=False, device="cpu"):
    return build_model(env.obj_factory, env.vis, env.obj_factory.name, seed, num_points, pause_at_end=pause_at_end,
                       device=device)


def test_icp(env: poke.PokeEnv, seed=0, name="", clean_cache=False, viewing_delay=0.1,
             register_num_points=500, eval_num_points=200,
             num_points_list=(2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100),
             num_freespace=0,
             freespace_x_filter_threshold=0.,  # 0 allows only positive, -10 allows any
             surface_delta=0.025,
             freespace_cost_scale=20,
             ground_truth_initialization=False,
             icp_method=registration.ICPMethod.VOLUMETRIC,
             debug=False):
    obj_name = env.obj_factory.name
    fullname = os.path.join(cfg.DATA_DIR, f'icp_comparison_{obj_name}.pkl')
    if os.path.exists(fullname) and not clean_cache:
        cache = pd.read_pickle(fullname)
    else:
        cache = pd.DataFrame()
    target_obj_id = env.target_object_id()
    vis = env.vis
    freespace_ranges = env.freespace_ranges

    vis.draw_point("seed", (0, 0, 0.4), (1, 0, 0), label=f"seed {seed}")

    dbpath = os.path.join(cfg.DATA_DIR, "model_points_cache.pkl")
    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points, name=obj_name, seed=0,
                                                                  dbpath=dbpath, device=env.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_mesh_points(num_points=register_num_points, dbpath=dbpath,
                                                                          name=obj_name, seed=0, device=env.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    points = []
    points_free = []
    B = 30

    best_tsf_guess = initialization.random_upright_transforms(B, dtype, device)

    for num_points in num_points_list:
        model_points, model_normals, _ = sample_mesh_points(num_points=num_points, name=obj_name, seed=seed,
                                                            dbpath=dbpath, device=env.device)

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

        free_voxels = voxel.VoxelGrid(0.025, freespace_ranges, dtype=dtype, device=device)
        known_sdf = voxel.VoxelSet(model_points_world_frame,
                                   torch.zeros(model_points_world_frame.shape[0], dtype=dtype, device=device))
        volumetric_cost = icp_costs.VolumetricCost(free_voxels, known_sdf, env.target_sdf, scale=1,
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

        points.append(model_points_world_frame)
        points_free.append(free_space_world_frame_points)

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

        errors_per_batch = batch_chamfer_dist(T, model_points_world_frame_eval,
                                              env.obj_factory, viewing_delay,
                                              vis=vis if env.mode == p.GUI else None)
        errors.append(errors_per_batch)

        df = pd.DataFrame(
            {"date": datetime.today(), "method": icp_method.name, "name": name, "seed": seed,
             "points": num_points,
             "points_free": len(free_space_world_frame_points),
             "batch": np.arange(B),
             "chamfer_err": errors_per_batch.cpu().numpy()})
        cache = pd.concat([cache, df])

    cache.to_pickle(fullname)
    for i in range(len(num_points_list)):
        print(f"num {num_points_list[i]} err {errors[i]}")

    # export point cloud per "poke" similar to the actual poking experiment
    export_traj_filename = os.path.join(cfg.DATA_DIR, f"icp_sample/{obj_name}_{seed}.txt")
    os.makedirs(os.path.dirname(export_traj_filename), exist_ok=True)
    with open(export_traj_filename, "w") as f:
        for i in range(len(points)):
            f.write(f"{i} {len(points_free[i]) + len(points[i])}\n")
            serialization.export_pcs(f, points_free[i], points[i])


def marginalize_over_suffix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[:suffix_start_idx - 1] if suffix_start_idx > 0 else "base"


def marginalize_over_prefix(name):
    suffix_start_idx = re.search(r"\d", name).start()
    return name[suffix_start_idx:] if suffix_start_idx > 0 else name


def marginalize_over_registration_num(name):
    registration_num = re.search(r"\d+", name)
    return f"{registration_num[0]} registered points" if registration_num is not None else name


def export_pc_to_register(point_cloud_file: str, pokes: int, env: poke.PokeEnv, method: TrackingMethod):
    # append to the end and create it if it doesn't exist
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'a') as f:
        # write out the poke index and the size of the point cloud
        pc_free, _ = env.free_voxels.get_known_pos_and_values()
        _, pc_occ = method.get_labelled_moved_points()
        total_pts = len(pc_free) + len(pc_occ)
        f.write(f"{pokes} {total_pts}\n")
        serialization.export_pcs(f, pc_free, pc_occ)


def debug_volumetric_loss(env: poke.PokeEnv, seed=0, show_free_voxels=False, pokes=4):
    # load from file
    env.reset()
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

    empty_sdf = voxel.VoxelSet(torch.empty(0), torch.empty(0))
    volumetric_cost = icp_costs.VolumetricCost(env.free_voxels, empty_sdf, env.target_sdf, scale=1,
                                               scale_known_freespace=1, obj_factory=env.obj_factory,
                                               vis=env.vis, debug=True)

    this_pts = tensor_utils.ensure_tensor(device, dtype, pts)
    volumetric_cost.sdf_voxels = voxel.VoxelSet(this_pts,
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
        H = Hgt.clone()
        # breakpoint here and change the transform H with respect to the ground truth one to evaluate the cost
        if H.dim() == 2:
            H = H.unsqueeze(0)
            H.requires_grad = True
        cost = volumetric_cost(H.inverse()[:, :3, :3], H.inverse()[:, :3, -1], None)
        cost.mean().backward()
        volumetric_cost.visualize(H.inverse()[:, :3, :3], H.inverse()[:, :3, -1], None)
        env.draw_user_text("{}".format(cost.item()), xy=(0.5, 0.7, -.5))
        draw_pose_distribution(H, pose_obj_map, env.vis, env.obj_factory)


class PokeRunner:
    KEY_COLUMNS = ("method", "name", "seed", "poke", "level", "batch")

    def __init__(self, env: poke.PokeEnv, tracking_method_name: str, reg_method, B=30,
                 read_stored=False, ground_truth_initialization=False, init_method=initialization.InitMethod.RANDOM,
                 register_num_points=500, start_at_num_pts=4, eval_num_points=200, hide_surroundings=False):
        self.env = env
        self.B = B
        self.dbname = os.path.join(cfg.DATA_DIR, f'poking_{env.obj_factory.name}.pkl')
        self.tracking_method_name = tracking_method_name
        self.reg_method = reg_method
        self.start_at_num_pts = start_at_num_pts
        self.read_stored = read_stored
        self.ground_truth_initialization = ground_truth_initialization
        self.init_method = init_method
        self.hide_surroundings = hide_surroundings

        model_name = self.env.target_model_name
        # get a fixed number of model points to evaluate against (this will be independent on points used to register)
        dbpath = os.path.join(cfg.DATA_DIR, "model_points_cache.pkl")
        self.model_points_eval, self.model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points,
                                                                                name=model_name,
                                                                                seed=0, dbpath=dbpath,
                                                                                device=env.device)
        self.device, self.dtype = self.model_points_eval.device, self.model_points_eval.dtype

        # get a large number of model points to register to
        self.model_points_register, self.model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                                        name=model_name,
                                                                                        seed=0, dbpath=dbpath,
                                                                                        device=env.device)

        # need to get these after
        self.model_points_world_frame_eval = None
        self.model_normals_world_frame_eval = None

        self.draw_pose_distribution_separately = False
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
        self.elapsed = None
        self.to_export = {}
        self.cache = None

    def create_volumetric_cost(self):
        # placeholder for now; have to be filled manually
        empty_sdf = voxel.VoxelSet(torch.empty(0), torch.empty(0))
        self.volumetric_cost = icp_costs.VolumetricDirectSDFCost(self.env.free_voxels, empty_sdf, self.env.target_sdf,
                                                                 scale=1,
                                                                 scale_known_freespace=20,
                                                                 vis=self.env.vis, obj_factory=self.env.obj_factory,
                                                                 debug=False)

    def register_transforms_with_points(self, seed):
        """Exports best_segment_idx, transforms_per_object, and rmse_per_object"""
        # note that we update our registration regardless if we're in contact or not
        self.best_distance = None
        self.dist_per_est_obj = []
        self.transforms_per_object = []
        self.rmse_per_object = []
        self.best_segment_idx = None
        self.elapsed = None
        for k, this_pts in enumerate(self.method):
            N = len(this_pts)
            if N < self.start_at_num_pts or self.pokes < self.start_at_num_pts:
                continue
            # this_pts corresponds to tracked contact points that are segmented together
            this_pts = tensor_utils.ensure_tensor(self.device, self.dtype, this_pts)
            self.volumetric_cost.sdf_voxels = voxel.VoxelSet(this_pts, torch.zeros(this_pts.shape[0], dtype=self.dtype,
                                                                                   device=self.device))

            if self.best_tsf_guess is None:
                self.best_tsf_guess = initialize_transform_estimates(self.B, self.env.freespace_ranges,
                                                                     self.init_method, None,
                                                                     device=self.env.device, dtype=self.env.dtype)
            if self.ground_truth_initialization:
                self.best_tsf_guess = self.link_to_current_tf_gt.get_matrix().repeat(self.B, 1, 1)

            # avoid giving methods that don't use freespace more training iterations
            if registration_method_uses_only_contact_points(self.reg_method) and N in self.num_points_to_T_cache:
                T, distances = self.num_points_to_T_cache[N]
            else:
                if self.read_stored or self.reg_method == registration.ICPMethod.CVO:
                    T, distances, self.elapsed = read_offline_output(self.reg_method, self.env.level, seed, self.pokes)
                    T = T.to(device=self.device, dtype=self.dtype)
                    distances = distances.to(device=self.device, dtype=self.dtype)
                elif self.reg_method in [registration.ICPMethod.MEDIAL_CONSTRAINT, registration.ICPMethod.MEDIAL_CONSTRAINT_CMAME]:
                    T, distances, self.elapsed = do_medial_constraint_registration(self.reg_method, this_pts,
                                                                                   self.volumetric_cost.sdf,
                                                                                   self.best_tsf_guess, self.B,
                                                                                   self.env.level,
                                                                                   seed, self.pokes,
                                                                                   vis=self.env.vis,
                                                                                   obj_factory=self.env.obj_factory)
                else:
                    start = timer()
                    T, distances = do_registration(this_pts, self.model_points_register, self.best_tsf_guess, self.B,
                                                   self.volumetric_cost,
                                                   self.reg_method)
                    end = timer()
                    self.elapsed = end - start
                self.num_points_to_T_cache[N] = T, distances
                logger.info("registration elapsed %fs", self.elapsed)

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
        self.best_tsf_guess = initialization.reinitialize_transform_estimates(self.B, self.best_tsf_guess)
        return self.best_tsf_guess

    def evaluate_registrations(self):
        """Responsible for populating to_export"""
        self.method.register_transforms(self.transforms_per_object[self.best_segment_idx], self.best_tsf_guess)
        logger.debug(f"err each obj {np.round(self.dist_per_est_obj, 4)}")
        best_T = self.best_tsf_guess

        self.reinitialize_best_tsf_guess()

        T = self.transforms_per_object[self.best_segment_idx]

        # when evaluating, move the best guess pose far away to improve clarity
        self.env.pause()
        self.env.draw_mesh("base_object", ([0, 0, 100], [0, 0, 0, 1]), (0.0, 0.0, 1., 0.5),
                           object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
        if self.draw_pose_distribution_separately:
            evaluate_chamfer_dist_extra_args = [self.env.obj_factory,
                                                0.05,
                                                False]
        else:
            draw_pose_distribution(T.inverse(), self.pose_obj_map, self.env.vis, self.env.obj_factory)
            evaluate_chamfer_dist_extra_args = [self.env.obj_factory, 0., False]

        # evaluate with chamfer distance
        errors_per_batch = batch_chamfer_dist(T, self.model_points_world_frame_eval,
                                              *evaluate_chamfer_dist_extra_args,
                                              vis=self.env.vis if self.env.mode == p.GUI else None)
        self.env.unpause()

        link_to_current_tf = tf.Transform3d(matrix=T)
        interior_pts = link_to_current_tf.transform_points(self.volumetric_cost.model_interior_points_orig)
        occupied = self.env.free_voxels[interior_pts]

        self.chamfer_err.append(errors_per_batch)
        self.num_freespace_voxels.append(self.env.free_voxels.get_known_pos_and_values()[0].shape[0])
        self.freespace_violations.append(occupied.sum(dim=-1).detach().cpu())
        logger.info(f"chamfer distance {self.pokes}: {torch.mean(errors_per_batch)}")

        # draw mesh at where our best guess is
        guess_pose = pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(best_T)
        # self.env.draw_mesh("base_object", guess_pose, (0.0, 0.0, 1., 0.5),
        #                    object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)

    def export_metrics(self, cache, name, seed):
        """Responsible for populating to_export and saving to database"""
        _c = np.array(self.chamfer_err[-1].cpu().numpy())
        _f = np.array(self.freespace_violations[-1])
        _n = self.num_freespace_voxels[-1]
        _r = _f / _n
        batch = np.arange(self.B)
        rmse = self.rmse_per_object[self.best_segment_idx]

        df = pd.DataFrame(
            {"date": datetime.today(), "method": self.reg_method.name, "level": self.env.level.name,
             "name": name,
             "seed": seed, "poke": self.pokes,
             "batch": batch,
             "chamfer_err": _c, 'freespace_violations': _f,
             'num_freespace_voxels': _n,
             "freespace_violation_percent": _r,
             "rmse": rmse.cpu().numpy(),
             "elapsed": self.elapsed,
             })
        cache = pd.concat([cache, df])
        cache.to_pickle(self.dbname)
        # additional data to export fo file
        self.to_export[self.pokes] = {
            'T': self.transforms_per_object[self.best_segment_idx],
            'rmse': rmse,
            'elapsed': self.elapsed,
        }
        return cache

    def run(self, name="", seed=0, ctrl_noise_max=0.005, draw_text=None):
        quality_diversity.previous_solutions = None
        if os.path.exists(self.dbname):
            self.cache = pd.read_pickle(self.dbname)
        else:
            self.cache = pd.DataFrame()

        env = self.env
        # if draw_text is None:
        #     self.env.draw_user_text(f"{self.reg_method.name}{name} seed {seed}", xy=[-0.3, 1., -0.5])
        # else:
        #     self.env.draw_user_text(draw_text, xy=[-0.3, 1., -0.5])

        obs = self.env.reset()
        if self.hide_surroundings:
            p.removeBody(self.env.planeId)
            for obj in self.env.objects:
                if obj != self.env.target_object_id():
                    p.removeBody(obj)
            self.env.objects = []
            self.env.immovable = []
            self.env.movable = []
        self.create_volumetric_cost()
        self.method = create_tracking_method(self.env, self.tracking_method_name)
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
        self.num_points_to_T_cache = {}
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
            # env.draw_user_text("{}".format(self.pokes), xy=(0.5, 0.4, -0.5))

            action = self.ctrl.command(obs, info)
            self.method.visualize_contact_points(env)

            if action is None:
                self.pokes += 1
                self.hook_after_poke(name, seed)
            registration_util.poke_index = self.pokes

            if action is not None:
                if torch.is_tensor(action):
                    action = action.cpu()

                action = np.array(action, dtype=float).flatten()
                action += action_noise[simTime]
                obs, rew, done, info = env.step(action)

        self.env.vis.clear_visualizations()
        self.hook_after_last_poke(seed)

    # hooks for derived classes to add behavior at specific locations
    def hook_before_first_poke(self, seed):
        pose = p.getBasePositionAndOrientation(self.env.target_object_id())
        self.link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        self.model_points_world_frame_eval = self.link_to_current_tf_gt.transform_points(self.model_points_eval)
        self.model_normals_world_frame_eval = self.link_to_current_tf_gt.transform_normals(self.model_normals_eval)

    def hook_after_poke(self, name, seed):
        self.register_transforms_with_points(seed)
        # has at least one contact segment
        if self.best_segment_idx is not None:
            self.evaluate_registrations()
            self.cache = self.export_metrics(self.cache, name, seed)

    def hook_after_last_poke(self, seed):
        if not self.read_stored:
            serialization.export_registration(saved_traj_file(self.reg_method, self.env.level, seed), self.to_export)


class ExportProblemRunner(PokeRunner):
    """If registration method is None, export the point clouds, initial transforms, and ground truth transforms"""

    def __init__(self, *args, export_freespace_surface=True, **kwargs):
        super(ExportProblemRunner, self).__init__(*args, **kwargs)
        self.reg_method = registration.ICPMethod.NONE
        self.pc_to_register_file = None
        self.free_surface_file = None
        self.transform_file = None
        self.export_free_surface = export_freespace_surface

    def hook_before_first_poke(self, seed):
        super().hook_before_first_poke(seed)
        self.pc_to_register_file = f"{saved_traj_dir_base(self.env.level)}_{seed}.txt"
        pc_register_against_file = f"{saved_traj_dir_base(self.env.level)}.txt"
        self.transform_file = f"{saved_traj_dir_base(self.env.level)}_{seed}_trans.txt"
        self.free_surface_file = f"{saved_traj_dir_base(self.env.level)}_{seed}_free_surface.txt"
        transform_gt_file = f"{saved_traj_dir_base(self.env.level)}_{seed}_gt_trans.txt"
        # exporting data for offline baselines, remove the stale file
        serialization.export_pc_register_against(pc_register_against_file, self.env.target_sdf)
        serialization.export_init_transform(transform_gt_file,
                                            self.link_to_current_tf_gt.get_matrix().repeat(self.B, 1, 1))
        try:
            os.remove(self.pc_to_register_file)
        except OSError:
            pass
        try:
            os.remove(self.free_surface_file)
        except OSError:
            pass

    def hook_after_poke(self, name, seed):
        if self.pokes >= self.start_at_num_pts:
            if self.best_tsf_guess is None:
                self.best_tsf_guess = initialize_transform_estimates(self.B, self.env.freespace_ranges,
                                                                     self.init_method, None,
                                                                     device=self.env.device, dtype=self.env.dtype)
                serialization.export_init_transform(self.transform_file, self.best_tsf_guess)
            export_pc_to_register(self.pc_to_register_file, self.pokes, self.env, self.method)
            self.do_export_free_surface()
            logger.info(f"Export poke {self.pokes} data for {self.env.level.name} seed {seed}")

    def do_export_free_surface(self, debug=False):
        if not self.export_free_surface:
            return
        serialization.export_free_surface(self.free_surface_file, self.env.free_voxels, self.pokes,
                                          self.env.vis if debug else None)


class PlausibleSetRunner(PokeRunner):
    def plausible_set_filename(self, seed):
        return f"{saved_traj_dir_base(self.env.level)}_plausible_set_{seed}.pkl"


class GeneratePlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, gt_position_max_offset=0.2, position_steps=15, N_rot=10000,
                 max_plausible_set=1000,  # prevent running out of memory and very long evaluation times
                 **kwargs):
        super(GeneratePlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plausible_suboptimality = self.env.obj_factory.plausible_suboptimality
        self.max_plausibile_set = max_plausible_set
        # self.gt_position_max_offset = gt_position_max_offset
        # self.position_steps = position_steps
        self.N_rot = N_rot

        # maps poke number to a set of plausible transforms
        self.plausible_set = {}
        # without max restriction
        self.plausible_set_all = {}
        self.gt_position_max_offset = gt_position_max_offset
        self.position_steps = position_steps

        # variations
        self.pos = None
        self.rot = None

        # ground truth transforms - need the environment to reset first to simulate what happens in a run
        # therefore need to get them inside the hook before first poke rather than in the constructor
        self.Hgt = None
        self.Hgtinv = None

        self.contact_pts = None

    # needs more tuning for the suboptimality for this loss
    def create_volumetric_cost(self):
        # placeholder for now; have to be filled manually
        empty_sdf = voxel.VoxelSet(torch.empty(0), torch.empty(0))
        self.volumetric_cost = icp_costs.DiscreteNondifferentiableCost(self.env.free_voxels, empty_sdf,
                                                                       self.env.target_sdf,
                                                                       cmax=1000, vis=self.env.vis,
                                                                       obj_factory=self.env.obj_factory)

    def hook_before_first_poke(self, seed):
        super(GeneratePlausibleSetRunner, self).hook_before_first_poke(seed)
        with rand.SavedRNG():
            rand.seed(0)

            pose = self.env.target_pose
            gt_tf = tf.Transform3d(pos=pose[0],
                                   rot=tf.xyzw_to_wxyz(tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])),
                                   dtype=self.dtype, device=self.device)
            self.Hgt = gt_tf.get_matrix()
            self.Hgtinv = self.Hgt.inverse()

            # assume that there can only be plausible completions close enough to the ground truth
            target_pos = pose[0]
            offset = self.gt_position_max_offset / self.position_steps
            x1 = torch.linspace(target_pos[0] - self.gt_position_max_offset * 0.5,
                                target_pos[0],
                                steps=self.position_steps // 3, device=self.device)
            x2 = torch.linspace(target_pos[0] + offset,
                                target_pos[0] + self.gt_position_max_offset * 1.5,
                                steps=self.position_steps // 3 * 2, device=self.device)
            x = torch.cat((x1, x2))
            y1 = torch.linspace(target_pos[1] - self.gt_position_max_offset, target_pos[1],
                                steps=self.position_steps // 2, device=self.device)
            y2 = torch.linspace(target_pos[1] + offset, target_pos[1] + self.gt_position_max_offset,
                                steps=self.position_steps // 2, device=self.device)
            y = torch.cat((y1, y2))
            z = torch.linspace(target_pos[2], target_pos[2] + self.gt_position_max_offset * 0.5,
                               steps=self.position_steps, device=self.device)
            self.pos = torch.cartesian_prod(x, y, z)
            self.N_pos = len(self.pos)
            # uniformly sample rotations
            self.rot = tf.random_rotations(self.N_rot, device=self.device)
            # we know most of the ground truth poses are actually upright, so let's add those in as hard coded
            N_upright = min(100, self.N_rot)
            axis_angle = torch.zeros((N_upright, 3), dtype=self.dtype, device=self.device)
            axis_angle[:, -1] = torch.linspace(0, 2 * np.pi, N_upright)
            self.rot[:N_upright] = tf.axis_angle_to_matrix(axis_angle)
            # ensure the ground truth rotation is sampled
            self.rot[N_upright] = self.Hgt[:, :3, :3]

    def hook_after_poke(self, name, seed):
        # assume all contact points belong to the object
        contact_pts = []
        for k, this_pts in enumerate(self.method):
            contact_pts.append(this_pts)
        self.contact_pts = tensor_utils.ensure_tensor(self.device, self.dtype, torch.cat(contact_pts))
        if len(self.contact_pts) < self.start_at_num_pts or self.pokes < self.start_at_num_pts:
            return

        self.volumetric_cost.sdf_voxels = voxel.VoxelSet(self.contact_pts,
                                                         torch.zeros(self.contact_pts.shape[0], dtype=self.dtype,
                                                                     device=self.device))
        self.evaluate_registrations()

    def _evaluate_transforms(self, transforms):
        return self.volumetric_cost(transforms[:, :3, :3], transforms[:, :3, -1], None)

    def evaluate_registrations(self):
        # evaluate all the transforms
        plausible_transforms = []
        gt_cost = self._evaluate_transforms(self.Hgtinv)

        # if we're doing it for the first time we need to evaluate over everything
        # if we've narrowed it down to a small number of plausible transforms, we only need to keep evaluating those
        # since new pokes can only prune previously plausible transforms
        if len(self.plausible_set) > 0:
            trans_chunk = 500
            Hall = self.plausible_set_all[self.pokes - 1]
            for i in range(0, Hall.shape[0], trans_chunk):
                H = Hall[i:i + trans_chunk]
                Hinv = H.inverse()
                costs = self._evaluate_transforms(Hinv)
                plausible = costs < self.plausible_suboptimality + gt_cost

                if torch.any(plausible):
                    Hp = H[plausible]
                    plausible_transforms.append(Hp)
        else:
            # evaluate the pts in chunks since we can't load all points in memory at the same time
            rot_chunk = 1
            pos_chunk = 1000
            for i in range(0, self.N_rot, rot_chunk):
                logger.debug(f"chunked {i}/{self.N_rot} plausible: {sum(h.shape[0] for h in plausible_transforms)}")
                min_cost_per_chunk = 100000
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

                    costs = self._evaluate_transforms(Hinv)
                    plausible = costs < self.plausible_suboptimality + gt_cost
                    min_cost_per_chunk = min(min_cost_per_chunk, costs.min())

                    if torch.any(plausible):
                        Hp = H[plausible]
                        plausible_transforms.append(Hp)
                logger.debug(f"min cost for chunk: {min_cost_per_chunk}")

        all_plausible_transforms = torch.cat(plausible_transforms)
        self.plausible_set_all[self.pokes] = all_plausible_transforms
        if len(all_plausible_transforms) > self.max_plausibile_set:
            to_keep = torch.randperm(len(all_plausible_transforms))[:self.max_plausibile_set]
            all_plausible_transforms = all_plausible_transforms[to_keep]
        self.plausible_set[self.pokes] = all_plausible_transforms
        logger.info("poke %d with %d (%d) plausible completions gt cost: %f allowable cost: %f", self.pokes,
                    all_plausible_transforms.shape[0], self.plausible_set_all[self.pokes].shape[0], gt_cost,
                    gt_cost + self.plausible_suboptimality)
        # plot plausible transforms
        to_plot = torch.randperm(len(all_plausible_transforms))[:200]
        self.env.pause()
        draw_pose_distribution(all_plausible_transforms[to_plot], self.pose_obj_map, self.env.vis, self.env.obj_factory,
                               show_only_latest=True, sequential_delay=0.1)
        self.env.unpause()

    def hook_after_last_poke(self, seed):
        # export plausible set to file
        filename = self.plausible_set_filename(seed)
        logger.info("saving plausible set to %s", filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.plausible_set, filename)


class PlotPlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, show_only_latest=False, sequential_delay=0.0, max_shown=16, **kwargs):
        super(PlotPlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plausible_set = {}
        self.max_shown = max_shown
        self.show_only_latest = show_only_latest
        self.sequential_delay = sequential_delay

    def hook_before_first_poke(self, seed):
        filename = self.plausible_set_filename(seed)
        self.plausible_set = torch.load(filename)

    def hook_after_poke(self, name, seed):
        self.env.pause()
        if self.pokes in self.plausible_set:
            ps = self.plausible_set[self.pokes]
            if not self.show_only_latest:
                self.env.vis.clear_visualizations()
                self.pose_obj_map = {}
            idx = np.random.permutation(range(len(ps)))
            ps = ps[idx[:self.max_shown]]
            draw_pose_distribution(ps, self.pose_obj_map, self.env.vis, self.env.obj_factory,
                                   show_only_latest=self.show_only_latest, sequential_delay=self.sequential_delay)
        self.env.unpause()

    def hook_after_last_poke(self, seed):
        pass


class EvaluatePlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, plot_meshes=True, sleep_between_plots=0.1, **kwargs):
        super(EvaluatePlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plot_meshes = plot_meshes and self.env.mode == p.GUI
        self.sleep_between_plots = sleep_between_plots
        # always read stored with the plausible set evaluation
        self.read_stored = True
        self.plausible_set = {}
        self.plausibility = None
        self.coverage = None

    def hook_before_first_poke(self, seed):
        super(EvaluatePlausibleSetRunner, self).hook_before_first_poke(seed)
        # TODO actually pass seed in if we're generating different plausible sets per seed
        # they are different trajectories due to control noise, but they are almost the exact same
        filename = self.plausible_set_filename(0)
        self.plausible_set = torch.load(filename)
        self.cache = self.cache.drop_duplicates(subset=self.KEY_COLUMNS, keep='last')

    def hook_after_poke(self, name, seed):
        if self.pokes in self.plausible_set:
            self.register_transforms_with_points(seed)
            # has at least one contact segment
            if self.best_segment_idx is None:
                logger.warning("No sufficient contact segment on poke %d despite having data for the plausible set",
                               self.pokes)
                return
            self.evaluate_registrations()
            self.cache = self.export_metrics(self.cache, name, seed)

    def hook_after_last_poke(self, seed):
        pass

    def _do_evaluate_plausible_diversity_on_best_quantile(self, errors_per_batch):
        B, P = errors_per_batch.shape

        best_per_sampled = errors_per_batch.min(dim=1)
        best_per_plausible = errors_per_batch.min(dim=0)

        bp_plausibility = best_per_sampled.values.sum() / B
        bp_coverage = best_per_plausible.values.sum() / P

        return bp_plausibility, bp_coverage, best_per_sampled, best_per_plausible

    def evaluate_registrations(self):
        """Responsible for populating to_export"""
        self.method.register_transforms(self.transforms_per_object[self.best_segment_idx], self.best_tsf_guess)
        logger.debug(f"err each obj {np.round(self.dist_per_est_obj, 4)}")

        # sampled transforms and all plausible transforms
        T = self.transforms_per_object[self.best_segment_idx]
        Tp = self.plausible_set[self.pokes]
        rmse = self.rmse_per_object[self.best_segment_idx]

        # NOTE that T is already inverted, so no need for T.inverse() * T
        # however, Tinv is useful for plotting purposes
        Tinv = T.inverse()
        # effectively can apply one transform then take the inverse using the other one; if they are the same, then
        # we should end up in the base frame if that T == Tp
        # want pairwise matrix multiplication |T| x |Tp| x 4 x 4 T[0]@Tp[0], T[0]@Tp[1]
        # Iapprox = torch.einsum("bij,pjk->bpik", T, Tp)
        # the einsum does the multiplication below and is about twice as fast
        Iapprox = T.view(-1, 1, 4, 4) @ Tp.view(1, -1, 4, 4)

        B, P = Iapprox.shape[:2]
        errors_per_batch = batch_chamfer_dist(Iapprox.view(B * P, 4, 4), self.model_points_eval,
                                              self.env.obj_factory, 0, vis=None)
        errors_per_batch = errors_per_batch.view(B, P)

        # when evaluating, move the best guess pose far away to improve clarity
        self.env.draw_mesh("base_object", ([0, 0, 100], [0, 0, 0, 1]), (0.0, 0.0, 1., 0.5),
                           object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)

        self.plausibility = {}
        self.coverage = {}
        for quantile in [1.0, 0.75, 0.5]:
            thresh = torch.quantile(rmse, quantile)
            used = rmse <= thresh
            to_plot = quantile == 1.0
            quantile_errors_per_batch = errors_per_batch[used]
            bp_plausibility, bp_coverage, best_per_sampled, best_per_plausible = self._do_evaluate_plausible_diversity_on_best_quantile(
                quantile_errors_per_batch)
            logger.info(f"pokes {self.pokes} quantile {quantile} BP plausibility {bp_plausibility.item():.0f} "
                        f"coverage {bp_coverage.item():.0f} against {P} plausible transforms")
            self.plausibility[quantile] = bp_plausibility
            self.coverage[quantile] = bp_coverage

            # sampled vs closest plausible transform
            if self.plot_meshes and to_plot:
                self.env.draw_user_text("sampled (green) vs closest plausible (blue)", xy=[-0.1, -0.1, -0.5])
                for b in range(B):
                    p = best_per_sampled.indices[b]
                    self.env.draw_user_text(f"{best_per_sampled.values[b].item():.0f}", xy=[-0.1, -0.2, -0.5])
                    self.env.draw_mesh("sampled", pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(Tinv[b]), (0.0, 1.0, 0., 0.5),
                                       object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                    self.env.draw_mesh("plausible", pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(Tp[p]), (0.0, 0.0, 1., 0.5),
                                       object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                    time.sleep(self.sleep_between_plots)

                self.env.draw_user_text("plausible (blue) vs closest sampled (blue)", xy=[-0.1, -0.1, -0.5])
                for p in range(P):
                    b = best_per_plausible.indices[p]
                    self.env.draw_user_text(f"{best_per_plausible.values[p].item():.0f}", xy=[-0.1, -0.2, -0.5])
                    self.env.draw_mesh("sampled", pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(Tinv[b]), (0.0, 1.0, 0., 0.5),
                                       object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                    self.env.draw_mesh("plausible", pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(Tp[p]), (0.0, 0.0, 1., 0.5),
                                       object_id=self.env.vis.USE_DEFAULT_ID_FOR_NAME)
                    time.sleep(self.sleep_between_plots)

    def export_metrics(self, cache, name, seed):
        """Responsible for populating to_export and saving to database"""
        batch = np.arange(self.B)
        data = {"method": self.reg_method.name, "level": self.env.level.name,
                "name": name,
                "seed": seed, "poke": self.pokes,
                "batch": batch,
                }
        for quantile in self.plausibility.keys():
            data[f"plausibility_q{quantile}"] = self.plausibility[quantile].item()
            data[f"coverage_q{quantile}"] = self.coverage[quantile].item()
            data[f"plausible_diversity_q{quantile}"] = (self.plausibility[quantile] + self.coverage[quantile]).item()

        df = pd.DataFrame(data)
        cols = list(cache.columns)
        # filter out stale columns (seems to be interferring with combine_first)
        cols = [c for c in cols if c not in ('plausibility', 'coverage', 'plausible_diversity')]
        cache = cache[cols]
        cols_next = [c for c in df.columns if c not in cols]
        if "plausibility_q1.0" not in cols:
            dd = pd.merge(cache, df, how="outer", suffixes=('', '_y'))
        else:
            dd = pd.merge(cache, df, on=self.KEY_COLUMNS, how='outer')
            # combine shared columns
            pd_cols = [c for c in df.columns if ('plausi' in c) or ('coverage' in c)]
            for c in pd_cols:
                if c + "_x" in dd.columns:
                    x = c + "_x"
                    y = c + "_y"
                    dd[c] = dd[y].fillna(dd[x])
                    dd.drop([x, y], axis=1, inplace=True)

        # rearrange back in proper order
        dd = dd[cols + cols_next]
        dd.to_pickle(self.dbname)

        return dd


def create_tracking_method(env, method_name) -> TrackingMethod:
    if method_name == "ours":
        return OurSoftTrackingMethod(env, PokeGetter.contact_parameters(env), poke.ArmMovableSDF(env), dim=3)
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


def plot_exported_pcd(env: poke.PokeEnv, seed=0, surface_only=True):
    target_file = os.path.join(cfg.DATA_DIR, f"poke/{env.level.name}.txt")
    source_file = os.path.join(cfg.DATA_DIR, f"poke/{env.level.name}_{seed}.txt")

    with open(target_file) as f:
        num = f.readline()
        points = f.readlines()
        points = np.array([[float(v) for v in line.strip().split()] for line in points])
        env.draw_user_text(f"target pcd", xy=[-0.3, 1., -0.5])
        for i, pt in enumerate(points):
            if pt[-1] == 0:
                if surface_only:
                    continue
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


def main(args):
    level = task_map[args.task]
    tracking_method_name = args.tracking
    registration_method = registration_map[args.registration]
    obj_name = level_to_obj_map[level]
    obj_factory = obj_factory_map(obj_name)

    rand.seed(0)
    logger.info(
        f"--experiment {args.experiment} --registration {args.registration} --task {level.name} --seeds {args.seed}")

    # -- Build object models (sample points from their surface)
    if args.experiment == "build":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=True, device="cuda")
        env.draw_mesh(name='objframe', pose=([0, 0, 0], [0, 0, 0, 1]), rgba=(1, 1, 1, 0.5),
                      object_id=env.vis.USE_DEFAULT_ID_FOR_NAME)
        cache = None
        model_points_cache = os.path.join(cfg.DATA_DIR, model_points_dbname)
        if os.path.exists:
            cache = torch.load(model_points_cache)
        for num_points in (2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500):
            for seed in args.seed:
                cache = build_model(env.obj_factory, env.vis, args.task, seed=seed, num_points=num_points,
                                    pause_at_end=False, cache=cache)
        torch.save(cache, model_points_cache)
    elif args.experiment == "plot-sdf":
        env = PokeGetter.env(level=level, mode=p.GUI, device="cuda")

        def filter(pts):
            c1 = (pts[:, 0] > -0.15) & (pts[:, 0] < 0.15)
            c2 = (pts[:, 1] > 0.) & (pts[:, 1] < 0.2)
            c3 = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.4)
            c = c1 & c2 & c3
            return pts[c][::2]

        plot_sdf(env.obj_factory, env.target_sdf, env.vis, filter_pts=filter)
    elif args.experiment == "poke-visualize-sdf":
        env = PokeGetter.env(level=level, mode=p.GUI, clean_cache=True)
        env.close()
    elif args.experiment == "poke-visualize-pcd":
        env = PokeGetter.env(level=level, mode=p.GUI)
        plot_exported_pcd(env, seed=args.seed[0])
        env.close()
    elif args.experiment == "poke":
        env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=False, device="cuda")
        if registration_method == registration.ICPMethod.NONE:
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

    elif args.experiment == "plot-poke-ce":
        plot_poke_chamfer_err(args, level, obj_factory, PokeRunner.KEY_COLUMNS)

    elif args.experiment == "plot-poke-pd":
        plot_poke_plausible_diversity(args, level, obj_factory, PokeRunner.KEY_COLUMNS, quantile=1.0, fmt='line',
                                      legend=True)

    elif args.experiment == "debug":
        env = PokeGetter.env(level=level, mode=p.GUI, device="cuda")
        debug_volumetric_loss(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object registration from contact')
    parser.add_argument('experiment',
                        choices=['build', 'plot-sdf', 'globalmin', 'baseline', 'random-sample', 'freespace', 'poke',
                                 'poke-visualize-sdf', 'poke-visualize-pcd',
                                 'plot-poke-ce', 'plot-poke-pd',
                                 'generate-plausible-set', 'plot-plausible-set', 'evaluate-plausible-diversity',
                                 'debug'],
                        help='which experiment to run')
    registration_map = {m.name.lower().replace('_', '-'): m for m in registration.ICPMethod}
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
    parser.add_argument('--read_stored', action='store_true',
                        help='read and process previously output results rather than'
                             ' rerunning where possible')
    parser.add_argument('--marginalize', action='store_true',
                        help='average results across configurations for each object for plotting')

    args = parser.parse_args()

    main(args)
