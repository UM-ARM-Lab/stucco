import os
import typing
import pandas as pd
import argparse
import torch
import numpy as np
from timeit import default_timer as timer

import stucco.icp.initialization
from arm_pytorch_utilities.controller import Controller
from arm_pytorch_utilities import tensor_utils, rand

import pytorch_kinematics as tf

from stucco import cfg, registration_util
from stucco import icp
from stucco import voxel
from stucco import sdf
from stucco.evaluation import evaluate_chamfer_distance
from stucco.env import poke_real_nonros
from stucco.icp import costs as icp_costs, quality_diversity
from stucco.experiments import registration
from stucco.experiments import registration_nopytorch3d
from stucco import serialization
from stucco import util
from stucco.icp import initialization

from datetime import datetime
import logging

from stucco.sdf import draw_pose_distribution, sample_mesh_points

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

model_points_dbname = 'real_model_points_cache.pkl'
experiment_name = 'poke_real_processed'
db_prefix = "poking_real"


class RunData:
    def __init__(self):
        self.pokes = 0
        self.link_to_current_tf_gt = None
        self.best_tsf_guess = None
        self.num_points_to_T_cache = {}
        self.best_distance = None
        self.dist_per_est_obj = []
        self.transforms_per_object = []
        self.rmse_per_object = []
        self.chamfer_err = []
        self.freespace_violations = []
        self.num_freespace_voxels = []
        self.best_segment_idx = None
        self.contact_pts = None
        # for debug rendering of object meshes and keeping track of their object IDs
        self.pose_obj_map = {}
        # for exporting out to file, maps poke # -> data
        self.elapsed = None
        self.to_export = {}
        self.cache = None

    def reset_registration(self):
        self.best_distance = None
        self.dist_per_est_obj = []
        self.transforms_per_object = []
        self.rmse_per_object = []
        self.best_segment_idx = None
        self.elapsed = None

    def reset_run(self):
        self.pokes = 0
        self.best_tsf_guess = None
        self.contact_pts = None
        self.chamfer_err = []
        self.freespace_violations = []
        self.num_freespace_voxels = []
        self.num_points_to_T_cache = {}
        # for debug rendering of object meshes and keeping track of their object IDs
        self.pose_obj_map = {}
        self.to_export = {}


class PokeRunner:
    KEY_COLUMNS = ("method", "name", "seed", "poke", "level", "batch")

    def __init__(self, env: poke_real_nonros.PokeRealNoRosEnv, reg_method, B=30,
                 read_stored=False, ground_truth_initialization=False,
                 init_method=stucco.icp.initialization.InitMethod.RANDOM,
                 register_num_points=500, start_at_num_pts=2, eval_num_points=200,
                 ):

        self.env = env
        self.B = B
        self.dbname = os.path.join(cfg.DATA_DIR, f'{db_prefix}_{env.obj_factory.name}.pkl')
        self.reg_method = reg_method
        self.start_at_num_pts = start_at_num_pts
        self.read_stored = read_stored
        self.ground_truth_initialization = ground_truth_initialization
        self.init_method = init_method

        model_name = self.env.obj_factory.name
        # get a fixed number of model points to evaluate against (this will be independent on points used to register)
        self.model_points_eval, self.model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points,
                                                                                name=model_name,
                                                                                seed=0, dbname=model_points_dbname,
                                                                                device=self.env.device)
        self.dtype = self.model_points_eval.dtype
        self.device = self.env.device

        # get a large number of model points to register to
        self.model_points_register, self.model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                                        name=model_name, seed=0,
                                                                                        dbname=model_points_dbname,
                                                                                        device=self.env.device)

        # need to get these after
        self.model_points_world_frame_eval = None
        self.model_normals_world_frame_eval = None

        self.draw_pose_distribution_separately = True
        # self.method: typing.Optional[TrackingMethod] = None
        self.volumetric_cost: typing.Optional[icp_costs.VolumetricCost] = None
        # intermediary data for bookkeeping during a run
        self.r = RunData()

    def create_volumetric_cost(self):
        # placeholder for now; have to be filled manually
        empty_sdf = voxel.VoxelSet(torch.empty(0), torch.empty(0))
        self.volumetric_cost = icp_costs.VolumetricDirectSDFCost(self.env.free_voxels, empty_sdf, self.env.target_sdf,
                                                                 scale=1,
                                                                 scale_known_freespace=20,
                                                                 obj_factory=self.env.obj_factory,
                                                                 debug=False)

    def register_transforms_with_points(self, seed):
        """Exports best_segment_idx, transforms_per_object, and rmse_per_object"""
        self.r.reset_registration()

        # note that we update our registration regardless if we're in contact or not
        if self.r.contact_pts is None:
            return
        N = len(self.r.contact_pts)
        if N < self.start_at_num_pts or self.r.pokes < self.start_at_num_pts:
            return

        self.volumetric_cost.sdf_voxels = voxel.VoxelSet(self.r.contact_pts,
                                                         torch.zeros(N, dtype=self.dtype, device=self.env.device))

        if self.r.best_tsf_guess is None:
            self.r.best_tsf_guess = initialization.initialize_transform_estimates(self.B, self.env.freespace_ranges,
                                                                                  self.init_method, None,
                                                                                  device=self.env.device,
                                                                                  dtype=self.env.dtype)
        if self.ground_truth_initialization and self.r.link_to_current_tf_gt is not None:
            self.r.best_tsf_guess = self.r.link_to_current_tf_gt.get_matrix().repeat(self.B, 1, 1)

        # avoid giving methods that don't use freespace more training iterations
        if icp.registration_method_uses_only_contact_points(self.reg_method) and N in self.r.num_points_to_T_cache:
            T, distances = self.r.num_points_to_T_cache[N]
        else:
            if self.read_stored or self.reg_method == icp.ICPMethod.CVO:
                T, distances, self.r.elapsed = registration_nopytorch3d.read_offline_output(
                    self.reg_method, self.env.level, seed,
                    self.r.pokes, experiment_name=experiment_name)
                T = T.to(device=self.env.device, dtype=self.dtype)
                distances = distances.to(device=self.env.device, dtype=self.dtype)
            elif self.reg_method == icp.ICPMethod.MEDIAL_CONSTRAINT:
                T, distances, self.r.elapsed = registration.do_medial_constraint_registration(self.r.contact_pts,
                                                                                              self.volumetric_cost.sdf,
                                                                                              self.r.best_tsf_guess,
                                                                                              self.B,
                                                                                              self.env.level,
                                                                                              seed, self.r.pokes,
                                                                                              experiment_name=experiment_name,
                                                                                              obj_factory=self.env.obj_factory)
            else:
                start = timer()
                T, distances = registration.do_registration(self.r.contact_pts, self.model_points_register,
                                                            self.r.best_tsf_guess, self.B,
                                                            self.volumetric_cost,
                                                            self.reg_method)
                end = timer()
                self.r.elapsed = end - start
            self.r.num_points_to_T_cache[N] = T, distances
            logger.info("registration elapsed %fs", self.r.elapsed)

        self.r.transforms_per_object.append(T)
        T = T.inverse()
        score = distances
        best_tsf_index = np.argmin(score.detach().cpu())

        # pick object with lowest variance in its translation estimate
        translations = T[:, :3, 3]
        best_tsf_distances = (translations.var(dim=0).sum()).item()

        # only 1; assume all the given points are for the target object
        self.r.best_segment_idx = 0
        self.r.dist_per_est_obj.append(best_tsf_distances)
        self.r.rmse_per_object.append(distances)
        self.r.best_distance = best_tsf_distances
        self.r.best_tsf_guess = T[best_tsf_index]

    def reinitialize_best_tsf_guess(self):
        self.r.best_tsf_guess = stucco.icp.initialization.reinitialize_transform_estimates(self.B,
                                                                                           self.r.best_tsf_guess)
        return self.r.best_tsf_guess

    def evaluate_registrations(self):
        """Responsible for populating to_export"""
        logger.debug(f"err each obj {np.round(self.r.dist_per_est_obj, 4)}")

        self.reinitialize_best_tsf_guess()

        T = self.r.transforms_per_object[self.r.best_segment_idx]

        # evaluate with chamfer distance
        if self.model_points_world_frame_eval is not None:
            errors_per_batch = evaluate_chamfer_distance(T, self.model_points_world_frame_eval, vis=None,
                                                         obj_factory=self.env.obj_factory, viewing_delay=0)

            link_to_current_tf = tf.Transform3d(matrix=T)
            interior_pts = link_to_current_tf.transform_points(self.volumetric_cost.model_interior_points_orig)
            occupied = self.env.free_voxels[interior_pts]

            self.r.chamfer_err.append(errors_per_batch)
            self.r.num_freespace_voxels.append(self.env.free_voxels.get_known_pos_and_values()[0].shape[0])
            self.r.freespace_violations.append(occupied.sum(dim=-1).detach().cpu())
            logger.info(f"chamfer distance {self.r.pokes}: {torch.mean(errors_per_batch)}")
        else:
            logger.info(f"skipping chamfer error evaluation due to lack of ground truth pose {self.r.pokes}")

    def export_metrics(self, cache, name, seed):
        """Responsible for populating to_export and saving to database"""
        batch = np.arange(self.B)
        rmse = self.r.rmse_per_object[self.r.best_segment_idx]

        d = {"date": datetime.today(), "method": self.reg_method.name, "level": self.env.level.name,
             "name": name,
             "seed": seed, "poke": self.r.pokes,
             "batch": batch,
             "rmse": rmse.cpu().numpy(),
             "elapsed": self.r.elapsed,
             }
        if len(self.r.chamfer_err) > 0:
            _c = np.array(self.r.chamfer_err[-1].cpu().numpy())
            _f = np.array(self.r.freespace_violations[-1])
            _n = self.r.num_freespace_voxels[-1]
            _r = _f / _n
            dd = {
                "chamfer_err": _c, 'freespace_violations': _f,
                'num_freespace_voxels': _n,
                "freespace_violation_percent": _r,
            }
            d.update(dd)
        df = pd.DataFrame(d)

        cache = pd.concat([cache, df])
        cache.to_pickle(self.dbname)
        # additional data to export fo file
        self.r.to_export[self.r.pokes] = {
            'T': self.r.transforms_per_object[self.r.best_segment_idx],
            'rmse': rmse,
            'elapsed': self.r.elapsed,
        }
        return cache

    def run(self, name="", seed=0, ctrl_noise_max=0.005, draw_text=None):
        quality_diversity.previous_solutions = None
        if os.path.exists(self.dbname):
            self.r.cache = pd.read_pickle(self.dbname)
        else:
            self.r.cache = pd.DataFrame()

        env = self.env
        self.create_volumetric_cost()
        # load file based on seed and task; load given points
        traj_file = f"{registration_nopytorch3d.saved_traj_dir_base(self.env.level, experiment_name=experiment_name)}_{seed}.txt"
        if not os.path.exists(traj_file):
            raise RuntimeError(f"Missing processed expected given points per poke file {traj_file}")
        pokes_to_data = {}
        with open(traj_file) as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                poke, num_points = [int(v) for v in lines[i].split()]

                all_pts = torch.tensor(lines[i + 1: i + 1 + num_points], device=self.device, dtype=self.dtype)
                freespace = all_pts[:, -1] == 0

                pokes_to_data[poke] = {
                    'free': all_pts[freespace, :-1],
                    'contact': all_pts[~freespace, :-1]
                }
                i += num_points + 1

        self.r.reset_run()

        rand.seed(seed)

        self.hook_before_first_poke(seed)
        for poke, data in pokes_to_data.items():
            self.r.pokes = poke
            registration_util.poke_index = self.r.pokes

            # simulate poke by setting freespace and contact points
            env.free_voxels[data['free']] = 1
            self.r.contact_pts = data['contact']

            self.hook_after_poke(name, seed)

        self.hook_after_last_poke(seed)

    # hooks for derived classes to add behavior at specific locations
    def hook_before_first_poke(self, seed):
        optimal_pose_file = registration_nopytorch3d.optimal_pose_file(self.env.level, seed,
                                                                       experiment_name=experiment_name)
        if os.path.exists(optimal_pose_file):
            with open(optimal_pose_file, 'r') as f:
                pos = [float(v) for v in f.readline().split()]
                rot = [float(v) for v in f.readline().split()]
                pose = (pos, rot)
                self.r.link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
                    tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
                self.model_points_world_frame_eval = self.r.link_to_current_tf_gt.transform_points(
                    self.model_points_eval)
                self.model_normals_world_frame_eval = self.r.link_to_current_tf_gt.transform_normals(
                    self.model_normals_eval)
        else:
            logger.info(f"No optimal pose found: {optimal_pose_file}")

    def hook_after_poke(self, name, seed):
        self.register_transforms_with_points(seed)
        # has at least one contact segment
        if self.r.best_segment_idx is not None:
            self.evaluate_registrations()
            self.r.cache = self.export_metrics(self.r.cache, name, seed)

    def hook_after_last_poke(self, seed):
        if not self.read_stored:
            serialization.export_registration(
                registration_nopytorch3d.saved_traj_file(self.reg_method, self.env.level, seed,
                                                         experiment_name=experiment_name),
                self.r.to_export)


class PlausibleSetRunner(PokeRunner):
    def plausible_set_filename(self, seed):
        return f"{registration_nopytorch3d.saved_traj_dir_base(self.env.level, experiment_name=experiment_name)}_plausible_set_{seed}.pkl"


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
                                                                       cmax=1000, vis=None,
                                                                       obj_factory=self.env.obj_factory)

    def hook_before_first_poke(self, seed):
        super(GeneratePlausibleSetRunner, self).hook_before_first_poke(seed)
        with rand.SavedRNG():
            rand.seed(0)
            # TODO start with a guess of the pose (by manually setting an approximate pose
            # TODO then do one pass to find the optimal pose and save that (also for use in estimating chamfer error)

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
        # TODO fill up contact points
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

    def hook_after_last_poke(self, seed):
        # export plausible set to file
        filename = self.plausible_set_filename(seed)
        logger.info("saving plausible set to %s", filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.plausible_set, filename)


class EvaluatePlausibleSetRunner(PlausibleSetRunner):
    def __init__(self, *args, plot_meshes=True, sleep_between_plots=0.1, **kwargs):
        super(EvaluatePlausibleSetRunner, self).__init__(*args, **kwargs)
        self.plot_meshes = False
        self.sleep_between_plots = sleep_between_plots
        # always read stored with the plausible set evaluation
        self.read_stored = True
        self.plausible_set = {}
        self.plausibility = None
        self.coverage = None

    def hook_before_first_poke(self, seed):
        super(EvaluatePlausibleSetRunner, self).hook_before_first_poke(seed)
        # they are different trajectories due to control noise in the real world
        filename = self.plausible_set_filename(seed)
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
        errors_per_batch = evaluate_chamfer_distance(Iapprox.view(B * P, 4, 4), self.model_points_eval, None,
                                                     self.env.obj_factory, 0)
        errors_per_batch = errors_per_batch.view(B, P)

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


def main(args):
    level = task_map[args.task]
    registration_method = registration_map[args.registration]
    obj_name = poke_real_nonros.level_to_obj_map[level]
    obj_factory = poke_real_nonros.obj_factory_map(obj_name)

    rand.seed(0)
    env = poke_real_nonros.PokeRealNoRosEnv(environment_level=level, device="cuda")
    logger.info(
        f"--experiment {args.experiment} --registration {args.registration} --task {level.name} --seeds {args.seed}")

    # -- Build object models (sample points from their surface)
    if args.experiment == "build":
        for num_points in (2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500):
            for seed in args.seed:
                registration_nopytorch3d.build_model(env.obj_factory, None, args.task, seed=seed,
                                                     num_points=num_points,
                                                     pause_at_end=False)
    elif args.experiment == "generate-plausible-set":
        env = poke_real_nonros.PokeRealNoRosEnv(environment_level=level, device="cuda")
        runner = GeneratePlausibleSetRunner(env, registration_method,
                                            ground_truth_initialization=False, read_stored=args.read_stored)
        for seed in args.seed:
            runner.run(name=args.name, seed=seed, draw_text=f"seed {seed} plausible set")

    elif args.experiment == "evaluate-plausible-diversity":
        env = poke_real_nonros.PokeRealNoRosEnv(environment_level=level, device="cuda")
        runner = EvaluatePlausibleSetRunner(env, registration_method, read_stored=True)
        for seed in args.seed:
            runner.run(name=args.name, seed=seed, draw_text=f"seed {seed}")

    elif args.experiment == "plot-poke-ce":
        registration_nopytorch3d.plot_poke_chamfer_err(args, level, obj_factory,
                                                       PokeRunner.KEY_COLUMNS, db_prefix=db_prefix)

    elif args.experiment == "plot-poke-pd":
        registration_nopytorch3d.plot_poke_plausible_diversity(args, level, obj_factory,
                                                               PokeRunner.KEY_COLUMNS, quantile=1.0,
                                                               db_prefix=db_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object registration from contact')
    parser.add_argument('experiment',
                        choices=['build',
                                 'plot-poke-ce', 'plot-poke-pd',
                                 'generate-plausible-set', 'evaluate-plausible-diversity',
                                 ],
                        help='which experiment to run')
    registration_map = {m.name.lower().replace('_', '-'): m for m in icp.ICPMethod}
    parser.add_argument('--registration',
                        choices=registration_map.keys(),
                        default='volumetric',
                        help='which registration method to run')
    parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                        default=[0],
                        help='random seed(s) to run')
    parser.add_argument('--no_gui', action='store_true', help='force no GUI')
    # run parameters
    task_map = {level.name.lower(): level for level in poke_real_nonros.Levels}
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
