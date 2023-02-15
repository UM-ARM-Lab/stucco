import abc
import copy
import os
import typing
from datetime import datetime

import gpytorch
import matplotlib
import numpy as np
import pybullet as p
import pybullet_data
import pymeshlab
import torch

import pytorch_volumetric.sdf
import chsel.initialization
from arm_pytorch_utilities import tensor_utils, rand
from matplotlib import pyplot as plt
from pytorch_kinematics import transforms as tf
from torchmcubes import marching_cubes

from chsel_experiments import registration
from base_experiments import cfg
from chsel_experiments.env import obj_factory_map
from base_experiments.env.pybullet_env import closest_point_on_surface, ContactInfo, surface_normal_at_point
from base_experiments.env.env import draw_AABB
from base_experiments.env.real_env import CombinedVisualizer
from pytorch_volumetric.chamfer import batch_chamfer_dist
from stucco.exploration import PlotPointType, ShapeExplorationPolicy, ICPEVExplorationPolicy, GPVarianceExploration
from pytorch_volumetric.sdf import sample_mesh_points


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

    dbpath = os.path.join(cfg.DATA_DIR, "model_points_cache.pkl"),
    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_mesh_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                  dbpath=dbpath,
                                                                  device=exp.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_mesh_points(num_points=register_num_points,
                                                                          dbpath=dbpath,
                                                                          name=model_name, seed=0, device=exp.device)

    # # test ICP using fixed set of points
    # can incrementally increase the number of model points used to evaluate how efficient the ICP is
    errors = []
    B = 30

    best_tsf_guess = None if upright_bias == 0 else chsel.initialization.random_upright_transforms(B, dtype,
                                                                                                   device)

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
    T, distances = registration.icp_pytorch3d(model_points_world_frame, model_points_register, given_init_pose=best_tsf_guess,
                                              batch=B)

    errors_per_batch = batch_chamfer_dist(T, model_points_world_frame_eval, exp.obj_factory, viewing_delay, vis=vis)


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
                                                            dbpath=os.path.join(cfg.DATA_DIR, "model_points_cache.pkl"),
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
        obj_frame_sdf = pytorch_volumetric.sdf.MeshSDF(self.obj_factory)
        range_per_dim = copy.copy(self.obj_factory.ranges)
        if clean_cache:
            # draw the bounding box of the object frame SDF
            draw_AABB(self.dd, range_per_dim)

        self.sdf = pytorch_volumetric.sdf.CachedSDF(self.obj_factory.name, sdf_resolution, range_per_dim,
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
