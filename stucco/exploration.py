import abc
import enum
import math

import os
import gpytorch
import numpy as np
import pybullet as p
import pytorch_kinematics
import torch
import logging
from arm_pytorch_utilities import tensor_utils, rand, linalg
from arm_pytorch_utilities.grad import jacobian
from pytorch_kinematics import transforms as tf

from stucco import cfg
from stucco import icp
from stucco.env.env import Visualizer
from stucco.env.pybullet_env import closest_point_on_surface, ContactInfo
from stucco import util
from multidim_indexing import torch_view

logger = logging.getLogger(__name__)


def weighted_mean_and_variance(x, weights):
    """
    Compute the weighted mean and unbiased weighted sample variance
    :param x: B x N batched scalar quantity
    :param weights: B x N batched unnormalized weight
    :return: weighted mean \mu^*, unbiased weighted sample variance s_w^2
    """
    # normalize the weights such that they sum to 1
    w = weights / weights.sum(dim=-1, keepdim=True)
    v_1 = 1
    v_2 = torch.square(w).sum(dim=-1, keepdim=True)

    mean = (x * w).sum(dim=-1, keepdim=True)
    sq_err = w * torch.square(x - mean)
    sigma = sq_err.sum(dim=-1, keepdim=True)
    # remove bias
    s = sigma / (v_1 - (v_2 / v_1))

    return mean.squeeze(-1), s.squeeze(-1)


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


def sample_dx_on_tangent_plane(n, alpha, num_samples=100):
    v0 = n.clone()
    v0[0] += 1
    v1 = torch.cross(n, v0)
    v2 = torch.cross(n, v1)
    v1 /= v1.norm()
    v2 /= v2.norm()
    angles = torch.linspace(0, math.pi * 2, num_samples)
    dx_samples = (torch.cos(angles).view(-1, 1) * v1 + torch.sin(angles).view(-1, 1) * v2) * alpha
    return dx_samples


class PybulletObjectFactory(abc.ABC):
    def __init__(self, name, scale=2.5, vis_frame_pos=(0, 0, 0), vis_frame_rot=(0, 0, 0, 1), **kwargs):
        self.name = name
        self.scale = scale
        self.vis_frame_pos = vis_frame_pos
        self.vis_frame_rot = vis_frame_rot
        self.other_load_kwargs = kwargs

    @abc.abstractmethod
    def make_collision_obj(self, z, rgba=None):
        """Create collision object of fixed and position along x-y; returns the object ID and bounding box"""

    @abc.abstractmethod
    def get_mesh_resource_filename(self):
        """Return the path to the mesh resource file (.obj, .stl, ...)"""


# TODO score with freespace penetration
class ICPPoseScore:
    """Score an ICP result on domain-specific plausibility; lower score is better.
    This function is designed for objects that manipulator robots will usually interact with"""

    def __init__(self, dim=3, upright_bias=.3, physics_bias=1., reject_proportion_on_score=.3):
        self.dim = dim
        self.upright_bias = upright_bias
        self.physics_bias = physics_bias
        self.reject_proportion_on_score = reject_proportion_on_score

    def __call__(self, T, all_points, icp_rmse):
        # T is matrix taking world frame to link frame
        # score T on rotation's distance away from prior of being upright
        if self.dim == 3:
            rot_axis = tf.matrix_to_axis_angle(T[..., :self.dim, :self.dim])
            rot_axis /= rot_axis.norm(dim=-1, keepdim=True)
            # project onto the z axis (take z component)
            # should be positive; upside down will be penalized and so will sideways
            upright_score = (1 - rot_axis[..., -1])
        else:
            upright_score = 0

        # inherent local minima quality from ICP
        fit_quality_score = icp_rmse if icp_rmse is not None else 0
        # should not float off the ground
        physics_score = all_points[:, :, 2].min(dim=1).values.abs()

        score = fit_quality_score + upright_score * self.upright_bias + physics_score * self.physics_bias

        # reject a portion of the input (assign them inf score) based on their quantile in terms of fit quality
        score_threshold = torch.quantile(score, 1 - self.reject_proportion_on_score)
        score[score > score_threshold] = float('inf')

        return score


class PlotPointType(enum.Enum):
    NONE = 0
    ICP_MODEL_POINTS = 1
    ERROR_AT_MODEL_POINTS = 2
    ERROR_AT_REP_SURFACE = 3
    ICP_ERROR_POINTS = 4


class ShapeExplorationPolicy(abc.ABC):
    def __init__(self, icp_pose_score=ICPPoseScore(), vis: Visualizer = None,
                 plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS,
                 vis_obj_id=None, debug=False, debug_name=""):
        self.model_points = None
        self.model_normals = None
        self.model_points_world_transformed_ground_truth = None
        self.model_normals_world_transformed_ground_truth = None
        self.plot_point_type = plot_point_type
        self.visId = vis_obj_id
        self.icp_pose_score = icp_pose_score

        self.best_tsf_guess = None
        self.device = None
        self.dtype = None

        self.dd = vis
        self.debug = debug
        self.debug_name = debug_name  # name to distinguish this run from others if saving anything for debugging

    def set_debug(self, debug_on):
        self.debug = debug_on

    def start_step(self, xs, df):
        """Bookkeeping at the start of a step"""
        pass

    def end_step(self, xs, df, t):
        """Bookkeeping at the end of a step, with new x and d appended"""
        pass

    @abc.abstractmethod
    def get_next_dx(self, xs, df, t):
        """Get control for deciding which state to visit next"""

    def register_transforms(self, T, best_T):
        pass

    def save_model_points(self, model_points, model_normals, pose):
        self.model_points, self.model_normals = model_points, model_normals
        self.device, self.dtype = self.model_points.device, self.model_points.dtype
        link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        self.model_points_world_transformed_ground_truth = link_to_current_tf.transform_points(self.model_points)
        self.model_normals_world_transformed_ground_truth = link_to_current_tf.transform_normals(self.model_normals)


class ICPEVExplorationPolicy(ShapeExplorationPolicy):
    def __init__(self, obj_frame_sdf: util.ObjectFrameSDF, num_samples_each_action=100,
                 icp_batch=30, alpha=0.01,
                 alpha_evaluate=0.05,
                 upright_bias=0.3,
                 verify_icp_error=False, evaluate_icpev_correlation=False,
                 distance_filter=None,
                 debug_obj_factory: PybulletObjectFactory = None, **kwargs):
        """Test object ID is something we can test the distances to"""
        super(ICPEVExplorationPolicy, self).__init__(**kwargs)

        self.alpha = alpha
        self.alpha_evaluate = alpha_evaluate
        self.N = num_samples_each_action
        self.B = icp_batch

        self.upright_bias = upright_bias
        self.best_tsf_guess = random_upright_transforms(self.B, self.dtype, self.device) if upright_bias > 0 else None
        self.T = None
        self.icp_rmse = None
        # allow external computation of ICP to use inside us, in which case we don't need to redo ICP
        self.unused_cache_transforms = False
        self._cached_link_to_world = None
        self._cached_world_to_link = None

        # need a method to measure the distance to the surface of the object in object frame
        # treat this as a signed distance function
        self.obj_frame_sdf = obj_frame_sdf
        self.distance_filter = distance_filter

        # debug flags
        self.verify_icp_error = verify_icp_error
        self.evaluate_icpev_correlation = evaluate_icpev_correlation

        # for creating visual shapes in debugging
        self.debug_obj_factory = debug_obj_factory
        self.debug_obj_ids = {}
        self.debug_visual_shape_id = None

    def register_transforms(self, T, rmse, best_T=None):
        self.T = T
        self.icp_rmse = rmse
        self.unused_cache_transforms = True
        if best_T is not None:
            self.best_tsf_guess = best_T

    def sample_dx(self, xs, df):
        dx_samples = sample_dx_on_tangent_plane(df[-1], self.alpha_evaluate, num_samples=self.N)
        return dx_samples

    def select_dx(self, dx_samples, icp_error_var):
        most_informative_idx = icp_error_var.argmax()
        dx = dx_samples[most_informative_idx]
        logger.debug(f"chose action {most_informative_idx.item()} / {self.N} {dx} "
                     f"error var {icp_error_var[most_informative_idx].item():.5f} "
                     f"avg error var {icp_error_var.mean().item():.5f}")
        return dx

    def _clear_cached_tf(self):
        self._cached_world_to_link = None
        self._cached_link_to_world = None

    def _link_to_world_tf(self):
        if self._cached_link_to_world is None:
            self._cached_link_to_world = tf.Transform3d(matrix=self.T.inverse())
        return self._cached_link_to_world

    def _world_to_link_tf(self):
        if self._cached_world_to_link is None:
            self._cached_world_to_link = tf.Transform3d(matrix=self.T)
        return self._cached_world_to_link

    def get_next_dx(self, xs, df, t):
        x = xs[-1]
        n = df[-1]
        if t > 5:
            # query for next place to go based on approximated uncertainty using ICP error variance
            this_pts = torch.stack(xs).reshape(-1, 3)
            if self.unused_cache_transforms:
                # consume this transform set
                self.unused_cache_transforms = False
            else:
                # do ICP
                self.T, self.icp_rmse = icp.icp_pytorch3d_sgd(this_pts, self.model_points,
                                                              given_init_pose=self.best_tsf_guess, batch=self.B)
                # self.T, self.icp_rmse = icp.icp_stein(this_pts, self.model_points,
                #                                       given_init_pose=self.T.inverse(),
                #                                       batch=self.B)

            # every time we update T need to clear the transform cache
            self._clear_cached_tf()

            # sample points and see how they evaluate against this ICP result
            dx_samples = self.sample_dx(xs, df)
            new_points_world_frame = x + dx_samples
            # model points are given in link frame
            new_points_object_frame = self._world_to_link_tf().transform_points(new_points_world_frame)

            # compute ICP error for new sampled points
            query_icp_error = self.obj_frame_sdf(new_points_object_frame)
            # don't care about sign of penetration or separation
            query_icp_error = query_icp_error.abs()
            # pass through filters
            if self.distance_filter is not None:
                query_icp_error = self.distance_filter(query_icp_error)

            if self.debug and self.verify_icp_error:
                self._debug_verify_icp_error(new_points_world_frame, query_icp_error)

            # score ICP and save best one to initialize for next step
            model_points_world_frame = self._link_to_world_tf().transform_points(self.model_points)
            # all_normals = link_to_current_tf.transform_normals(self.model_normals)
            score = self.icp_pose_score(self.T, model_points_world_frame, self.icp_rmse)
            # find error variance for each sampled dx
            weight = 1 / score  # score lower is better; want weight where higher is better
            icp_error_mean, icp_error_var = weighted_mean_and_variance(query_icp_error.transpose(0, 1),
                                                                       weight.repeat(query_icp_error.shape[1], 1))
            # icp_error_var = query_icp_error.var(dim=0)
            dx = self.select_dx(dx_samples, icp_error_var)

            best_tsf_index = torch.argmin(score)
            self.best_tsf_guess = self.T[best_tsf_index].inverse()

            if self.debug and self.evaluate_icpev_correlation and t % 10 == 0:
                self._debug_icpev_correlation(this_pts, new_points_world_frame, icp_error_var)
            if self.debug and self.debug_obj_factory is not None:
                self._debug_icp_distribution(new_points_world_frame, icp_error_var)
        else:
            dx = torch.randn(3, dtype=self.dtype, device=self.device)
            dx = dx / dx.norm() * self.alpha
        return dx

    def _debug_icpev_correlation(self, icp_points, new_points_world_frame, query_icp_error):
        # evaluate how good of a proxy is ICPEV for how much our ICP
        # estimated parameter distribution change as a result of knowing contact at a point
        # TODO consider orientation as well, which have to be modelled with a mixture of gaussians
        T = self.T.inverse()
        translations = T[:, :2, 2]
        # treat it as gaussian and compare distance between gaussians

        base_mean = translations.mean(dim=0)
        # base_var = torch.diag(translations.var(dim=0))
        base_var = linalg.cov(translations)
        base_dist = torch.distributions.MultivariateNormal(base_mean, base_var)

        icpevs = []
        actual_diff = []
        # TODO consider evaluate this in batch form?
        total_i = len(query_icp_error)
        for i in range(len(query_icp_error)):
            if i % 500 == 0:
                logger.debug(f"evaluated ICPEV {i}/{total_i}")
            pt = new_points_world_frame[i]
            T, icp_rmse = icp.icp_pytorch3d(torch.cat((icp_points, pt.view(1, -1))), self.model_points,
                                            given_init_pose=self.best_tsf_guess, batch=self.B)
            T = T.inverse()
            translations = T[:, :2, 2]
            m = translations.mean(dim=0)
            # v = torch.diag(translations.var(dim=0))
            v = linalg.cov(translations)
            dist = torch.distributions.MultivariateNormal(m, v)

            diff = torch.distributions.kl_divergence(base_dist, dist).mean()

            icpevs.append(query_icp_error[i].item())
            actual_diff.append(diff.item())

        import matplotlib.pyplot as plt
        save_loc = os.path.join(cfg.DATA_DIR, "icpev_correlation")
        os.makedirs(save_loc, exist_ok=True)

        def do_plot(extra_name="", highlight_point=None):
            def save_and_close_fig(f, name):
                plt.savefig(os.path.join(save_loc, f"{self.debug_name} {name}.png"))
                plt.close(f)

            f = plt.figure()
            fig_name = f"{len(icp_points)} points explored {extra_name}"
            f.suptitle(fig_name)
            ax = plt.gca()
            ax.scatter(icpevs, actual_diff, alpha=0.5)
            if highlight_point is not None:
                ax.scatter(highlight_point[0], highlight_point[1], color='r')
            ax.set_xlabel("ICPEV")
            ax.set_ylabel("KL divergence")
            save_and_close_fig(f, fig_name)

        do_plot()

        # TODO examine problematic points qualitatively
        def map_to_quantile(values):
            values = torch.tensor(values)
            values = values - values.min()
            return values / values.max()

        icpev_quantile = map_to_quantile(icpevs)
        kl_quantile = map_to_quantile(actual_diff)

        higher_icpev_than_expected = icpev_quantile / (kl_quantile + 1e-8)
        lower_icpev_than_expected = kl_quantile / (icpev_quantile + 1e-8)

        interesting_indices_1 = torch.argsort(higher_icpev_than_expected)
        interesting_indices_2 = torch.argsort(lower_icpev_than_expected)
        for interesting_indices in [interesting_indices_1, interesting_indices_2]:
            for i in range(5):
                x = icpevs[interesting_indices[i]]
                y = actual_diff[interesting_indices[i]]
                # TODO actually go to this state again so we can see it in the visualizer
                do_plot(f"{(x, y)}", (x, y))

    def _debug_verify_icp_error(self, new_points_world_frame, query_icp_error):
        # compare our method of transforming all points to link frame with transforming all objects
        query_icp_error_ground_truth = torch.zeros(self.B, self.N)
        m = self._link_to_world_tf().get_matrix()
        for b in range(self.B):
            pos, rot = util.matrix_to_pos_rot(m[b])
            p.resetBasePositionAndOrientation(self.visId, pos, rot)

            # transform our visual object to the pose
            for i in range(self.N):
                closest = closest_point_on_surface(self.visId, new_points_world_frame[i])
                query_icp_error_ground_truth[b, i] = closest[ContactInfo.DISTANCE]
                if self.plot_point_type == PlotPointType.ICP_ERROR_POINTS:
                    self.dd.draw_point("test_point", new_points_world_frame[i], color=(1, 0, 0), length=0.005)
                    self.dd.draw_point("test_point_surf", closest[ContactInfo.POS_A], color=(0, 1, 0),
                                       length=0.005,
                                       label=f'{closest[ContactInfo.DISTANCE]:.5f}')
        query_icp_error_ground_truth = query_icp_error_ground_truth.abs()
        assert (query_icp_error_ground_truth - query_icp_error).sum() < 1e-4

    def _debug_icp_distribution(self, new_points_world_frame, icp_error_var):
        # create visual objects
        # transform to the ICP'd pose
        # plot distribution of poses and the candidate points in terms of their information metric
        m = self._link_to_world_tf().get_matrix()
        for b in range(self.B):
            pos, rot = util.matrix_to_pos_rot(m[b])
            object_id = self.debug_obj_ids.get(b, None)
            object_id = self.dd.draw_mesh("icp_distribution", self.debug_obj_factory.get_mesh_resource_filename(),
                                          (pos, rot), scale=self.debug_obj_factory.scale, object_id=object_id,
                                          rgba=(0, 0.8, 0.2, 0.2),
                                          vis_frame_pos=self.debug_obj_factory.vis_frame_pos,
                                          vis_frame_rot=self.debug_obj_factory.vis_frame_rot)
            self.debug_obj_ids[b] = object_id

        if new_points_world_frame is not None:
            # plot the candidate points in world frame with color map indicating their ICPEV metric
            import matplotlib.colors, matplotlib.cm
            error_norm = matplotlib.colors.Normalize(vmin=0, vmax=icp_error_var.max())
            color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
            rgb = color_map.to_rgba(icp_error_var.reshape(-1))
            rgb = rgb[:, :-1]

            for i in range(new_points_world_frame.shape[0]):
                self.dd.draw_point(f"candidate_icpev_pt.{i}", new_points_world_frame[i], color=rgb[i], length=0.005)

            most_informative_idx = icp_error_var.argmax()
            self.dd.draw_point(f"most info pt", new_points_world_frame[most_informative_idx], color=(1, 0, 0),
                               length=0.01,
                               label=f"icpev: {icp_error_var[most_informative_idx].item():.5f}")
            print()


class RandomSlidePolicy(ICPEVExplorationPolicy):
    def select_dx(self, dx_samples, icp_error_var):
        idx = torch.randint(self.N, (1,)).item()
        dx = dx_samples[idx]
        logger.debug(f"chose action {idx} / {self.N} {dx} "
                     f"error var {icp_error_var[idx].item():.5f} "
                     f"avg error var {icp_error_var.mean().item():.5f}")
        return dx


class RandomSamplePolicy(RandomSlidePolicy):
    def sample_dx(self, xs, df):
        # random points in space
        dx_samples = torch.randn((self.N, 3)) * self.alpha_evaluate * 5
        return dx_samples


class ICPEVSampleRandomPointsPolicy(ICPEVExplorationPolicy):
    """Choose the point with the best ICPEV metric, but select candidate points randomly in space around current"""

    def sample_dx(self, xs, df):
        # random points in space
        dx_samples = torch.randn((self.N, 3)) * self.alpha_evaluate * 5
        return dx_samples


class ICPEVSampleModelPointsPolicy(ICPEVExplorationPolicy):
    """ICPEV exploration where we sample model points instead of fixed sliding around self"""

    def __init__(self, *args, capped=False, **kwargs):
        super(ICPEVSampleModelPointsPolicy, self).__init__(*args, **kwargs)
        self.capped = capped

    def _sample_model_points(self):
        # sample which ICP to use for each of the points
        all_candidates = self._link_to_world_tf().transform_points(self.model_points).view(-1, 3)
        # uniformly random choose a transformed model point
        idx = torch.randperm(all_candidates.shape[0], device=self.device)[:self.N]
        pts = all_candidates[idx]
        return pts

    def sample_dx(self, xs, df):
        x = xs[-1]
        pts = self._sample_model_points()
        dx_samples = pts[:self.N] - x
        if self.capped:
            # cap step size
            over_step = dx_samples.norm() > self.alpha_evaluate
            dx_samples[over_step] = dx_samples[over_step] / dx_samples[over_step].norm() * self.alpha_evaluate
        return dx_samples


class ICPEVVoxelizedPolicy(ICPEVExplorationPolicy):
    """ICPEV exploration where we sample model points instead of fixed sliding around self"""

    def __init__(self, *args, resolution=0.05, range_per_dim=0.25, **kwargs):
        super(ICPEVVoxelizedPolicy, self).__init__(*args, **kwargs)
        self.resolution = resolution
        if isinstance(range_per_dim, (float, int)):
            range_per_dim = tuple(range_per_dim for _ in range(3))
        self.range_per_dim = range_per_dim

    def sample_dx(self, xs, df):
        # x = xs[-1]
        # sample voxels around current x
        dx_samples = torch.cartesian_prod(
            *[torch.arange(-half_span, half_span + self.resolution, step=self.resolution) for half_span in
              self.range_per_dim])
        return dx_samples


class GPVarianceExploration(ShapeExplorationPolicy):
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

    def start_step(self, xs, df):
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
                dx_samples = sample_dx_on_tangent_plane(n, self.alpha)
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

    def end_step(self, xs, df, t):
        # augment data with ICP shape pseudo-observations
        if t > 0 and t % self.icp_period == 0:
            logger.debug('conditioning exploration on ICP results')
            # perform ICP and visualize the transformed points
            this_pts = torch.stack(xs).reshape(-1, 3)
            T, distances, _ = icp.icp_3(this_pts, self.model_points,
                                        given_init_pose=self.best_tsf_guess, batch=30)
            link_to_current_tf = tf.Transform3d(matrix=T.inverse())
            all_points = link_to_current_tf.transform_points(self.model_points)

            score = self.icp_pose_score(T, all_points, distances).numpy()
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

    def get_next_dx(self, xs, df, t):
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


def random_upright_transforms(B, dtype, device, translation=None):
    # initialize guesses with a prior; since we're trying to retrieve an object, it's reasonable to have the prior
    # that the object only varies in yaw (and is upright)
    axis_angle = torch.zeros((B, 3), dtype=dtype, device=device)
    axis_angle[:, -1] = torch.rand(B, dtype=dtype, device=device) * 2 * np.pi
    R = tf.axis_angle_to_matrix(axis_angle)
    init_pose = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)
    init_pose[:, :3, :3] = R
    if translation is not None:
        init_pose[:, :3, 3] = translation
    return init_pose


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


class PyBulletNaiveSDF(util.ObjectFrameSDF):
    def __init__(self, test_obj_id, plot_point_type=PlotPointType.NONE, vis=None):
        self.test_obj_id = test_obj_id
        self.plot_point_type = plot_point_type
        self.vis = vis

    def __call__(self, points_in_object_frame):
        if len(points_in_object_frame.shape) == 2:
            points_in_object_frame = points_in_object_frame.unsqueeze(0)
        B, N, d = points_in_object_frame.shape
        dtype = points_in_object_frame.dtype
        device = points_in_object_frame.device
        # compute SDF value for new sampled points
        sdf = torch.zeros(B, N, dtype=dtype, device=device)
        sdf_grad = [[None] * N for _ in range(B)]
        # points are transformed to link frame, thus it needs to compare against the object in link frame
        # objId is not in link frame and shouldn't be moved
        for b in range(B):
            for i in range(N):
                closest = closest_point_on_surface(self.test_obj_id, points_in_object_frame[b, i])
                sdf[b, i] = closest[ContactInfo.DISTANCE]
                sdf_grad[b][i] = closest[ContactInfo.NORMAL_DIR_B]

                if self.vis is not None and self.plot_point_type == PlotPointType.ICP_ERROR_POINTS:
                    self.vis.draw_point("test_point", points_in_object_frame[b, i], color=(1, 0, 0), length=0.005)
                    self.vis.draw_2d_line(f"test_normal", points_in_object_frame[b, i],
                                          [-v for v in closest[ContactInfo.NORMAL_DIR_B]], color=(0, 0, 0),
                                          size=2., scale=0.03)
                    self.vis.draw_point("test_point_surf", closest[ContactInfo.POS_A], color=(0, 1, 0),
                                        length=0.005,
                                        label=f'{closest[ContactInfo.DISTANCE]:.5f}')
        # want the gradient from low to high value (pointing out of surface), so need negative
        sdf_grad = -torch.tensor(sdf_grad, dtype=dtype, device=device)
        return sdf, sdf_grad

    def get_voxel_view(self, voxels: util.VoxelGrid = None) -> torch_view.TorchMultidimView:
        if voxels is None:
            voxels = util.VoxelGrid(0.01, [[-1, 1], [-1, 1], [-0.6, 1]])

        pts = voxels.get_voxel_center_points()
        sdf_val, sdf_grad = self.__call__(pts.unsqueeze(0))
        cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in voxels.coords])

        return torch_view.TorchMultidimView(cached_underlying_sdf, voxels.range_per_dim, invalid_value=self.__call__)


class CachedSDF(util.ObjectFrameSDF):
    def __init__(self, object_name, resolution, range_per_dim, gt_sdf, device="cpu", clean_cache=False,
                 debug_check_sdf=False):
        fullname = os.path.join(cfg.DATA_DIR, f'sdf_cache.pkl')
        self.device = device
        # cache for signed distance field to object
        self.voxels = None
        # voxel grid can't handle vector values yet
        self.voxels_grad = None

        cached_underlying_sdf = None
        cached_underlying_sdf_grad = None

        self.gt_sdf = gt_sdf
        self.resolution = resolution

        range_per_dim = util.get_divisible_range_by_resolution(resolution, range_per_dim)
        self.ranges = range_per_dim

        self.name = f"{object_name} {resolution} {tuple(range_per_dim)}"
        self.debug_check_sdf = debug_check_sdf

        if os.path.exists(fullname):
            data = torch.load(fullname) or {}
            try:
                cached_underlying_sdf, cached_underlying_sdf_grad = data[self.name]
                logger.info("cached sdf for %s loaded from %s", self.name, fullname)
            except (ValueError, KeyError):
                logger.info("cached sdf invalid %s from %s, recreating", self.name, fullname)
        else:
            data = {}

        # if we didn't load anything, then we need to create the cache and save to it
        if cached_underlying_sdf is None or clean_cache:
            if gt_sdf is None:
                raise RuntimeError("Cached SDF did not find the cache and requires an initialize queryable SDF")

            coords, pts = util.get_coordinates_and_points_in_grid(resolution, range_per_dim)
            sdf_val, sdf_grad = gt_sdf(pts.unsqueeze(0))
            cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in coords])
            cached_underlying_sdf_grad = sdf_grad.squeeze(0)
            # cached_underlying_sdf_grad = sdf_grad.reshape(cached_underlying_sdf.shape + (3,))
            # confirm the values work
            if self.debug_check_sdf:
                debug_view = torch_view.TorchMultidimView(cached_underlying_sdf, range_per_dim,
                                                          invalid_value=self._fallback_sdf_value_func)
                query = debug_view[pts]
                assert torch.allclose(sdf_val, query)

            data[self.name] = cached_underlying_sdf, cached_underlying_sdf_grad

            torch.save(data, fullname)
            logger.info("caching sdf for %s to %s", self.name, fullname)

        cached_underlying_sdf = cached_underlying_sdf.to(device=device)
        cached_underlying_sdf_grad = cached_underlying_sdf_grad.to(device=device)
        self.voxels = torch_view.TorchMultidimView(cached_underlying_sdf, range_per_dim,
                                                   invalid_value=self._fallback_sdf_value_func)
        # TODO handle vector valued views
        self.voxels_grad = cached_underlying_sdf_grad.squeeze()

    def _fallback_sdf_value_func(self, *args, **kwargs):
        sdf_val, _ = self.gt_sdf(*args, **kwargs)
        sdf_val = sdf_val.to(device=self.device)
        return sdf_val

    def __call__(self, points_in_object_frame):
        res = self.voxels[points_in_object_frame]

        keys = self.voxels.ensure_index_key(points_in_object_frame)
        keys_ravelled = self.voxels.ravel_multi_index(keys, self.voxels.shape)
        grad = self.voxels_grad[keys_ravelled]

        if self.debug_check_sdf:
            res_gt = self._fallback_sdf_value_func(points_in_object_frame)
            # the ones that are valid should be close enough to the ground truth
            diff = torch.abs(res - res_gt)
            close_enough = diff < self.resolution
            within_bounds = self.voxels.get_valid_values(points_in_object_frame)
            assert torch.all(close_enough[within_bounds])
        return res, grad

    def get_voxel_view(self, voxels: util.VoxelGrid = None) -> torch_view.TorchMultidimView:
        if voxels is None:
            return self.voxels

        pts = voxels.get_voxel_center_points()
        sdf_val, sdf_grad = self.gt_sdf(pts.unsqueeze(0))
        sdf_val = sdf_val.to(device=self.device)
        cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in voxels.coords])

        return torch_view.TorchMultidimView(cached_underlying_sdf, voxels.range_per_dim,
                                            invalid_value=self._fallback_sdf_value_func)
