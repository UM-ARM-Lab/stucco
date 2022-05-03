import abc
import enum
import math

import gpytorch
import numpy as np
import pybullet as p
import pytorch_kinematics
import torch
from arm_pytorch_utilities import tensor_utils, rand
from arm_pytorch_utilities.grad import jacobian
from pytorch_kinematics import transforms as tf

from stucco import icp
from stucco.env.pybullet_env import closest_point_on_surface, ContactInfo


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


class ShapeExplorationPolicy(abc.ABC):
    def __init__(self, debug_drawer=None, plot_point_type=PlotPointType.ERROR_AT_MODEL_POINTS, vis_obj_id=None):
        self.model_points = None
        self.model_normals = None
        self.model_points_world_transformed_ground_truth = None
        self.model_normals_world_transformed_ground_truth = None
        self.plot_point_type = plot_point_type
        self.visId = vis_obj_id

        self.best_tsf_guess = None
        self.device = None
        self.dtype = None

        self.dd = debug_drawer

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
    def __init__(self, test_obj_id, num_samples_each_action=100, icp_batch=30, alpha=0.01, alpha_evaluate=0.05,
                 verify_icp_error=False, **kwargs):
        """Test object ID is something we can test the distances to"""
        super(ICPEVExplorationPolicy, self).__init__(**kwargs)

        self.alpha = alpha
        self.alpha_evaluate = alpha_evaluate
        self.verify_icp_error = verify_icp_error
        self.N = num_samples_each_action
        self.B = icp_batch

        self.best_tsf_guess = None
        self.T = None
        # allow external computation of ICP to use inside us, in which case we don't need to redo ICP
        self.unused_cache_transforms = False

        self.testObjId = test_obj_id

    def register_transforms(self, T, best_T):
        self.T = T
        self.best_tsf_guess = best_T
        self.unused_cache_transforms = True

    def sample_dx(self, xs, df):
        dx_samples = sample_dx_on_tange_plane(df[-1], self.alpha_evaluate, num_samples=self.N)
        return dx_samples

    def get_next_dx(self, xs, df, t):
        x = xs[-1]
        n = df[-1]
        if t > 5:
            # query for next place to go based on approximated uncertainty using ICP error variance
            with rand.SavedRNG():
                if self.unused_cache_transforms:
                    # consume this transform set
                    self.unused_cache_transforms = False
                else:
                    # do ICP
                    this_pts = torch.stack(xs).reshape(-1, 3)
                    self.T, distances, _ = icp.icp_3(this_pts, self.model_points, given_init_pose=self.best_tsf_guess,
                                                     batch=self.B)

                # sample points and see how they evaluate against this ICP result
                dx_samples = self.sample_dx(xs, df)
                new_x_samples = x + dx_samples
                # model points are given in link frame; new_x_sample points are in world frame
                point_tf_to_link = tf.Transform3d(matrix=self.T)
                all_points = point_tf_to_link.transform_points(new_x_samples)

                # compute ICP error for new sampled points
                query_icp_error = torch.zeros(self.B, self.N)
                # points are transformed to link frame, thus it needs to compare against the object in link frame
                # objId is not in link frame and shouldn't be moved
                for b in range(self.B):
                    for i in range(self.N):
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
                    query_icp_error_ground_truth = torch.zeros(self.B, self.N)
                    link_to_world = tf.Transform3d(matrix=self.T.inverse())
                    m = link_to_world.get_matrix()
                    for b in range(self.B):
                        pos = m[b, :3, 3]
                        rot = pytorch_kinematics.matrix_to_quaternion(m[b, :3, :3])
                        rot = tf.wxyz_to_xyzw(rot)
                        p.resetBasePositionAndOrientation(self.visId, pos, rot)

                        # transform our visual object to the pose
                        for i in range(self.N):
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
                print(f"chose action {most_informative_idx.item()} / {self.N} {dx} "
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


class ICPEVSampleModelPointsPolicy(ICPEVExplorationPolicy):
    """ICPEV exploration where we sample model points instead of fixed sliding around self"""

    def __init__(self, capped=False, **kwargs):
        super(ICPEVSampleModelPointsPolicy, self).__init__(**kwargs)
        self.capped = capped

    def sample_dx(self, xs, df):
        x = xs[-1]
        pts = torch.zeros((self.N, 3))
        # sample which ICP to use for each of the points
        which_to_use = torch.randint(low=0, high=self.B - 1, size=(self.N,), device=self.device)
        for i in range(self.B):
            selection = which_to_use == i
            if torch.any(selection):
                link_to_current_tf = tf.Transform3d(matrix=self.T[i].inverse())
                pts[selection] = link_to_current_tf.transform_points(self.model_points[selection])

        dx_samples = pts - x
        if self.capped:
            # cap step size
            over_step = dx_samples.norm() > self.alpha_evaluate
            dx_samples[over_step] = dx_samples[over_step] / dx_samples[over_step].norm() * self.alpha_evaluate
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

    def end_step(self, xs, df, t):
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
