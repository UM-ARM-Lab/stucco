import matplotlib.colors, matplotlib.cm
import torch
from torch.nn import MSELoss
from pytorch3d.ops.knn import knn_gather, _KNN
from stucco.icp_sgd import _apply_similarity_transform
from typing import Any

from stucco import util


class ICPPoseCost:
    def __call__(self, knn_res: _KNN, R, T, s):
        """Cost for given pose guesses of ICP

        :param R: B x 3 x 3
        :param T: B x 3
        :param s: B
        :return: scalar that is the mean of the costs
        """
        return 0

    def visualize(self, R, T, s):
        pass


class ComposeCost(ICPPoseCost):
    def __init__(self, *args):
        self.costs = args

    def __call__(self, *args, **kwargs):
        return sum(cost(*args, **kwargs) for cost in self.costs)


class SurfaceNormalCost(ICPPoseCost):
    """Cost of matching the surface normals at corresponding points"""

    def __init__(self, Xnorm, Ynorm, scale=1.):
        self.Xnorm = Xnorm
        self.Ynorm = Ynorm
        self.loss = MSELoss()
        self.scale = scale

    def __call__(self, knn_res: _KNN, R, T, s):
        corresponding_ynorm = knn_gather(self.Ynorm, knn_res.idx).squeeze(-2)
        transformed_norms = _apply_similarity_transform(self.Xnorm, R, s=s)
        return self.loss(corresponding_ynorm, transformed_norms) * self.scale


class KnownFreeSpaceCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, world_frame_interior_points: torch.tensor, world_frame_interior_gradients: torch.tensor,
                interior_point_weights: torch.tensor, world_frame_voxels: util.VoxelGrid) -> torch.tensor:
        # interior points should not be occupied
        occupied = world_frame_voxels[world_frame_interior_points]
        # voxels should be 1 where it is known free space, otherwise 0
        loss = occupied * interior_point_weights
        ctx.occupied = occupied
        ctx.save_for_backward(world_frame_interior_gradients, interior_point_weights)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # need to output the gradient of the loss w.r.t. all the inputs of forward
        dl_dpts = None
        dl_dgrad = None
        dl_dweights = None
        dl_dvoxels = None
        if ctx.needs_input_grad[0]:
            world_frame_interior_gradients, interior_point_weights = ctx.saved_tensors
            # SDF grads point away from the surface; in this case we want to move the surface away from the occupied
            # free space point, so the surface needs to go in the opposite direction
            grads = ctx.occupied[:, :, None] * world_frame_interior_gradients * interior_point_weights[None, :, None]
            # TODO consider averaging the gradients out across all points?
            dl_dpts = grad_outputs[:, :, None] * grads

        # gradients for the other inputs not implemented
        return dl_dpts, dl_dgrad, dl_dweights, dl_dvoxels


class KnownSDFDistanceCost:
    @staticmethod
    def apply(world_frame_all_points: torch.tensor, all_point_weights: torch.tensor,
              known_voxel_centers: torch.tensor, known_voxel_values: torch.tensor, epsilon=0.005) -> torch.tensor:
        # difference between current SDF value at each point and the desired one
        sdf_diff = torch.cdist(all_point_weights.view(-1, 1), known_voxel_values.view(-1, 1))

        # vector from each known voxel's center with known value to each point
        known_voxel_to_pt = world_frame_all_points.unsqueeze(-2) - known_voxel_centers
        known_voxel_to_pt_dist = known_voxel_to_pt.norm(dim=-1)

        # loss for each point, corresponding to each known voxel center
        # only consider points with sdf_diff less than epsilon between desired and model (take the level set)
        mask = sdf_diff < epsilon
        # ensure we have at least one element included in the mask
        while torch.any(mask.sum(dim=0) == 0):
            epsilon *= 2
            mask = sdf_diff < epsilon
        # low distance should have low difference
        # remove those not masked from contention
        known_voxel_to_pt_dist[:, ~mask] = 10000

        loss = known_voxel_to_pt_dist
        # each point may not satisfy two targets simulatenously, so we just care about the best one
        loss = loss.min(dim=1).values.mean(dim=-1)

        # ctx.sdf_diff = sdf_diff
        # ctx.known_voxel_to_pt = known_voxel_to_pt
        # ctx.save_for_backward(world_frame_interior_gradients, interior_point_weights)
        return loss

    # @staticmethod
    # def backward(ctx: Any, grad_outputs: Any) -> Any:
    #     # need to output the gradient of the loss w.r.t. all the inputs of forward
    #     dl_dpts = None
    #     dl_dgrad = None
    #     dl_dweights = None
    #     dl_dvoxels = None
    #     if ctx.needs_input_grad[0]:
    #         world_frame_interior_gradients, interior_point_weights = ctx.saved_tensors
    #         # SDF grads point away from the surface; in this case we want to move the surface away from the occupied
    #         # free space point, so the surface needs to go in the opposite direction
    #         grads = ctx.occupied[:, :, None] * world_frame_interior_gradients * interior_point_weights[None, :, None]
    #         # TODO consider averaging the gradients out across all points?
    #         dl_dpts = grad_outputs[:, :, None] * grads
    #
    #     # gradients for the other inputs not implemented
    #     return dl_dpts, dl_dgrad, dl_dweights, dl_dvoxels


class VolumetricCost(ICPPoseCost):
    """Cost of transformed model pose intersecting with known freespace voxels"""

    def __init__(self, free_voxels: util.Voxels, sdf_voxels: util.Voxels, obj_sdf: util.ObjectFrameSDF, scale=1,
                 vis=None, debug=False, scale_known_freespace=1., scale_known_sdf=1.):
        """
        :param free_voxels: representation of freespace
        :param sdf_voxels: voxels for which we know the exact SDF values for
        :param model_interior_points: points on the inside of the model (not on the surface)
        :param obj_sdf: signed distance function of the target object in object frame
        :param scale:
        """

        self.free_voxels = free_voxels
        self.sdf_voxels = sdf_voxels

        self.scale = scale
        self.scale_known_freespace = scale_known_freespace
        self.scale_known_sdf = scale_known_sdf

        # SDF gives us a volumetric representation of the target object
        self.sdf = obj_sdf

        # ---- for +, known free space points, we just have to care about interior points of the object
        # to facilitate comparison between volumes that are rotated, we sample points at the center of the object voxels
        interior_threshold = -0.01
        interior_filter = lambda voxel_sdf: voxel_sdf < interior_threshold
        self.model_interior_points_orig = self.sdf.get_filtered_points(interior_filter)
        self.model_interior_weights, self.model_interior_normals_orig = self.sdf(self.model_interior_points_orig)
        self.model_interior_weights *= -1

        self.model_all_points = self.sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < 0.01)
        self.model_all_weights, self.model_all_normals = self.sdf(self.model_all_points)

        # batch
        self.B = None

        # intermediate products for visualization purposes
        self._pts = None
        self._grad = None

        self._pts_all = None
        self.debug = debug

        self.vis = vis

    def __call__(self, knn_res: _KNN, R, T, s):
        # assign batch and reuse for later for efficiency
        if self.B is None or self.B != R.shape[0]:
            self.B = R.shape[0]
            self.model_interior_points = self.model_interior_points_orig.repeat(self.B, 1, 1)
            self.model_interior_normals = self.model_interior_normals_orig.repeat(self.B, 1, 1)

        # voxels are in world frame
        # need points transformed into world frame
        # transform the points via the given similarity transformation parameters, then evaluate their occupancy
        # should transform the interior points from link frame to world frame
        self._transform_model_to_world_frame(R, T, s)
        # self.visualize(R, T, s)

        loss = torch.zeros(self.B, device=self._pts.device, dtype=self._pts.dtype)

        if self.scale_known_freespace != 0:
            known_free_space_loss = KnownFreeSpaceCost.apply(self._pts, self._grad, self.model_interior_weights,
                                                             self.free_voxels)
            loss += known_free_space_loss * self.scale_known_freespace
        if self.scale_known_sdf != 0:
            known_sdf_voxel_centers, known_sdf_voxel_values = self.sdf_voxels.get_known_pos_and_values()
            known_sdf_loss = KnownSDFDistanceCost.apply(self._pts_all, self.model_all_weights,
                                                        known_sdf_voxel_centers, known_sdf_voxel_values)
            loss += known_sdf_loss * self.scale_known_sdf

        return loss * self.scale

    def _transform_model_to_world_frame(self, R, T, s):
        Rt = R.transpose(-1, -2)
        tt = (-Rt @ T.reshape(-1, 3, 1)).squeeze(-1)
        self._pts = _apply_similarity_transform(self.model_interior_points, Rt, tt, s)
        self._pts_all = _apply_similarity_transform(self.model_all_points, Rt, tt, s)
        if self.debug and self._pts.requires_grad:
            self._pts.retain_grad()
            self._pts_all.retain_grad()
        self._grad = _apply_similarity_transform(self.model_interior_normals, Rt)

    def visualize(self, R, T, s):
        if not self.debug:
            return
        if self.vis is not None:
            if self._pts is None:
                self._transform_model_to_world_frame(R, T, s)
            with torch.no_grad():
                # occupied = self.voxels.voxels.raw_data > 0
                # indices = occupied.nonzero()
                # coord = self.voxels.voxels.ensure_value_key(indices)
                # for i, xyz in enumerate(coord):
                #     self.vis.draw_point(f"free.{i}", xyz, color=(1, 0, 0), scale=5)
                point_grads = self._pts_all.grad
                have_gradients = point_grads.sum(dim=-1).sum(dim=-1) != 0

                batch_with_gradients = have_gradients.nonzero()
                for b in batch_with_gradients:
                    b = b.item()
                    # only visualize it for one sample at a time
                    if b != 0:
                        continue

                    b = 0
                    i = 0
                    # for i, pt in enumerate(self._pts_all[b]):
                    #     self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
                    #     # self.vis.draw_2d_line(f"min.{i}", pt, self._grad[b][i], color=(1, 0, 0), size=2.,
                    #     #                       scale=0.02)
                    #     # visualize the computed gradient on the point
                    #     # gradient descend goes along negative gradient so best to show the direction of movement
                    #     self.vis.draw_2d_line(f"mingrad.{i}", pt, -self._pts_all.grad[b, i], color=(0, 1, 0), size=5.,
                    #                           scale=10)
                    # self.vis.clear_visualization_after("mipt", i + 1)
                    # self.vis.clear_visualization_after("min", i + 1)
                    # self.vis.clear_visualization_after("mingrad", i + 1)

                    # visualize the known SDF loss directly
                    known_voxel_centers, known_voxel_values = self.sdf_voxels.get_known_pos_and_values()
                    world_frame_all_points = self._pts_all
                    all_point_weights = self.model_all_weights
                    # difference between current SDF value at each point and the desired one
                    sdf_diff = torch.cdist(all_point_weights.view(-1, 1), known_voxel_values.view(-1, 1))
                    # vector from each known voxel's center with known value to each point
                    known_voxel_to_pt = world_frame_all_points.unsqueeze(-2) - known_voxel_centers
                    known_voxel_to_pt_dist = known_voxel_to_pt.norm(dim=-1)

                    epsilon = 0.005
                    # loss for each point, corresponding to each known voxel center
                    # only consider points with sdf_diff less than epsilon between desired and model (take the level set)
                    mask = sdf_diff < epsilon
                    # ensure we have at least one element included in the mask
                    while torch.any(mask.sum(dim=0) == 0):
                        epsilon *= 2
                        mask = sdf_diff < epsilon
                    # low distance should have low difference
                    # remove those not masked from contention
                    known_voxel_to_pt_dist[:, ~mask] = 10000

                    loss = known_voxel_to_pt_dist

                    # closest of each known SDF; try to satisfy each target as best as possible
                    min_values, min_idx = loss.min(dim=1)

                    # just visualize the first one
                    min_values = min_values[b]
                    min_idx = min_idx[b]
                    for k in range(len(min_values)):
                        self.vis.draw_point(f"to_match", known_voxel_centers[k], color=(1, 0, 0), length=0.003,
                                            scale=10)
                        self.vis.draw_point(f"closest", world_frame_all_points[b, min_idx[k]], color=(0, 1, 0),
                                            length=0.003, scale=10)
                        self.vis.draw_2d_line(f"closest_grad",
                                              self._pts_all[b, min_idx[k]],
                                              -self._pts_all.grad[b, min_idx[k]],
                                              color=(0, 1, 0), size=2., scale=1)

                        # draw all losses corresponding to this to match voxel and color code their values
                        each_loss = loss[0, :, k].detach().cpu()
                        # draw all the masked out ones as 0
                        each_loss[each_loss > 1000] = each_loss.max()

                        error_norm = matplotlib.colors.Normalize(vmin=0, vmax=each_loss.max())
                        color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
                        rgb = color_map.to_rgba(each_loss.reshape(-1))
                        rgb = rgb[:, :-1]

                        for i in range(world_frame_all_points[0].shape[0]):
                            self.vis.draw_point(f"each_loss_pt.{i}", world_frame_all_points[0, i], color=rgb[i],
                                                length=0.003)

                        print(min_values[k])


class ICPPoseCostMatrixInputWrapper:
    def __init__(self, cost: ICPPoseCost, action_cost_scale=1.0):
        self.cost = cost
        self.action_cost_scale = action_cost_scale

    def __call__(self, H, dH=None):
        N = H.shape[0]
        H = H.view(N, 4, 4)
        R = H[:, :3, :3]
        T = H[:, :3, 3]
        s = torch.ones(N, dtype=T.dtype, device=T.device)
        state_cost = self.cost.__call__(None, R, T, s)
        action_cost = torch.norm(dH, dim=1) if dH is not None else 0
        return state_cost + action_cost * self.action_cost_scale
