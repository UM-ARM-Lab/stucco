import torch
from torch.nn import MSELoss
from pytorch3d.ops.knn import knn_gather, _KNN
from pytorch3d.ops.points_alignment import _apply_similarity_transform
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
        transformed_norms = s[:, None, None] * torch.bmm(self.Xnorm, R)
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
            grads = ctx.occupied[:, :, None] * -world_frame_interior_gradients * interior_point_weights[None, :, None]
            # TODO consider averaging the gradients out across all points?
            dl_dpts = grad_outputs[:, :, None] * grads

        # gradients for the other inputs not implemented
        return dl_dpts, dl_dgrad, dl_dweights, dl_dvoxels


class VolumetricCost(ICPPoseCost):
    """Cost of transformed model pose intersecting with known freespace voxels"""

    def __init__(self, voxels: util.VoxelGrid, obj_sdf: util.ObjectFrameSDF, scale=1, vis=None, debug=False):
        """
        :param voxels: representation of freespace
        :param model_interior_points: points on the inside of the model (not on the surface)
        :param obj_sdf: signed distance function of the target object in object frame
        :param scale:
        """

        self.voxels = voxels
        self.scale = scale
        # SDF gives us a volumetric representation of the target object
        self.sdf = obj_sdf

        # ---- for +, known free space points, we just have to care about interior points of the object
        # to facilitate comparison between volumes that are rotated, we sample points at the center of the object voxels
        interior_threshold = -0.01
        model_voxels = self.sdf.get_voxel_view()
        interior = model_voxels.raw_data < interior_threshold
        indices = interior.nonzero()
        # these points are in object frame
        self.model_interior_points = model_voxels.ensure_value_key(indices)
        self.model_interior_weights, self.model_interior_normals = self.sdf(self.model_interior_points)
        self.model_interior_weights *= -1
        # batch
        self.B = None

        # intermediate products for visualization purposes
        self._pts = None
        self._grad = None
        self.debug = debug

        self.vis = vis

    def __call__(self, knn_res: _KNN, R, T, s):
        # assign batch and reuse for later for efficiency
        if self.B is None:
            self.B = R.shape[0]
            self.model_interior_points = self.model_interior_points.repeat(self.B, 1, 1)
            self.model_interior_normals = self.model_interior_normals.repeat(self.B, 1, 1)

        # voxels are in world frame
        # need points transformed into world frame
        # transform the points via the given similarity transformation parameters, then evaluate their occupancy
        # should transform the interior points from link frame to world frame
        self._transform_model_to_world_frame(R, T, s)
        # self.visualize(R, T, s)

        known_free_space_loss = KnownFreeSpaceCost.apply(self._pts, self._grad, self.model_interior_weights,
                                                         self.voxels)
        loss = known_free_space_loss
        return loss.sum(dim=-1) * self.scale

    def _transform_model_to_world_frame(self, R, T, s):
        Rt = R.transpose(-1, -2)
        self._pts = _apply_similarity_transform(self.model_interior_points, Rt, (-Rt @ T.view(-1, 3, 1)).squeeze(), s)
        if self.debug:
            self._pts.retain_grad()
        self._grad = torch.bmm(self.model_interior_normals, R.transpose(-1, -2))

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
                point_grads = self._pts.grad
                have_gradients = point_grads.sum(dim=-1).sum(dim=-1) != 0

                batch_with_gradients = have_gradients.nonzero()
                for b in batch_with_gradients:
                    b = b.item()
                    # only visualize it for one sample at a time
                    if b != 0:
                        continue
                    i = 0
                    for i, pt in enumerate(self._pts[b]):
                        self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
                        self.vis.draw_2d_line(f"min.{i}", pt, self._grad[b][i], color=(1, 0, 0), size=2., scale=0.02)
                        # visualize the computed gradient on the point
                        self.vis.draw_2d_line(f"mingrad.{i}", pt, self._pts.grad[b, i], color=(0, 1, 0), size=5.,
                                              scale=10)
                    self.vis.clear_visualization_after("mipt", i + 1)
                    self.vis.clear_visualization_after("min", i + 1)
                    self.vis.clear_visualization_after("mingrad", i + 1)
