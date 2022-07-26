import torch
from torch.nn import MSELoss
from pytorch3d.ops.knn import knn_gather, _KNN
from pytorch3d.ops.points_alignment import _apply_similarity_transform

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


class FreeSpaceCost(ICPPoseCost):
    """Cost of transformed model pose intersecting with known freespace voxels"""

    def __init__(self, voxels: util.VoxelGrid, model_interior_points, model_interior_point_weights, scale=1, vis=None):
        """
        :param voxels: representation of freespace
        :param model_interior_points: points on the inside of the model (not on the surface)
        :param scale:
        """

        self.voxels = voxels
        self.scale = scale
        self.model_interior_points = model_interior_points
        self.weights = model_interior_point_weights
        self.vis = vis

    def __call__(self, knn_res: _KNN, R, T, s):
        # voxels are in world frame
        # need points transformed into world frame
        # transform the points via the given similarity transformation parameters, then evaluate their occupancy
        # should transform the interior points from link frame to world frame
        pts = _apply_similarity_transform(self.model_interior_points, R.transpose(-1, -2),
                                          (-R.transpose(-1, -2) @ T.view(-1, 3, 1)).squeeze(), s)

        occupied = self.voxels[pts]
        loss = occupied * self.weights
        return loss.sum(dim=-1) * self.scale

    def visualize(self, R, T, s):
        if self.vis is not None:
            with torch.no_grad():
                # occupied = self.voxels.voxels.raw_data > 0
                # indices = occupied.nonzero()
                # coord = self.voxels.voxels.ensure_value_key(indices)
                # for i, xyz in enumerate(coord):
                #     self.vis.draw_point(f"free.{i}", xyz, color=(1, 0, 0), scale=5)
                pts = _apply_similarity_transform(self.model_interior_points, R.transpose(-1, -2),
                                                  (-R.transpose(-1, -2) @ T.view(-1, 3, 1)).squeeze(), s)
                i = 0
                for i, pt in enumerate(pts[0].detach()):
                    self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
                self.vis.clear_visualization_after("mipt", i + 1)
