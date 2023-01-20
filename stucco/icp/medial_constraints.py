# implements a version of https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2723710/haugo2020iterative.pdf?sequence=2
import typing

import torch
from stucco.env.env import Visualizer
from stucco.registration_util import apply_similarity_transform
from stucco.icp.costs import RegistrationCost
from stucco import sdf


# balls are tensors for efficiency, with each dimension taking on these meanings
class MedialBall:
    X = 0
    Y = 1
    Z = 2
    PX = 3
    PY = 4
    PZ = 5
    R = 6


MedialBallType = torch.tensor


class MedialConstraintCost(RegistrationCost):
    """Cost of transformed model pose intersecting with known freespace medial balls"""

    def __init__(self, medial_balls: MedialBallType, obj_sdf: sdf.ObjectFrameSDF,
                 model_surface_points_world_frame: torch.tensor, scale=1,
                 vis: typing.Optional[Visualizer] = None, scale_medial_ball_penetration=1., scale_surface_points_sdf=1.,
                 obj_factory=None,
                 debug=False):
        """
        :param obj_sdf: signed distance function of the target object in object frame
        :param scale:
        """

        self.medial_balls = medial_balls
        self.sdf = obj_sdf

        self.scale = scale
        self.scale_medial_ball_penetration = scale_medial_ball_penetration
        self.scale_surface_points_sdf = scale_surface_points_sdf

        # the un-batched original in world frame, the batched world frame, and the batched transformed object frame
        self._pts_world_orig = model_surface_points_world_frame
        self._pts_world = None
        self._pts_obj = None
        self._ball_c_world_orig = self.medial_balls[:, MedialBall.X: MedialBall.Z + 1]
        self._ball_c_world = None
        self._ball_c_obj = None

        # batch
        self.B = None

        # intermediate products for visualization purposes
        self.debug = debug

        self.vis = vis
        self.obj_factory = obj_factory
        self.obj_map = {}
        self._penetration = None

    def __call__(self, R, T, s, knn_res=None):
        # assign batch and reuse for later for efficiency
        if self.B is None or self.B != R.shape[0]:
            self.B = R.shape[0]
            self._pts_world = self._pts_world_orig.repeat(self.B, 1, 1)
            self._ball_c_world = self._ball_c_world_orig.repeat(self.B, 1, 1)

        # sdf is evaluated in object frame; we're given points in world frame
        # transform the points via the given similarity transformation parameters, then evaluate their occupancy
        # should transform the interior points from link frame to world frame
        self._transform_model_to_object_frame(R, T, s)

        loss = torch.zeros(self.B, device=self._pts_world_orig.device, dtype=self._pts_world_orig.dtype)

        if self.scale_medial_ball_penetration != 0:
            d, _ = self.sdf(self._ball_c_obj)
            self._penetration = self.medial_balls[:, MedialBall.R] - d
            # non-violating so have 0 cost
            self._penetration[self._penetration < 0] = 0
            bl = self._penetration.square()
            # reduce across each ball
            loss += bl.mean(dim=-1) * self.scale_medial_ball_penetration
        if self.scale_surface_points_sdf != 0:
            d, _ = self.sdf(self._pts_obj)
            l = d.square()
            # reduce across each contact point
            loss += l.mean(dim=-1) * self.scale_surface_points_sdf

        self.visualize(R, T, s)
        return loss * self.scale

    def _transform_model_to_object_frame(self, R, T, s):
        # transform surface points and also ball centers
        Rt = R.transpose(-1, -2)
        tt = (-Rt @ T.reshape(-1, 3, 1)).squeeze(-1)

        self._pts_obj = apply_similarity_transform(self._pts_world, Rt, tt, s)
        self._ball_c_obj = apply_similarity_transform(self._ball_c_world, Rt, tt, s)

    def visualize(self, R, T, s):
        if not self.debug or self.vis is None:
            return
        device = R.device
        dtype = R.dtype
        batch = R.shape[0]
        H = torch.eye(4, device=device, dtype=dtype).repeat(batch, 1, 1)
        # H[:, :3, :3] = R
        # H[:, :3, 3] = T
        sdf.draw_pose_distribution(H[0].unsqueeze(0), self.obj_map, self.vis, self.obj_factory)
        b = 0
        for i, pt in enumerate(self._pts_obj[b]):
            self.vis.draw_point(f"mcpt.{i}", pt.cpu().numpy(), color=(0, 1, 0), length=0.005, scale=1)
        for i, pt in enumerate(self._ball_c_obj[b]):
            c = (0, 0, 1)
            label = None
            r = self.medial_balls[i, MedialBall.R].item()
            if self._penetration[b, i] > 0:
                c = (1, 0, 0)
                # label = f"{self._penetration[b, i] / r * 100:.1f}%"
            self.vis.draw_point(f"mcball.{i}", pt.cpu().numpy(), color=c,
                                length=r, scale=1, label=label)
