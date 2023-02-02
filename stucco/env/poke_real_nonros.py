import enum

import pybullet as p
import numpy as np
import torch
from stucco import voxel
from stucco import sdf

from stucco.env.poke import YCBObjectFactory


class Levels(enum.IntEnum):
    # no clutter environments
    MUSTARD = 0
    DRILL = 4
    # clamp
    CLAMP = 18


task_map = {str(c).split('.')[1]: c for c in Levels}
level_to_obj_map = {
    Levels.MUSTARD: "mustard",
    Levels.DRILL: "drill",
    Levels.CLAMP: "clamp",
}


def obj_factory_map(obj_name):
    if obj_name == "mustard":
        return YCBObjectFactory("mustard", "YcbMustardBottle", scale=1,
                                plausible_suboptimality=0.0003,
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, 1.57 - 0.1]),
                                vis_frame_pos=[-0.005, -0.005, 0.015])
    if obj_name == "drill":
        return YCBObjectFactory("drill", "YcbPowerDrill", scale=1,
                                plausible_suboptimality=0.0006,
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, -0.6]),
                                vis_frame_pos=[-0.002, -0.011, -.06])
    # TODO make sure the clamp is properly scaled with respect to the URDF
    if obj_name == "clamp":
        return YCBObjectFactory("clamp", "YcbMediumClamp", scale=2,
                                plausible_suboptimality=0.0007,
                                vis_frame_rot=p.getQuaternionFromEuler([0.1, 0, 0]),
                                vis_frame_pos=[-0.02, -0.005, -0.0407])


default_freespace_range = np.array([[0.7, 0.8], [-0.1, 0.1], [0.39, 0.45]])


class PokeRealNoRosEnv:
    def __init__(self, environment_level=0, device="cpu", freespace_voxel_resolution=0.01, clean_cache=False):
        p.connect(p.DIRECT)

        self.level = Levels(environment_level)
        self.device = device
        self.dtype = torch.float
        self.freespace_resolution = freespace_voxel_resolution

        # known cabinet workspace, can discard queries outside it
        self.freespace_ranges = np.array([[0.7, 1.1],
                                          [-0.2, 0.2],
                                          [0.33, 0.65]])

        self.free_voxels = None
        self.reset()

        self.obj_factory = obj_factory_map(level_to_obj_map[self.level])
        _, ranges = self.obj_factory.make_collision_obj(0)
        obj_frame_sdf = sdf.MeshSDF(self.obj_factory)
        sdf_resolution = 0.005
        self.target_sdf = sdf.CachedSDF(self.obj_factory.name, sdf_resolution, ranges, obj_frame_sdf,
                                        device=device, clean_cache=clean_cache)

    def reset(self):
        self.free_voxels = voxel.VoxelGrid(self.freespace_resolution, self.freespace_ranges, device=self.device)
        # fill boundaries of the box
        bc, all_pts = voxel.get_coordinates_and_points_in_grid(self.freespace_resolution,
                                                               self.freespace_ranges,
                                                               dtype=self.dtype, device=self.device)
        buffer = 0
        interior_pts = (all_pts[:, 0] > bc[0][buffer]) & (all_pts[:, 0] < bc[0][-buffer-1]) & \
                       (all_pts[:, 1] > bc[1][buffer]) & (all_pts[:, 1] < bc[1][-buffer-1]) & \
                       (all_pts[:, 2] > bc[2][buffer]) & (all_pts[:, 2] < bc[2][-buffer-1])
        boundary_pts = all_pts[~interior_pts]
        self.free_voxels[boundary_pts] = 1

        # self.free_voxels = voxel.ExpandingVoxelGrid(self.freespace_resolution, default_freespace_range,
        #                                             device=self.device)
