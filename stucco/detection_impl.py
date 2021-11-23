import os
import logging

import numpy as np
import pybullet as p
import torch
from stucco.detection import ContactDetectorPlanar
from pytorch_kinematics import transforms as tf
from stucco import cfg
from stucco.env.pybullet_env import ContactInfo, closest_point_on_surface
import typing

logger = logging.getLogger(__name__)


class ContactDetectorPlanarPybulletGripper(ContactDetectorPlanar):
    """Leverage pybullet to sample points on the robot;
    if the sampled robot points and normals are cached then pybullet information can be omitted."""

    def __init__(self, name, *args, sample_pt_min_separation=0.005, num_sample_points=100, robot_id=None,
                 canonical_pos=None, canonical_orientation=None, default_joint_config=None, visualizer=None, **kwargs):
        self.name = name
        self.robot_id = robot_id
        self.num_sample_points = num_sample_points
        self._canonical_pos = canonical_pos
        self._canonical_orientation = canonical_orientation
        self._default_joint_config = default_joint_config

        self._sample_pt_min_separation = sample_pt_min_separation

        super().__init__(None, None, *args, **kwargs)
        self._cached_points, self._cached_normals = self._load_sample_surface_points_in_canonical_pose(visualizer)

    def _project_sample_points_to_surface(self, body_ids: typing.List[int], ee_pos, ee_orientation, sampled_points,
                                          visualizer=None):
        cached_points = []
        cached_normals = []
        z = ee_pos[2]

        for pt in sampled_points:
            min_pts = [closest_point_on_surface(robot_id, pt) for robot_id in body_ids]
            min_pt = min(min_pts, key=lambda test_pt: test_pt[ContactInfo.DISTANCE])
            min_pt_at_z = [min_pt[ContactInfo.POS_A][0], min_pt[ContactInfo.POS_A][1], z]
            if len(cached_points) > 0:
                d = np.subtract(cached_points, min_pt_at_z)
                d = np.linalg.norm(d, axis=1)
                if np.any(d < self._sample_pt_min_separation):
                    continue
            cached_points.append(min_pt_at_z)
            normal = min_pt[ContactInfo.POS_B + 1]
            cached_normals.append([-normal[0], -normal[1], 0])

        if visualizer is not None:
            for i, min_pt_at_z in enumerate(cached_points):
                t = i / len(cached_points)
                visualizer.draw_point(f'c.{t}', min_pt_at_z, color=(t, t, 1 - t))
                visualizer.draw_2d_line(f'n.{i}', min_pt_at_z, cached_normals[i], color=(0.5, 0, 0), size=2., scale=0.1)

        # convert points back to link frame
        trans = tf.Transform3d(pos=ee_pos, rot=tf.xyzw_to_wxyz(torch.tensor(ee_orientation)), dtype=self.dtype,
                               device=self.device).inverse()
        cached_points = trans.transform_points(torch.tensor(cached_points, device=self.device, dtype=self.dtype))
        cached_normals = trans.transform_normals(torch.tensor(cached_normals, device=self.device, dtype=self.dtype))

        return cached_points, cached_normals

    def _load_sample_surface_points_in_canonical_pose(self, visualizer=None):
        fullname = os.path.join(cfg.DATA_DIR, f'detection_{self.name}_cache.pkl')
        if os.path.exists(fullname):
            logger.info("cached robot points and normals loaded from %s", fullname)
            return torch.load(fullname)

        cached_points, cached_normals = self._init_sample_surface_points_in_canonical_pose(visualizer=visualizer)

        torch.save((cached_points, cached_normals), fullname)
        logger.info("robot points and normals saved to %s", fullname)
        return cached_points, cached_normals

    def _init_sample_surface_points_in_canonical_pose(self, visualizer=None):
        # load if possible; otherwise would require a running pybullet instance
        evenly_sample = True
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(self.robot_id)
        z = orig_pos[2]

        # first reset to canonical location
        canonical_pos = [0, 0, z]
        p.resetBasePositionAndOrientation(self.robot_id, canonical_pos, self._canonical_orientation)
        for i, joint_value in enumerate(self._default_joint_config):
            p.resetJointState(self.robot_id, i, joint_value)

        sample_pts = []
        if evenly_sample:
            r = 0.115
            # sample evenly in terms of angles, but leave out the section in between the fingers
            leave_out = 0.01
            angles = np.linspace(leave_out, np.pi * 2 - leave_out, self.num_sample_points)
            for angle in angles:
                sample_pts.append([np.cos(angle) * r, np.sin(angle) * r, z])
        else:
            # randomly sample
            sigma = 0.2
            for _ in range(self.num_sample_points):
                sample_pts.append(np.r_[np.random.randn(2) * sigma, z])

        cached_points, cached_normals = self._project_sample_points_to_surface([self.robot_id], canonical_pos,
                                                                               self._canonical_orientation, sample_pts,
                                                                               visualizer=visualizer)

        # p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        # if visualizer is not None:
        #     for i, min_pt_at_z in enumerate(self._cached_points):
        #         t = i / len(self._cached_points)
        #         visualizer.draw_point(f'c{t}', min_pt_at_z, color=(t, t, 1 - t))

        p.resetBasePositionAndOrientation(self.robot_id, orig_pos, orig_orientation)
        return cached_points, cached_normals
