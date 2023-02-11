import logging
import time

import pytorch_kinematics.transforms.rotation_conversions

import base_experiments.util
import numpy as np
import torch

from pytorch_volumetric.sdf import ObjectFactory, ObjectFrameSDF
from base_experiments.env.pybullet_env import closest_point_on_surface, ContactInfo

logger = logging.getLogger(__name__)


class PyBulletNaiveSDF(ObjectFrameSDF):
    def __init__(self, test_obj_id, vis=None):
        self.test_obj_id = test_obj_id
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

                if self.vis is not None:
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


def draw_pose_distribution(link_to_world_tf_matrix, obj_id_map, dd, obj_factory: ObjectFactory, sequential_delay=None,
                           show_only_latest=False, max_shown=15):
    m = link_to_world_tf_matrix
    if max_shown is not None:
        idx = np.random.permutation(range(len(m)))
        m = m[idx[:max_shown]]

    for b in range(len(m)):
        mm = m[b]
        pos, rot = pytorch_kinematics.transforms.rotation_conversions.matrix_to_pos_rot(mm)
        # if we're given a sequential delay, then instead of drawing the distribution simultaneously, we render them
        # sequentially
        if show_only_latest:
            b = 0
            if sequential_delay is not None and sequential_delay > 0:
                time.sleep(sequential_delay)

        object_id = obj_id_map.get(b, None)
        object_id = obj_factory.draw_mesh(dd, "icp_distribution", (pos, rot), (0.7, 0.7, 0.7, 0.1), object_id=object_id)
        obj_id_map[b] = object_id


