import os
import torch
import numpy as np
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import pybullet as p
import matplotlib.pyplot as plt
import logging
from stucco import detection

plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

TEST_DIR = os.path.dirname(__file__)


def test_3d_contact_detection():
    # load the panda gripper into pytorch_volumetric's RobotSDF
    urdf = os.path.join(TEST_DIR, "panda_gripper.urdf")
    chain = pk.build_chain_from_urdf(open(urdf).read())
    d = "cuda" if torch.cuda.is_available() else "cpu"

    chain = chain.to(device=d)
    s = pv.RobotSDF(chain, path_prefix=TEST_DIR)

    # create a contact detector for the gripper
    residual_precision = np.diag([1, 1, 1, 50, 50, 50])
    residual_threshold = 3

    contact_detector = detection.ContactDetector(residual_precision, device=d)
    sensor = detection.RobotResidualContactSensor(s, residual_precision, residual_threshold)
    # visualize the robot in open3d
    contact_detector.register_contact_sensor(sensor)

    try:
        from base_experiments.env.env import draw_ordered_end_points, aabb_to_ordered_end_points
        from base_experiments.env.pybullet_env import DebugDrawer
        import time

        # visualize the robot
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.setAdditionalSearchPath(search_path)
        gripperId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)

        # query_range[0, :] = query_range[0].mean()
        # pv.draw_sdf_slice(s, query_range, resolution=0.005, device=d)

        vis = DebugDrawer(1., 0.5)
        vis.toggle_3d(True)
        vis.set_camera_position([-0.1, 0, 0], yaw=-30, pitch=-20)
        # draw bounding box for each link (set breakpoints here to better see the link frame bounding box)
        tfs = s.sdf.obj_frame_to_each_frame.inverse()
        for i in range(len(s.sdf.sdfs)):
            sdf = s.sdf.sdfs[i]
            aabb = aabb_to_ordered_end_points(np.array(sdf.surface_bounding_box(padding=0)))
            aabb = tfs.transform_points(torch.tensor(aabb, device=tfs.device, dtype=tfs.dtype))[i]
            draw_ordered_end_points(vis, aabb)
            time.sleep(0.2)

        vis.draw_points("surface", sensor._cached_points, length=0.001)
        vis.draw_2d_lines("surface.grad", sensor._cached_points, sensor._cached_normals, color=(0, 1, 0), scale=0.01)
    except ImportError:
        pass

    input()


if __name__ == "__main__":
    test_3d_contact_detection()
