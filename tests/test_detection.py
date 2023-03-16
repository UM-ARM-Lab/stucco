import os
import torch
import numpy as np
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from pytorch_seed import seed
import matplotlib.pyplot as plt
import open3d as o3d
import logging

from pytorch_volumetric.visualization import get_transformed_meshes
from stucco import detection

plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

TEST_DIR = os.path.dirname(__file__)


def test_3d_contact_detection():
    visualize = True
    first_rotate = False
    seed(1)

    def rotate_view(vis):
        nonlocal first_rotate
        ctr = vis.get_view_control()
        if first_rotate is False:
            ctr.rotate(0.0, -540.0)
            first_rotate = True
        ctr.rotate(5.0, 0.0)
        return False

    def draw_mesh(robot_sdf, world_to_obj_tsf=None, positions=None, normals=None):
        # visualize the robot in object frame and the cached surface points
        obj_meshes = get_transformed_meshes(robot_sdf, world_to_obj_tsf)

        # draw the surface points using PointCloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(positions.cpu())
        pc.paint_uniform_color([0, 1, 0])

        # also draw the surface normals using LineSet
        length = 0.01
        end_points = positions + normals * length
        pts = torch.cat([positions, end_points], dim=0)
        lines = torch.arange(0, len(positions))
        lines = torch.cat([lines.view(-1, 1), (lines + len(positions)).view(-1, 1)], dim=1)
        normals = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts.cpu()),
            lines=o3d.utility.Vector2iVector(lines.cpu())
        )
        normals.paint_uniform_color([1, 0, 0])

        geometries = obj_meshes + [pc, normals]
        o3d.visualization.draw_geometries_with_animation_callback(geometries, rotate_view)
        return geometries

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
    sensor = detection.RobotResidualContactSensor(s, residual_precision, residual_threshold, max_num_points=2000)
    # visualize the robot in open3d
    contact_detector.register_contact_sensor(sensor)

    # model surface points and normals
    positions = sensor._cached_points
    normals = sensor._cached_normals

    if visualize:
        draw_mesh(s, positions=positions, normals=normals)

    # random pose
    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), device=d)
    positions_world = gt_tf.transform_points(positions)
    normals_world = gt_tf.transform_normals(normals)

    if visualize:
        draw_mesh(s, world_to_obj_tsf=gt_tf, positions=positions_world, normals=normals_world)

    # choose a random point on the surface to make contact
    idx = torch.randint(0, len(positions_world), (1,)).item()
    # force direction has to be into the surface (negative dot product with surface normal)
    force_mag = 2
    force = -normals_world[idx] * force_mag
    # assume the measuring frame is the robot origin
    torque = torch.cross(positions_world[idx], force)

    # TODO feed force and torque into the contact detector
    # TODO score the surface points in terms of explaining the measured resdiual
    # TODO visualize the surface point scores with color indicating their score


if __name__ == "__main__":
    test_3d_contact_detection()
