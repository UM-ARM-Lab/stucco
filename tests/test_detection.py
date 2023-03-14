import copy
import os
import torch
import numpy as np
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import matplotlib.pyplot as plt
import open3d as o3d
import logging
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

    def rotate_view(vis):
        nonlocal first_rotate
        ctr = vis.get_view_control()
        if first_rotate is False:
            ctr.rotate(0.0, -540.0)
            first_rotate = True
        ctr.rotate(5.0, 0.0)
        return False

    def get_transformed_meshes(robot_sdf: pv.RobotSDF):
        meshes = []
        tsfs = robot_sdf.sdf.obj_frame_to_link_frame.inverse().get_matrix()
        for i in range(len(robot_sdf.sdf_to_link_name)):
            mesh = copy.deepcopy(robot_sdf.sdf.sdfs[i].obj_factory._mesh)
            mesh = mesh.transform(tsfs[i].cpu().numpy())
            meshes.append(mesh)
        return meshes

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

    if visualize:
        # visualize the robot in object frame and the cached surface points
        obj_meshes = get_transformed_meshes(s)

        # draw the surface points using PointCloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sensor._cached_points.cpu())
        pc.paint_uniform_color([0, 1, 0])

        # also draw the surface normals using LineSet
        length = 0.01
        end_points = sensor._cached_points + sensor._cached_normals * length
        pts = torch.cat([sensor._cached_points, end_points], dim=0)
        lines = torch.arange(0, len(sensor._cached_points))
        lines = torch.cat([lines.view(-1, 1), (lines + len(sensor._cached_points)).view(-1, 1)], dim=1)
        normals = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts.cpu()),
            lines=o3d.utility.Vector2iVector(lines.cpu())
        )
        normals.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries_with_animation_callback(obj_meshes + [pc, normals], rotate_view)


if __name__ == "__main__":
    test_3d_contact_detection()
