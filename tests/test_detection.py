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

    def draw_mesh(robot_sdf, world_to_obj_tsf=None, positions=None, normals=None, errors=None, positions_gt=None, normals_gt=None):
        # visualize the robot in object frame and the cached surface points
        obj_meshes = get_transformed_meshes(robot_sdf, world_to_obj_tsf)

        # draw the surface points using PointCloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(positions.cpu())
        if errors is None:
            pc.paint_uniform_color([0, 1, 0])
        else:
            # errors as color 
            e_min = errors.min()
            e_max = errors.max()
            e_range = e_max - e_min
            colors = (errors - e_min) / e_range
            colors = torch.stack([1 - colors, 1 - colors, 1 - colors], dim=1) # green to red, green is good? # result is actually highest is better
            # plot the points with color
            pc.colors = o3d.utility.Vector3dVector(colors.cpu())




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

        if positions_gt is not None:
            pc_gt = o3d.geometry.PointCloud()
            pc_gt.points = o3d.utility.Vector3dVector(positions_gt.cpu())
            pc_gt.paint_uniform_color([0, 0, 1])
            # size of the points

            obj_meshes.append(pc_gt)
            geometries = obj_meshes + [pc, normals] + [pc_gt]
        else:
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
    pos = torch.randn(3, device=d)
    rot = pk.random_rotation(device=d)
    gt_tf = pk.Transform3d(pos=pos, rot=rot, device=d)
    positions_world = gt_tf.transform_points(positions)
    normals_world = gt_tf.transform_normals(normals)
    
    if visualize:
        draw_mesh(s, world_to_obj_tsf=gt_tf, positions=positions_world, normals=normals_world)

    # choose a random point on the surface to make contact
    idx = torch.randint(0, len(positions_world), (1,)).item()
    # force direction has to be into the surface (negative dot product with surface normal)
    force_mag = 2
    force = -normals_world[idx] * force_mag
    import pdb; pdb.set_trace()
    print(force.shape)
    # assume the measuring frame is the robot origin
    torque = torch.cross(positions_world[idx], force)
    # transform the force and torque into the measurement frame (end effector frame)
    force = gt_tf.inverse().transform_normals(force.unsqueeze(0))
    torque = gt_tf.inverse().transform_normals(torque.unsqueeze(0))

    EE_FT = torch.cat([force, torque], dim=0)
    EE_FT = EE_FT.reshape(1,6)
    pos = pos.unsqueeze(0)
    # rot to quat
    rot = pk.matrix_to_quaternion(rot)
    rot = rot.unsqueeze(0)
    rot = pk.wxyz_to_xyzw(rot)
    # TODO feed force and torque into the contact detector
    score = sensor.score_contact_points(EE_FT, [pos, rot])
    score.link_frame_pts
    score.world_frame_pts
    score.valid
    score.error
    # TODO score the surface points in terms of explaining the measured resdiual
    # TODO visualize the surface point scores with color indicating their score
    # import pdb; pdb.set_trace()
    if visualize:

        draw_mesh(s, world_to_obj_tsf=gt_tf, positions=positions_world[score.valid], normals=normals_world[score.valid], errors=score.error, positions_gt=positions_world[idx].unsqueeze(0), normals_gt=normals_world[idx].unsqueeze(0))


if __name__ == "__main__":
    test_3d_contact_detection()
