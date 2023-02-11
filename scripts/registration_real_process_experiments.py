import os
import typing
from datetime import datetime
import argparse

import pybullet as p
import pytorch_kinematics as tf
from mmint_camera_utils.camera_utils.camera_utils import project_depth_image
from mmint_camera_utils.ros_utils.utils import pose_to_matrix
from mmint_camera_utils.camera_utils.point_cloud_utils import tr_pointcloud
from arm_pytorch_utilities import rand
from window_recorder.recorder import WindowRecorder

from stucco.env import poke_real
from stucco.env import poke_real_nonros
from stucco import icp
from base_experiments import cfg
from base_experiments.env.env import draw_AABB
from base_experiments.env.real_env import DebugRvizDrawer
import numpy as np
import torch
import logging
import scipy
from stucco import serialization
from stucco.experiments import registration_nopytorch3d
from stucco.icp import initialization
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_pos_rot
from pytorch_volumetric import voxel

try:
    import rospy

    rospy.init_node("poke_processing", log_level=rospy.INFO)
    # without this we get not logging from the library
    import importlib

    importlib.reload(logging)
except RuntimeError as e:
    print("Proceeding without ROS: {}".format(e))

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

experiment_name = 'poke_real_processed'
B = 30
starting_poke = 2


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def free_points_from_gripper(bubbles, sample):
    # complete the gripper from the points since there are gaps in between
    zs = bubbles['left']['pts_world_filtered'][..., 2]
    z = np.linspace(zs.min(), zs.max(), num=30)
    xs = bubbles['left']['pts_world_filtered'][..., 0]
    x = np.linspace(xs.max() - 0.01, xs.max() - 0.005, num=10)
    y = np.linspace(bubbles['left']['pts_world_filtered'][..., 1].min(),
                    bubbles['right']['pts_world_filtered'][..., 1].max(), num=30)
    filling_pts = cartesian_product(x, y, z)
    pts_free = np.concatenate((bubbles['left']['pts_world_filtered'].reshape(-1, 3),
                               bubbles['right']['pts_world_filtered'].reshape(-1, 3), filling_pts))

    # vis.draw_points(f"known.2", filling_pts.reshape(-1, 3), color=(0, 0.8, 1), scale=0.1)
    extend_along = -sample['wrist_z']
    pts_to_set = []
    for delta in np.linspace(0.02, 0.15, num=10):
        pts_to_set.append(pts_free + extend_along * delta)
    pts_to_set = np.concatenate(pts_to_set)
    return pts_free, pts_to_set


def contact_points_from_gripper(bubbles, sample, imprint_threshold=0.004):
    contact_pts = []
    for camera in ['left', 'right']:
        data = bubbles[camera]
        pts = data['pts_world_filtered']

        imprint = data['imprint_final']
        imprint = imprint[..., 0]
        # first smooth the imprint
        imprint_filtered = scipy.ndimage.uniform_filter(imprint, size=10, output=None, mode='reflect',
                                                        cval=0.0, origin=0)
        logger.info(f"{camera} max filtered imprint {imprint_filtered.max()}")
        # get top percentile
        above_quantile = np.quantile(imprint_filtered, 0.9)
        mask = (imprint_filtered > above_quantile) & (imprint_filtered > imprint_threshold)
        contact = pts[mask]

        # find local minima using the gradient
        # img_grad = np.gradient(imprint_filtered)
        # g = np.stack(img_grad, axis=-1)
        # gmag = np.linalg.norm(g, axis=-1)
        # u, v = np.where(gmag < np.quantile(gmag, 0.01))
        # test_imprints = imprint_filtered[u, v]
        # mask = test_imprints > imprint_threshold
        # contact = pts[u[mask], v[mask]]

        # num_test_points = 200
        # test_u = np.random.randint(0, imprint_filtered.shape[0], num_test_points)
        # test_v = np.random.randint(0, imprint_filtered.shape[1], num_test_points)
        # test_uv = np.stack((test_u, test_v))
        # for i in range(100):
        #     this_grad = img_grad[test_uv]
        #     test_uv += this_grad
        #
        # test_uv = np.unique(test_uv)

        # doesn't work because filter will always find some local min inside its window
        # then extract local min
        # imprint_local_min = scipy.ndimage.minimum_filter(-imprint_filtered, size=7, footprint=None, output=None,
        #                                                  mode='constant', cval=0.0, origin=0)
        # imprint_local_max = -imprint_local_min
        # mask = (imprint_local_max == imprint_filtered) & (imprint_filtered > imprint_threshold)
        # contact = pts[mask]

        # cluster contact into voxels
        pts_threshold = 50
        if len(contact) > pts_threshold:
            # random subsample
            contact = np.random.permutation(contact)[:pts_threshold]

            # cluster via voxelization
            # ranges = np.stack((contact.min(axis=0), contact.max(axis=0)))
            # contact_voxel = voxel.VoxelGrid(0.003, ranges.T)
            # contact_voxel[torch.tensor(contact)] = 1
            # contact, _ = contact_voxel.get_known_pos_and_values()
            # contact = contact.cpu().numpy()

            # k-means cluster
            # from sklearn.cluster import KMeans
            # kmeans = KMeans(n_clusters=8, n_init='auto').fit(contact)
            # contact = kmeans.cluster_centers_

        contact_pts.append(contact)

    return contact_pts


def free_points_from_camera(depth, K):
    # project depth image that's been "diluted" to be brought closer to the camera into points
    if len(depth.shape) > 2:
        depth = depth[..., 0]
    # for backward compatibility, in case it's given in mm
    if np.any(depth > 1000):
        depth = depth.astype(np.float32) / 1000
    pts_to_set = []
    for level in np.linspace(0.95, 0.4, 20):
        d = depth * level
        pts = project_depth_image(d, K, usvs=None)
        pts_to_set.append(pts.reshape(-1, 3))
    pts_to_set = np.concatenate(pts_to_set)
    return pts_to_set


def export_pc_to_register(point_cloud_file: str, pokes_to_data):
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'w') as f:
        for poke, data in pokes_to_data.items():
            # write out the poke index and the size of the point cloud
            total_pts = len(data['free']) + len(data['contact'])
            f.write(f"{poke} {total_pts}\n")
            serialization.export_pcs(f, data['free'], data['contact'])


def extract_known_points(task, to_process_seed, vis: typing.Optional[DebugRvizDrawer] = None,
                         clean_cache=False):
    p.connect(p.DIRECT)
    device = 'cpu'
    data_path = os.path.join(cfg.DATA_DIR, poke_real.DIR)
    dataset = poke_real.PokeBubbleDataset(data_name=data_path, load_shear=False, load_cache=True)

    # values for each trajectory
    pokes_to_data = {}
    env: typing.Optional[poke_real_nonros.PokeRealNoRosEnv] = None
    cur_seed = None
    cur_poke = None
    contact_pts = []
    ee_pos = []
    pts_free = None
    prev_pts_free = None

    def end_current_trajectory(new_seed):
        nonlocal cur_seed, env, contact_pts, pokes_to_data, ee_pos
        logger.info(f"{task.name} finished seed {cur_seed} processing {new_seed}")
        # ending an "empty" first trajectory
        if cur_seed is None:
            pass
        else:
            point_cloud_file = f"{registration_nopytorch3d.saved_traj_dir_base(task, experiment_name=experiment_name)}_{cur_seed}.txt"
            export_pc_to_register(point_cloud_file, pokes_to_data)

            if vis is not None:
                ee_pos = np.concatenate(ee_pos)
                ee_pos = ee_pos[:, :3]
                diffs = ee_pos[1:] - ee_pos[:-1]
                vis.draw_2d_lines("trajectory", ee_pos[:-1], diffs, color=(0, 0, 1), size=0.2)
            pc_register_against_file = f"{registration_nopytorch3d.saved_traj_dir_base(task, experiment_name=experiment_name)}.txt"
            if not os.path.exists(pc_register_against_file) or clean_cache:
                surface_thresh = 0.002
                serialization.export_pc_register_against(pc_register_against_file, env.target_sdf,
                                                         surface_thresh=surface_thresh)
                if vis is not None:
                    pc_surface = env.target_sdf.get_filtered_points(
                        lambda voxel_sdf: (voxel_sdf < surface_thresh) & (voxel_sdf > -surface_thresh))
                    vis.draw_points(f"surface.0", pc_surface, color=(1, 1, 1), scale=0.1)

        cur_seed = new_seed
        env = poke_real_nonros.PokeRealNoRosEnv(task, device=device)
        contact_pts = []
        pokes_to_data = {}
        ee_pos = []

    def end_current_poke(new_poke):
        nonlocal cur_poke, contact_pts
        free_surface_file = f"{registration_nopytorch3d.saved_traj_dir_base(task, experiment_name=experiment_name)}_{cur_seed}_free_surface.txt"
        # empty poke
        if cur_poke is None:
            pass
        else:
            # first poke, remove file
            if cur_poke == 1:
                try:
                    os.remove(free_surface_file)
                except OSError:
                    pass
            if cur_poke == starting_poke:
                best_tsf_guess = initialization.initialize_transform_estimates(B=B,
                                                                               freespace_ranges=env.freespace_ranges,
                                                                               init_method=initialization.InitMethod.RANDOM,
                                                                               contact_points=pokes_to_data[cur_poke][
                                                                                   'contact'])
                transform_file = f"{registration_nopytorch3d.saved_traj_dir_base(env.level, experiment_name=experiment_name)}_{seed}_trans.txt"
                serialization.export_init_transform(transform_file, best_tsf_guess)
                if vis is not None:
                    draw_AABB(vis, env.freespace_ranges)
            serialization.export_free_surface(free_surface_file, env.free_voxels, cur_poke)

        # known contact points from front of gripper for certain tasks and pokes
        if task == poke_real_nonros.Levels.MUSTARD and cur_poke in (1, 2):
            num = 5
            tip = np.argpartition(-prev_pts_free[:, 0], num)[:5]
            pt = prev_pts_free[tip].mean(axis=0).reshape(1, -1)
            contact_pts += [pt]

        cur_poke = new_poke

    for i, sample in enumerate(dataset):
        seed = sample['seed']
        poke = sample['poke']
        if poke_real_nonros.Levels[sample['task']] != task:
            continue
        if seed != to_process_seed:
            continue

        # start of new poke
        if cur_poke is None or cur_poke != poke:
            end_current_poke(poke)
        # start of new trajectory
        if cur_seed is None or cur_seed != seed:
            end_current_trajectory(seed)

        ee_pos.append(sample['info']['p'])
        bubbles = dataset.get_bubble_info(i)
        # complete the gripper from the points since there are gaps in between
        prev_pts_free = pts_free
        pts_free, pts_to_set = free_points_from_gripper(bubbles, sample)
        env.free_voxels[torch.from_numpy(pts_to_set)] = 1

        scene = dataset.get_scene_info(i, 1)
        camera_free_pts_cf = free_points_from_camera(scene['depth'], scene['K'])
        # transform points from camera frame to world frame
        w_X_cf = pose_to_matrix(scene['camera_tf'])
        camera_free_pts = tr_pointcloud(camera_free_pts_cf, w_X_cf[:3, :3], w_X_cf[:3, 3])  # in world frame

        env.free_voxels[torch.from_numpy(camera_free_pts)] = 1

        contact = contact_points_from_gripper(bubbles, sample)
        contact_pts += contact

        c_pts = np.concatenate(contact_pts)

        f_pts, _ = env.free_voxels.get_known_pos_and_values()
        logger.info(f"Poke {poke} with {len(c_pts)} contact points {len(f_pts)} free points")
        pokes_to_data[poke] = {'free': f_pts, 'contact': c_pts}

        if vis is not None:
            vis.draw_points(f"known.0", pts_free, color=(0, 1, 1), scale=0.1)

            # only draw interior free points
            bc, all_pts = voxel.get_coordinates_and_points_in_grid(env.freespace_resolution,
                                                                   env.freespace_ranges,
                                                                   dtype=env.dtype, device=env.device)
            buffer = 0
            interior_pts = (f_pts[:, 0] > bc[0][buffer]) & (f_pts[:, 0] < bc[0][-buffer - 1]) & \
                           (f_pts[:, 1] > bc[1][buffer]) & (f_pts[:, 1] < bc[1][-buffer - 1]) & \
                           (f_pts[:, 2] > bc[2][buffer]) & (f_pts[:, 2] < bc[2][-buffer - 1])
            f_pts = f_pts[interior_pts]

            vis.draw_points(f"free.0", f_pts, color=(1, 0, 1), scale=0.2)
            # vis.draw_points(f"camera_free.0", camera_free_pts, color=(1, 0.5, 1), scale=0.2)
            vis.draw_points(f"contact.0", c_pts, color=(1, 0, 0), scale=0.5)
            # for the teaser image, isolate the contribution from the camera
            free_voxels = voxel.VoxelGrid(env.freespace_resolution, env.freespace_ranges)
            free_voxels[torch.from_numpy(camera_free_pts)] = 1
            cf_pts, _ = free_voxels.get_known_pos_and_values()
            vis.draw_points(f"camera_free.0", cf_pts, color=(1, 0.5, 1), scale=0.2)

    # last poke of the last sample
    end_current_poke(None)
    end_current_trajectory(None)


def set_approximate_pose(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer):
    pose = None
    # use breakpoints
    approx_pose_file = registration_nopytorch3d.approximate_pose_file(env.level, experiment_name=experiment_name)
    if os.path.exists(approx_pose_file):
        pose = serialization.import_pose(approx_pose_file)
        rot = tf.xyzw_to_wxyz(torch.tensor(pose[1]))
        euler = tf.matrix_to_euler_angles(tf.quaternion_to_matrix(rot), "ZYX")
        logger.info(f"{pose[0]} {euler}")
        env.obj_factory.draw_mesh(vis, "hand_placed_obj", pose, (1, 1, 1, 0.5), object_id=0)

    while True:
        try:
            nums = [float(v) for v in input('xyz then ypr').split()]
            pos = nums[:3]
            rot = nums[3:]
            # xyzw quaternions
            rot = tf.matrix_to_quaternion(tf.euler_angles_to_matrix(torch.tensor(rot), "ZYX"))
            rot = tf.wxyz_to_xyzw(rot)
            pose = (pos, list(rot.numpy()))
            logger.info(pose)
        except:
            break
        env.obj_factory.draw_mesh(vis, "hand_placed_obj", pose, (1, 1, 1, 0.5), object_id=0)

    # rotation saved as xyzw
    serialization.export_pose(approx_pose_file, pose)


def plot_optimal_pose(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer, seed):
    optimal_pose_file = registration_nopytorch3d.optimal_pose_file(env.level, seed=seed,
                                                                   experiment_name=experiment_name)
    pose = serialization.import_pose(optimal_pose_file)
    logger.info(pose)
    env.obj_factory.draw_mesh(vis, "optimal_pose", pose, (0.5, 0.8, 0.9, 0.5), object_id=0)


def randomly_downsample(seq, seed, num=10):
    rand.seed(seed)
    selected = np.random.permutation(range(len(seq)))
    seq = seq[selected[:num]]
    return seq


def plot_plausible_set(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer, seed, poke):
    filename = f"{registration_nopytorch3d.saved_traj_dir_base(env.level, experiment_name=experiment_name)}_plausible_set_{seed}.pkl"
    plausible_set = torch.load(filename)
    if isinstance(plausible_set, tuple):
        plausible_set, _ = plausible_set
    plausible_transforms = plausible_set[poke]
    logger.info(
        f"plotting plausible set for {env.level.name} seed {seed} poke {poke} ({len(plausible_transforms)})")
    ns = "plausible_pose"
    vis.clear_visualization_after(ns, 0)
    # show some of them
    plausible_transforms = randomly_downsample(plausible_transforms, 19, num=8)
    for i, T in enumerate(plausible_transforms):
        pose = matrix_to_pos_rot(T)
        env.obj_factory.draw_mesh(vis, ns, pose, (0., .0, 0.8, 0.1), object_id=i)


def plot_estimate_set(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer, seed, reg_method, poke=-1):
    from view_controller_msgs.msg import CameraPlacement
    from geometry_msgs.msg import Point, Vector3
    from math import cos, pi, sin

    estimate_set, _, _ = registration_nopytorch3d.read_offline_output(reg_method, env.level, seed, poke,
                                                                      experiment_name)
    estimate_set = estimate_set.inverse()
    logger.info(f"plotting estimate set for {reg_method.name} on {env.level.name} seed {seed} poke {poke}")
    ns = "estimate_set"
    vis.clear_visualization_after(ns, 0)
    estimate_set = randomly_downsample(estimate_set, 64, num=8)
    for i, T in enumerate(estimate_set):
        pose = matrix_to_pos_rot(T)
        env.obj_factory.draw_mesh(vis, ns, pose, (0., .0, 0.8, 0.1), object_id=i)

    pub = rospy.Publisher("/rviz/camera_placement", CameraPlacement, queue_size=1)

    rate_float = 10
    rate = rospy.Rate(rate_float)

    optimal_pose_file = registration_nopytorch3d.optimal_pose_file(env.level, seed=seed,
                                                                   experiment_name=experiment_name)
    pose = serialization.import_pose(optimal_pose_file)

    t_offset = 0
    t_start = rospy.get_time()

    while not rospy.is_shutdown():
        t = rospy.get_time()
        cp = CameraPlacement()
        r = 0.5
        period = 12

        # cp.target_frame = "wsg50_finger_left_tip"
        # cp.target_frame = "obj_frame"
        dt = t - t_start + t_offset

        p = Point(r * cos(2 * pi * dt / period) + pose[0][0], r * sin(2 * pi * dt / period) + pose[0][1], 0.65)
        # p = Point(5,5,0)
        cp.eye.point = p
        cp.eye.header.frame_id = "world"

        # f = Point(0, 0, 2*cos(2*pi*t/5))
        f = Point(0.9, 0, 0.34)
        cp.focus.point = f
        cp.focus.header.frame_id = "world"

        up = Vector3(0, 0, 1)
        # up = Vector3(0, sin(2*pi*t/10), cos(2*pi*t/10))
        cp.up.vector = up
        cp.up.header.frame_id = "world"

        cp.time_from_start = rospy.Duration(1.0 / rate_float)
        "Publishing a message!"
        pub.publish(cp)
        # print "Sleeping..."
        rate.sleep()
        # complete a cycle
        if dt > period * 1.02:
            break


def main(args):
    task = task_map[args.task]
    registration_method = registration_map[args.registration]

    if args.no_gui:
        vis = None
    else:
        vis = DebugRvizDrawer(world_frame='world')

    if args.experiment == "extract-known-points":
        extract_known_points(task, args.seed, vis=vis)
    elif args.experiment == "plot-sdf":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda", clean_cache=True)

        def filter(pts):
            c1 = (pts[:, 0] > -0.15) & (pts[:, 0] < 0.15)
            c2 = (pts[:, 1] > 0.) & (pts[:, 1] < 0.2)
            c3 = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.4)
            c = c1 & c2 & c3
            return pts[c]

        with WindowRecorder(name_suffix="rviz",
                            frame_rate=30.0,
                            save_dir=cfg.VIDEO_DIR):
            import time
            time.sleep(6)
            # registration_nopytorch3d.plot_sdf(env.obj_factory, env.target_sdf, vis, filter_pts=filter)

    elif args.experiment == "set-approximate-pose":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        set_approximate_pose(env, vis)
    elif args.experiment == "plot-optimal-pose":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        plot_optimal_pose(env, vis, args.seed)
    elif args.experiment == "plot-plausible-set":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        plot_plausible_set(env, vis, args.seed, args.poke)
    elif args.experiment == "plot-estimate-set":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        with WindowRecorder(["video_zoom.rviz* - RViz"], name_suffix="rviz",
                            frame_rate=30.0,
                            save_dir=cfg.VIDEO_DIR):
            plot_estimate_set(env, vis, args.seed, registration_method, args.poke)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pokes from a real robot')
    parser.add_argument('--experiment',
                        choices=['extract-known-points', 'plot-sdf', 'set-approximate-pose', 'plot-optimal-pose',
                                 'plot-plausible-set', 'plot-estimate-set'],
                        default='extract-known-points',
                        help='which experiment to run')
    registration_map = {m.name.lower().replace('_', '-'): m for m in icp.ICPMethod}
    parser.add_argument('--registration',
                        choices=registration_map.keys(),
                        default='volumetric',
                        help='which registration method to run')
    parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
    task_map = {level.name.lower(): level for level in poke_real_nonros.Levels}
    parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
    parser.add_argument('--no_gui', action='store_true', help='force no GUI')
    parser.add_argument('--seed', type=int, default=0, help='random seed to process')
    parser.add_argument('--poke', type=int, default=2, help='poke for some experiments that need it specified')

    args = parser.parse_args()
    main(args)
