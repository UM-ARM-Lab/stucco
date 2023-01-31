import os
import typing
from datetime import datetime
import argparse

import pybullet as p
import pytorch_kinematics as tf

from stucco.env import poke_real
from stucco.env import poke_real_nonros
from stucco import cfg
from stucco.env.env import draw_AABB
from stucco.env.real_env import DebugRvizDrawer
import numpy as np
import torch
import logging
import scipy
from stucco import serialization
from stucco.experiments import registration_nopytorch3d
from stucco.icp import initialization
from stucco.util import matrix_to_pos_rot

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
        imprint_filtered = scipy.ndimage.median_filter(imprint, size=5, footprint=None, output=None, mode='reflect',
                                                       cval=0.0, origin=0)
        # find local minima using the gradient
        img_grad = np.gradient(imprint_filtered)
        g = np.stack(img_grad, axis=-1)
        gmag = np.linalg.norm(g, axis=-1)
        u, v = np.where(gmag == 0)
        test_imprints = imprint_filtered[u, v]
        mask = test_imprints > imprint_threshold
        contact = pts[u[mask], v[mask]]

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
        contact_pts.append(contact)

    return contact_pts


def export_pc_to_register(point_cloud_file: str, pokes_to_data):
    os.makedirs(os.path.dirname(point_cloud_file), exist_ok=True)
    with open(point_cloud_file, 'w') as f:
        for poke, data in pokes_to_data.items():
            # write out the poke index and the size of the point cloud
            total_pts = len(data['free']) + len(data['contact'])
            f.write(f"{poke} {total_pts}\n")
            serialization.export_pcs(f, data['free'], data['contact'])


def extract_known_points(task, vis: typing.Optional[DebugRvizDrawer] = None,
                         clean_cache=False):
    p.connect(p.DIRECT)
    device = 'cpu'
    data_path = os.path.join(cfg.DATA_DIR, poke_real.DIR)
    dataset = poke_real.PokeBubbleDataset(data_name=data_path, load_shear=False)

    # values for each trajectory
    pokes_to_data = {}
    env: typing.Optional[poke_real_nonros.PokeRealNoRosEnv] = None
    cur_seed = None
    cur_poke = None
    contact_pts = []

    def end_current_trajectory(new_seed):
        nonlocal cur_seed, env, contact_pts, pokes_to_data
        logger.info(f"{task.name} finished seed {cur_seed} processing {new_seed}")
        # ending an "empty" first trajectory
        if cur_seed is None:
            pass
        else:
            point_cloud_file = f"{registration_nopytorch3d.saved_traj_dir_base(task, experiment_name=experiment_name)}_{cur_seed}.txt"
            export_pc_to_register(point_cloud_file, pokes_to_data)

            pc_register_against_file = f"{registration_nopytorch3d.saved_traj_dir_base(task, experiment_name=experiment_name)}.txt"
            if not os.path.exists(pc_register_against_file) or clean_cache:
                surface_thresh = 0.002
                serialization.export_pc_register_against(pc_register_against_file, env.target_sdf,
                                                         surface_thresh=surface_thresh)
                if vis is not None:
                    pc_surface = env.target_sdf.get_filtered_points(
                        lambda voxel_sdf: (voxel_sdf < surface_thresh) & (voxel_sdf > -surface_thresh))
                    vis.draw_points(f"surface.0", pc_surface, color=(1, 1, 1), scale=0.1)

        cur_seed = seed
        env = poke_real_nonros.PokeRealNoRosEnv(task, device=device)
        contact_pts = []
        pokes_to_data = {}

    def end_current_poke(new_poke):
        nonlocal cur_poke
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
        cur_poke = new_poke

    for i, sample in enumerate(dataset):
        if poke_real_nonros.Levels[sample['task']] != task:
            continue
        seed = sample['seed']
        poke = sample['poke']

        # start of new poke
        if cur_poke is None or cur_poke != poke:
            end_current_poke(poke)
        # start of new trajectory
        if cur_seed is None or cur_seed != seed:
            end_current_trajectory(seed)

        bubbles = dataset.get_bubble_info(i)
        # complete the gripper from the points since there are gaps in between
        pts_free, pts_to_set = free_points_from_gripper(bubbles, sample)
        env.free_voxels[torch.from_numpy(pts_to_set)] = 1

        contact = contact_points_from_gripper(bubbles, sample)
        contact_pts += contact

        c_pts = np.concatenate(contact_pts)
        f_pts, _ = env.free_voxels.get_known_pos_and_values()
        logger.info(f"Poke {poke} with {len(c_pts)} contact points {len(f_pts)} free points")
        pokes_to_data[poke] = {'free': f_pts, 'contact': c_pts}

        if vis is not None:
            vis.draw_points(f"known.0", pts_free, color=(0, 1, 1), scale=0.1)
            vis.draw_points(f"free.0", f_pts, color=(1, 0, 1), scale=0.2)
            vis.draw_points(f"contact.0", c_pts, color=(1, 0, 0), scale=0.5)

    # last poke of the last sample
    end_current_poke(None)
    end_current_trajectory(None)


# TODO plot plausible set
def set_approximate_pose(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer):
    pose = None
    # use breakpoints
    approx_pose_file = registration_nopytorch3d.approximate_pose_file(env.level, experiment_name=experiment_name)
    if os.path.exists(approx_pose_file):
        pose = serialization.import_pose(approx_pose_file)
        env.obj_factory.draw_mesh(vis, "hand_placed_obj", pose, (0, 0, 0, 0.5), object_id=0)

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
        env.obj_factory.draw_mesh(vis, "hand_placed_obj", pose, (0, 0, 0, 0.5), object_id=0)

    # rotation saved as xyzw
    serialization.export_pose(approx_pose_file, pose)


def plot_optimal_pose(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer, seed):
    optimal_pose_file = registration_nopytorch3d.optimal_pose_file(env.level, seed=seed,
                                                                   experiment_name=experiment_name)
    pose = serialization.import_pose(optimal_pose_file)
    logger.info(pose)
    env.obj_factory.draw_mesh(vis, "optimal_pose", pose, (0., 0.5, 0.5, 0.5), object_id=0)


def plot_plausible_set(env: poke_real_nonros.PokeRealNoRosEnv, vis: DebugRvizDrawer, seed):
    filename = f"{registration_nopytorch3d.saved_traj_dir_base(env.level, experiment_name=experiment_name)}_plausible_set_{seed}.pkl"
    plausible_set = torch.load(filename)
    last_poke = max(plausible_set.keys())
    plausible_transforms = plausible_set[last_poke]
    ns = "plausible_pose"
    vis.clear_visualization_after(ns, 0)
    for i, T in enumerate(plausible_transforms):
        pose = matrix_to_pos_rot(T)
        env.obj_factory.draw_mesh(vis, ns, pose, (0.5, 0.5, 0, 0.5), object_id=i)


def main(args):
    task = task_map[args.task]

    if args.no_gui:
        vis = None
    else:
        vis = DebugRvizDrawer(world_frame='world')

    if args.experiment == "extract-known-points":
        extract_known_points(task, vis=vis)
    elif args.experiment == "plot-sdf":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda", clean_cache=True)

        def filter(pts):
            c1 = (pts[:, 0] > -0.15) & (pts[:, 0] < 0.15)
            c2 = (pts[:, 1] > 0.) & (pts[:, 1] < 0.2)
            c3 = (pts[:, 2] > -0.2) & (pts[:, 2] < 0.4)
            c = c1 & c2 & c3
            return pts[c][::2]

        registration_nopytorch3d.plot_sdf(env.obj_factory, env.target_sdf, vis, filter_pts=filter)

    elif args.experiment == "set-approximate-pose":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        set_approximate_pose(env, vis)
    elif args.experiment == "plot-optimal-pose":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        plot_optimal_pose(env, vis, args.seed)
    elif args.experiment == "plot-plausible-set":
        env = poke_real_nonros.PokeRealNoRosEnv(task, device="cuda")
        plot_plausible_set(env, vis, args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pokes from a real robot')
    parser.add_argument('--experiment',
                        choices=['extract-known-points', 'plot-sdf', 'set-approximate-pose', 'plot-optimal-pose',
                                 'plot-plausible-set'],
                        default='extract-known-points',
                        help='which experiment to run')
    parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
    task_map = {level.name.lower(): level for level in poke_real_nonros.Levels}
    parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
    parser.add_argument('--no_gui', action='store_true', help='force no GUI')
    parser.add_argument('--seed', type=int, default=0, help='random seed to process')

    args = parser.parse_args()
    main(args)
