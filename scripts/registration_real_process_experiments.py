import os
import typing
from datetime import datetime
import argparse
from stucco.env import poke_real
from stucco import cfg
from stucco.env.real_env import DebugRvizDrawer
import numpy as np
import torch
import logging
from stucco import voxel
import scipy

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


def extract_known_points(task, freespace_voxel_resolution=0.01, vis: typing.Optional[DebugRvizDrawer] = None):
    device = 'cpu'
    data_path = os.path.join(cfg.DATA_DIR, poke_real.DIR)
    dataset = poke_real.PokeBubbleDataset(data_name=data_path, load_shear=False)

    default_freespace_range = np.array([[0.7, 0.8], [-0.1, 0.1], [0.39, 0.45]])

    # values for each trajectory
    free_voxels = None
    cur_seed = None
    contact_pts = []

    for i, sample in enumerate(dataset):
        if poke_real.Levels[sample['task']] != task:
            continue
        seed = sample['seed']
        poke = sample['poke']
        # start of new trajectory
        if cur_seed is None or cur_seed != seed:
            logger.info(f"Process new {task.name} trajectory seed {seed}")
            cur_seed = seed
            free_voxels = voxel.ExpandingVoxelGrid(freespace_voxel_resolution, default_freespace_range, device=device)
            contact_pts = []

        bubbles = dataset.get_bubble_info(i)
        # complete the gripper from the points since there are gaps in between
        pts_free, pts_to_set = free_points_from_gripper(bubbles, sample)
        free_voxels[torch.from_numpy(pts_to_set)] = 1

        contact = contact_points_from_gripper(bubbles, sample)
        contact_pts += contact

        # TODO write to file
        c_pts = np.concatenate(contact_pts)
        f_pts, _ = free_voxels.get_known_pos_and_values()
        logger.info(f"Poke {poke} with {len(c_pts)} contact points {len(f_pts)} free points")

        if vis is not None:
            vis.draw_points(f"known.0", pts_free, color=(0, 1, 1), scale=0.1)
            vis.draw_points(f"free.0", f_pts, color=(1, 0, 1), scale=0.2)
            vis.draw_points(f"contact.0", c_pts, color=(1, 0, 0), scale=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pokes from a real robot')
    parser.add_argument('--experiment',
                        choices=['extract-known-points'],
                        default='extract-known-points',
                        help='which experiment to run')
    parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')
    task_map = {level.name.lower(): level for level in poke_real.Levels}
    parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
    parser.add_argument('--no_gui', action='store_true', help='force no GUI')

    args = parser.parse_args()
    task = task_map[args.task]

    if args.no_gui:
        vis = None
    else:
        vis = DebugRvizDrawer(world_frame='world')

    if args.experiment == "extract-known-points":
        extract_known_points(task, vis=vis)
