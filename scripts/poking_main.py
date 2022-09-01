import argparse
import copy
import time
import math
import torch
import pybullet as p
import numpy as np
import logging
import os
from datetime import datetime

from sklearn.cluster import Birch, DBSCAN, KMeans
from window_recorder.recorder import WindowRecorder

from stucco.baselines.cluster import OnlineAgglomorativeClustering, OnlineSklearnFixedClusters
from stucco.defines import NO_CONTACT_ID
from stucco.evaluation import compute_contact_error, clustering_metrics, evaluate_chamfer_distance
from stucco.env.env import InfoKeys

from arm_pytorch_utilities import rand, tensor_utils, math_utils

from stucco import cfg
from stucco import icp, tracking, exploration
from stucco.env import poke
from stucco.env_getters.poke import PokeGetter
from pytorch_kinematics import transforms as tf
from stucco.icp import costs as icp_costs
from stucco import util

from stucco.retrieval_controller import rot_2d_mat_to_angle, \
    sample_model_points, pose_error, TrackingMethod, OurSoftTrackingMethod, \
    SklearnTrackingMethod, KeyboardController, PHDFilterTrackingMethod, OurSoftTrackingWithRummagingMethod

ch = logging.StreamHandler()
fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, "logs", "{}.log".format(datetime.now())))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S', handlers=[ch, fh])

logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger(__name__)


def run_poke(env: poke.PokeEnv, method: TrackingMethod, reg_method, name="", seed=0, clean_cache=False,
             register_num_points=500,
             eval_num_points=200, ctrl_noise_max=0.005):
    name = reg_method + name
    # [name][seed] to access
    # chamfer_err: T x B number of steps by batch chamfer error
    fullname = os.path.join(cfg.DATA_DIR, f'poking.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache or clean_cache:
            cache[name] = {}
        if seed not in cache[name] or clean_cache:
            cache[name][seed] = {}
    else:
        cache = {name: {seed: {}}}

    predetermined_control = {}

    ctrl = [[0., 0., 1]] * 10
    # ctrl += [[0.4, 0.4], [.5, -1]] * 6
    # ctrl += [[-0.2, 1]] * 4
    # ctrl += [[0.3, -0.3], [0.4, 1]] * 4
    # ctrl += [[1., -1]] * 3
    # ctrl += [[1., 0.6], [-0.7, 0.5]] * 4
    # ctrl += [[0., 1]] * 5
    # ctrl += [[1., 0]] * 4
    # ctrl += [[0.4, -1.], [0.4, 0.5]] * 4
    rand.seed(0)
    # noise = (np.random.rand(len(ctrl), 2) - 0.5) * 0.5
    # ctrl = np.add(ctrl, noise)
    predetermined_control[poke.Levels.MUSTARD] = ctrl

    ctrl = method.create_controller(predetermined_control[env.level])

    obs = env.reset()

    model_name = env.target_model_name
    # sample_in_order = env.level in [poke.Levels.COFFEE_CAN]
    # get a fixed number of model points to evaluate against (this will be independent on points used to register)
    model_points_eval, model_normals_eval, _ = sample_model_points(num_points=eval_num_points, name=model_name, seed=0,
                                                                   device=env.device)
    device, dtype = model_points_eval.device, model_points_eval.dtype

    # get a large number of model points to register to
    model_points_register, model_normals_register, _ = sample_model_points(num_points=register_num_points,
                                                                           name=model_name, seed=0, device=env.device)

    pose = p.getBasePositionAndOrientation(env.target_object_id)
    link_to_current_tf_gt = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
        tensor_utils.ensure_tensor(device, dtype, pose[1])), dtype=dtype, device=device)
    model_points_world_frame_eval = link_to_current_tf_gt.transform_points(model_points_eval)
    model_normals_world_frame_eval = link_to_current_tf_gt.transform_points(model_normals_eval)

    info = None
    simTime = 0

    B = 30
    device = env.device
    best_tsf_guess = exploration.random_upright_transforms(B, dtype, device)
    best_T = None
    guess_pose = None
    chamfer_err = []

    pt_to_config = poke.ArmPointToConfig(env)

    contact_id = []

    # placeholder for now
    empty_sdf = util.VoxelSet(torch.empty(0), torch.empty(0))
    volumetric_cost = icp_costs.VolumetricCost(env.free_voxels, empty_sdf, env.target_sdf, scale=1,
                                               scale_known_freespace=20 if reg_method == 'volumetric-freespace' else 0,
                                               vis=env.vis, debug=False)

    rand.seed(seed)
    while not ctrl.done():
        best_distance = None
        simTime += 1
        env.draw_user_text("{}".format(simTime), xy=(0.5, 0.7, -1))

        action = ctrl.command(obs, info)
        method.visualize_contact_points(env)

        if env.contact_detector.in_contact():
            contact_id.append(info[InfoKeys.CONTACT_ID])
        else:
            contact_id.append(NO_CONTACT_ID)

        # note that we update our registration regardless if we're in contact or not
        all_configs = torch.tensor(np.array(ctrl.x_history), dtype=dtype, device=device).view(-1, env.nx)
        dist_per_est_obj = []
        transforms_per_object = []
        rmse_per_object = []
        best_segment_idx = None
        k = -1
        for k, this_pts in enumerate(method):
            # this_pts corresponds to tracked contact points that are segmented together
            this_pts = tensor_utils.ensure_tensor(device, dtype, this_pts)
            volumetric_cost.sdf_voxels = util.VoxelSet(this_pts,
                                                       torch.zeros(this_pts.shape[0], dtype=dtype, device=device))

            # TODO verify by printing this_pts about what it actually is
            if reg_method in ["volumetric", "volumetric-freespace"]:
                T, distances = icp.icp_volumetric(volumetric_cost, this_pts, given_init_pose=best_tsf_guess,
                                                  batch=B, max_iterations=20, lr=0.01)
            elif reg_method == "icp":
                T, distances = icp.icp_pytorch3d(this_pts, model_points_register, given_init_pose=best_tsf_guess,
                                                 batch=B)
            elif reg_method == "icp-sgd":
                T, distances = icp.icp_pytorch3d_sgd(this_pts, model_points_register,
                                                     given_init_pose=best_tsf_guess, batch=B, learn_translation=True,
                                                     use_matching_loss=True)
            else:
                raise RuntimeError("Unrecognized registration method " + reg_method)

            transforms_per_object.append(T)
            T = T.inverse()
            score = distances
            best_tsf_index = np.argmin(score)

            # pick object with lowest variance in its translation estimate
            translations = T[:, :2, 2]
            best_tsf_distances = (translations.var(dim=0).sum()).item()

            dist_per_est_obj.append(best_tsf_distances)
            rmse_per_object.append(distances)
            if best_distance is None or best_tsf_distances < best_distance:
                best_distance = best_tsf_distances
                best_tsf_guess = T[best_tsf_index].inverse()
                best_segment_idx = k

        # has at least one contact segment
        if k != -1:
            method.register_transforms(transforms_per_object[best_segment_idx], best_tsf_guess)
            logger.debug(f"err each obj {np.round(dist_per_est_obj, 4)}")
            best_T = best_tsf_guess.inverse()

            # evaluate with chamfer distance
            errors_per_batch = evaluate_chamfer_distance(transforms_per_object[best_segment_idx],
                                                         model_points_world_frame_eval, env.vis, env.testObjId,
                                                         rmse_per_object[best_segment_idx], 0)
            # errors.append(np.mean(errors_per_batch))
            chamfer_err.append(errors_per_batch)
            logger.debug(f"chamfer distance {simTime}: {np.mean(errors_per_batch)}")

            # draw mesh at where our best guess is
            # TODO check if we need to invert this best guess
            guess_pose = util.matrix_to_pos_rot(best_T)
            env.draw_mesh("base_object", guess_pose, (0.0, 1.0, 0., 0.5))
            # TODO save current pose and contact point for playback

        if torch.is_tensor(action):
            action = action.cpu()

        action = np.array(action).flatten()
        obs, rew, done, info = env.step(action)

        if len(chamfer_err) > 0:
            cache[name][seed] = {'chamfer_err': np.stack(chamfer_err), }
            torch.save(cache, fullname)

    # evaluate FMI and contact error here
    labels, moved_points = method.get_labelled_moved_points(np.ones(len(contact_id)) * NO_CONTACT_ID)
    contact_id = np.array(contact_id)

    in_label_contact = contact_id != NO_CONTACT_ID

    m = clustering_metrics(contact_id[in_label_contact], labels[in_label_contact])
    contact_error = compute_contact_error(None, moved_points, env=env, visualize=False)
    cme = np.mean(np.abs(contact_error))

    # grasp_at_pose(env, guess_pose)

    return m, cme


def grasp_at_pose(env, pose):
    # object is symmetric so pose can be off by 180
    yaw = pose[2]
    grasp_offset = [0, 0]
    # if env.level == Levels.FLAT_BOX:
    #     grasp_offset = [0., -0.25]
    #     if yaw > np.pi / 2:
    #         yaw -= np.pi
    #     elif yaw < -np.pi / 2:
    #         yaw += np.pi
    # elif env.level == Levels.BEHIND_CAN or env.level == Levels.IN_BETWEEN:
    #     grasp_offset = [0., -0.25]
    #     if yaw > 0:
    #         yaw -= np.pi
    #     elif yaw < -np.pi:
    #         yaw += np.pi
    # elif env.level == Levels.TOMATO_CAN:
    #     grasp_offset = [0, -0.2]
    #     # cylinder so doesn't matter what yaw we come in at
    #     yaw = -np.pi / 2
    # else:
    #     raise RuntimeError(f"No data for level {env.level}")

    grasp_offset = math_utils.rotate_wrt_origin(grasp_offset, yaw)
    target_pos = [pose[0] + grasp_offset[0], pose[1] + grasp_offset[1]]
    z = env._observe_ee(return_z=True)[-1]
    env.vis.draw_point("pre_grasp", [target_pos[0], target_pos[1], z], color=(1, 0, 0))
    # get to target pos
    obs = env._obs()
    diff = np.subtract(target_pos, obs)
    start = time.time()
    while np.linalg.norm(diff) > 0.01 and time.time() - start < 5:
        obs, _, _, _ = env.step(diff / env.MAX_PUSH_DIST)
        diff = np.subtract(target_pos, obs)
    # rotate in place
    prev_ee_orientation = copy.deepcopy(env.endEffectorOrientation)
    env.endEffectorOrientation = p.getQuaternionFromEuler([0, np.pi / 2, yaw + np.pi / 2])
    env.sim_step_wait = 0.01
    env.step([0, 0])
    env.open_gripper()
    env.step([0, 0])
    env.sim_step_wait = None
    # go for the grasp

    move_times = 4
    move_dir = -np.array(grasp_offset)
    while move_times > 0:
        act_mag = move_times if move_times <= 1 else 1
        move_times -= 1
        u = move_dir / np.linalg.norm(move_dir) * act_mag
        obs, _, _, _ = env.step(u)
    env.sim_step_wait = 0.01
    env.close_gripper()
    env.step([0, 0])
    env.sim_step_wait = None

    env.endEffectorOrientation = prev_ee_orientation


def main(env, method_name, registration_method, seed=0, name=""):
    methods_to_run = {
        'ours': OurSoftTrackingMethod(env, PokeGetter.contact_parameters(env), poke.ArmPointToConfig(env)),
        'online-birch': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, Birch, n_clusters=None,
                                              inertia_ratio=0.2,
                                              threshold=0.08),
        'online-dbscan': SklearnTrackingMethod(env, OnlineAgglomorativeClustering, DBSCAN, eps=0.05, min_samples=1),
        'online-kmeans': SklearnTrackingMethod(env, OnlineSklearnFixedClusters, KMeans, inertia_ratio=0.2, n_clusters=1,
                                               random_state=0),
        'gmphd': PHDFilterTrackingMethod(env, fp_fn_bias=4, q_mag=0.00005, r_mag=0.00005, birth=0.001, detection=0.3)
    }
    env.draw_user_text(f"{method_name} seed {seed}", xy=[-0.1, 0.28, -0.5])
    return run_poke(env, methods_to_run[method_name], registration_method, seed=seed, name=name)


def build_model(env: poke.PokeEnv, seed, num_points, pause_at_end=False, device="cpu"):
    vis = env._dd
    model_name = env.obj_factory.name
    points, normals, _ = sample_model_points(env.target_object_id, reject_too_close=0.006,
                                             num_points=num_points,
                                             force_z=None,
                                             mid_z=0.05,
                                             seed=seed, clean_cache=True,
                                             random_sample_sigma=0.2,
                                             name=env.obj_factory.name, vis=None,
                                             device=device)

    for i, pt in enumerate(points):
        vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
        vis.draw_2d_line(f"mn.{i}", pt, -normals[i], color=(0, 0, 0), size=2., scale=0.03)

    print(f"finished building {model_name} {seed} {num_points}")
    if pause_at_end:
        input("paused for inspection")
    vis.clear_visualizations()


parser = argparse.ArgumentParser(description='Object registration from contact')
parser.add_argument('registration',
                    choices=['volumetric', 'volumetric-freespace', 'icp', 'icp-sgd'],
                    help='which registration method to run')
parser.add_argument('--method',
                    choices=['ours', 'online-birch', 'online-dbscan', 'online-kmeans', 'gmphd'],
                    default='ours',
                    help='which tracking method to run')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0],
                    help='random seed(s) to run')
parser.add_argument('--no_gui', action='store_true', help='force no GUI')
# run parameters
task_map = {"mustard": poke.Levels.MUSTARD, "coffee": poke.Levels.COFFEE_CAN, "cracker": poke.Levels.CRACKER}
parser.add_argument('--task', default="mustard", choices=task_map.keys(), help='what task to run')
parser.add_argument('--name', default="", help='additional name for the experiment (concatenated with method)')

args = parser.parse_args()

if __name__ == "__main__":
    level = task_map[args.task]
    method_name = args.method
    registration_method = args.registration

    env = PokeGetter.env(level=level, mode=p.DIRECT if args.no_gui else p.GUI, clean_cache=False)

    # build sample points for this environment level
    # for num_points in (200, 500):
    #     for seed in range(10):
    #         build_model(env, seed=seed, num_points=num_points, pause_at_end=False)

    fmis = []
    cmes = []
    # backup video logging in case ffmpeg and nvidia driver are not compatible
    # with WindowRecorder(window_names=("Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build",),
    #                     name_suffix="sim", frame_rate=30.0, save_dir=cfg.VIDEO_DIR):
    for seed in args.seed:
        m, cme = main(env, method_name, registration_method, seed=seed, name=args.name)
        fmi = m[0]
        fmis.append(fmi)
        cmes.append(cme)
        logger.info(f"{method_name} fmi {fmi} cme {cme}")
        env.vis.clear_visualizations()
        env.reset()

    logger.info(
        f"{method_name} mean fmi {np.mean(fmis)} median fmi {np.median(fmis)} std fmi {np.std(fmis)} {fmis}\n"
        f"mean cme {np.mean(cmes)} median cme {np.median(cmes)} std cme {np.std(cmes)} {cmes}")
    env.close()
