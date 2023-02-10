import copy
import os
import rospy
import ros_numpy
from threading import Lock

from scipy.ndimage import uniform_filter
import numpy as np
# for converting strings to dictionary
from numpy import array
from torch import tensor

from arc_utilities.listener import Listener
from mmint_camera_utils.camera_utils.camera_parsers import RealSenseCameraParser
from mmint_camera_utils.recording_utils.data_recording_wrappers import DictSelfSavedWrapper
from mmint_camera_utils.ros_utils.utils import pose_to_matrix

from bubble_utils.bubble_envs import BubbleBaseEnv, MedBaseEnv
from pytorch_kinematics import transforms as tf

from geometry_msgs.msg import Pose

import logging
import torch

from arm_pytorch_utilities import tensor_utils
from sensor_msgs.msg import Image
from stucco import cfg
from base_experiments.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource
from stucco.detection import ContactDetector
from stucco.env.arm_real import BubbleCameraContactSensor, RealArmEnv
from stucco.env.poke_real_nonros import Levels

from victor_hardware_interface_msgs.msg import MotionStatus
from bubble_utils.bubble_med.bubble_med import BubbleMed
from bubble_utils.bubble_parsers.bubble_parser import BubbleParser
from mmint_camera_utils.camera_utils.camera_utils import bilinear_interpolate, project_depth_points, project_depth_image
from bubble_utils.bubble_datasets.base_datasets.bubble_dataset_base import BubbleDatasetBase
from bubble_utils.bubble_tools.bubble_img_tools import process_bubble_img
from mmint_camera_utils.recording_utils.recording_utils import process_image

import gym.spaces

logger = logging.getLogger(__name__)

DIR = "poke_real"


def xy_pose_distance(a: Pose, b: Pose):
    pos_distance = np.linalg.norm(ros_numpy.numpify(a.position) - ros_numpy.numpify(b.position))
    return pos_distance


class RealPokeLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if self.config.predict_difference:
            y = RealPokeEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class PokeBubbleCameraContactSensor(BubbleCameraContactSensor):
    def _get_link_frame_deform_point(self, cache, camera):
        # TODO hill climbing on imprint to find local maxima imprints to extract those as contact points
        # NOT actually needed since I process the points from the bubbles offline afterwards
        # return the undeformed point to prevent penetration with the model
        depth_im = cache['ref_img']
        mask = cache['mask']
        # these are in optical/camera frame
        # average in image space, then project the average to point cloud
        vs, us, _ = np.nonzero(mask)
        v, u = vs.mean(), us.mean()
        # interpolation on the depth image to get depth value
        d = bilinear_interpolate(depth_im, u, v)

        pt = project_depth_points(u, v, d, self.K)
        pt_l = camera.transform_pc(np.array(pt).reshape(1, -1), camera.optical_frame['depth'], self.link_frame)
        cache['contact_avg_pixel'] = (v, u)
        cache['contact_pt_link_frame'] = pt_l
        return torch.tensor(pt_l[0], dtype=self.dtype, device=self.device)


class ContactDetectorPokeRealArmBubble(ContactDetector):
    """Contact detector using the bubble gripper"""

    def __init__(self, residual_precision,
                 camera_l: BubbleParser, camera_r: BubbleParser,
                 ee_link_frame: str, imprint_threshold=0.004, deform_number_threshold=20,
                 window_size=5, dtype=torch.float, device='cpu'):
        super().__init__(residual_precision, window_size=window_size, dtype=dtype, device=device)
        if camera_l is not None:
            self.register_contact_sensor(PokeBubbleCameraContactSensor(camera_l, ee_link_frame,
                                                                       imprint_threshold=imprint_threshold,
                                                                       deform_number_threshold=deform_number_threshold,
                                                                       dtype=dtype, device=device))
            self.register_contact_sensor(PokeBubbleCameraContactSensor(camera_r, ee_link_frame,
                                                                       imprint_threshold=imprint_threshold,
                                                                       deform_number_threshold=deform_number_threshold,
                                                                       dtype=dtype, device=device))

        else:
            rospy.logwarn("Creating contact detector without camera")

    def get_last_contact_location(self, pose=None, **kwargs):
        # call this first to process the caches
        ret = super().get_last_contact_location(pose=pose, **kwargs)
        # self.draw_deformation()
        return ret

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        pts = []

        for s in range(1, len(self.sensors)):
            pt_candidate = self.sensors[s].isolate_contact(ee_force_torque, pose, q=q,
                                                           visualizer=visualizer)
            if pt_candidate is not None:
                pts.append(pt_candidate)

        if pts is None or len(pts) == 0 or pts[0] is None:
            pts = None
        else:
            pts = torch.stack(pts)

        if visualizer is not None and pts is not None:
            xr = tf.Transform3d(device=self.device, dtype=self.dtype, pos=pose[0], rot=tf.xyzw_to_wxyz(pose[1]))
            pts_world = xr.transform_points(pts)
            for i, pt in enumerate(pts_world):
                visualizer.draw_point(f'most likely contact.{i}', pts[i], color=(0, 1, 0), scale=2)

        # immovable object
        dx = torch.zeros_like(pts)
        return pts, dx


class SingleSceneCamerasBaseEnv(MedBaseEnv):
    """
    This environment adds 1 scene camera with ID 1
    """

    def __init__(self, *args, record_scene_pcs=True, **kwargs):
        self.record_scene_pcs = record_scene_pcs
        self.scene_camera_indx = 1

        super().__init__(*args, **kwargs)

        self.scene_camera_parser = self._get_scene_camera_parser(camera_indx=self.scene_camera_indx)
        self.scene_camera_name = self.scene_camera_parser.camera_name

    @classmethod
    def get_name(cls):
        return 'single_scene_cameras'

    def _get_scene_camera_parser(self, camera_indx):
        scene_camera_parser = RealSenseCameraParser(camera_indx=camera_indx,
                                                    scene_name=self.scene_name, save_path=self.save_path,
                                                    verbose=False, buffered=self.buffered,
                                                    wrap_data=self.wrap_data,
                                                    save_depth_img_as_numpy=True,
                                                    record_pointcloud=self.record_scene_pcs)
        return scene_camera_parser

    def _get_tf_frames(self):
        scene_camera_frames = self.scene_camera_parser.get_camera_frames()
        tf_frames = super()._get_tf_frames() + scene_camera_frames
        return tf_frames

    def _get_scene_observation(self):
        scene_observation = {}
        obs = self._get_scene_observation_from_camera_parser(self.scene_camera_parser)
        for k, v in obs.items():
            scene_observation['scene_{}'.format(k)] = v
        scene_observation['scene_depth_img'].data_params['save_as_numpy'] = True
        if self.wrap_data:
            scene_observation = DictSelfSavedWrapper(scene_observation)
        return scene_observation

    def _get_scene(self, ref_frame='med_base'):
        scene = self.scene_camera_parser.get_point_cloud(ref_frame=ref_frame, return_ref_frame=False)
        return scene

    def _get_scene_observation_from_camera_parser(self, camera_parser):
        obs = {}
        obs['camera_info_color'] = camera_parser.get_camera_info_color()
        obs['camera_info_depth'] = camera_parser.get_camera_info_depth()
        obs['color_img'] = camera_parser.get_image_color()
        obs['depth_img'] = camera_parser.get_image_depth()
        return obs


class RealPokeEnv(BubbleBaseEnv, SingleSceneCamerasBaseEnv, RealArmEnv):
    nu = 3
    nx = 3
    MAX_FORCE = 30
    MAX_GRIPPER_FORCE = 30
    MAX_PUSH_DIST = 0.05
    OPEN_ANGLE = 0.055
    CLOSE_ANGLE = 0.0

    # REST_JOINTS = [0.142, 0.453, -0.133, -1.985, 0.027, -0.875, 0.041]
    REST_JOINTS = [0.1141799424293767, 0.45077912971064055, -0.1378142121392566, -1.9944268727534853,
                   -0.013151296594681122, -0.8713510018630727, 0.06839882517621013]
    REST_POS = [0.530, 0, 0.433]
    REST_ORIENTATION = [0, np.pi / 2, 0]

    EE_LINK_NAME = "med_kuka_link_7"
    WORLD_FRAME = "med_base"
    ACTIVE_MOVING_ARM = 0  # only 1 arm

    # --- BubbleBaseEnv overrides
    @classmethod
    def get_name(cls):
        return 'real_poke_env'

    def _is_done(self, observation, a):
        return a is None or a['dxyz'] is None

    def _get_action_space(self):
        u_min, u_max = self.get_control_bounds()
        actions = gym.spaces.Box(u_min, u_max)
        return gym.spaces.Dict({'dxyz': actions})
        # u_names = self.control_names()
        # action_dict = {name: gym.spaces.Box(low=u_min[i], high=u_max[i]) for i, name in enumerate(u_names)}
        # return gym.spaces.Dict(action_dict)

    def _get_observation(self):
        obs = {}
        obs.update(self._get_bubble_observation())
        obs.update(self._get_scene_observation())
        obs['tfs'] = self._get_tfs()
        obs['xyz'] = self._observe_ee(return_z=True)
        obs['seed'] = self.seed
        return obs

    @staticmethod
    def state_names():
        return ['x ee (m)', 'y ee (m)', 'z ee (m)']

    @staticmethod
    def get_ee_pos(state):
        return state[:3]

    @staticmethod
    @tensor_utils.ensure_2d_input
    def get_ee_pos_states(states):
        return states[:, :3]

    @classmethod
    @tensor_utils.ensure_2d_input
    def get_state_ee_pos(cls, pos):
        raise NotImplementedError()

    @classmethod
    @handle_data_format_for_state_diff
    def state_difference(cls, state, other_state):
        """Get state - other_state in state space"""
        dpos = state[:, :3] - other_state[:, :3]
        return dpos

    @classmethod
    def state_cost(cls):
        return np.diag([1, 1, 1])

    @classmethod
    def state_distance(cls, state_difference):
        return state_difference[:, :3].norm(dim=1)

    @staticmethod
    def control_names():
        return ['d$x_r$', 'd$y_r$', 'd$z_r$']

    @staticmethod
    def get_control_bounds():
        u_min = np.array([-1, -1, -1])
        u_max = np.array([1, 1, 1])
        return u_min, u_max

    @classmethod
    @handle_data_format_for_state_diff
    def control_similarity(cls, u1, u2):
        return torch.cosine_similarity(u1, u2, dim=-1).clamp(0, 1)

    @classmethod
    def control_cost(cls):
        return np.diag([1 for _ in range(cls.nu)])

    def __init__(self, *args, seed=0, environment_level=0, vel=0.2, downsample_info=50, use_cameras=True, **kwargs):
        self.seed = seed
        level = Levels(environment_level)
        save_path = os.path.join(cfg.DATA_DIR, DIR, level.name)
        self.vel = vel
        BubbleBaseEnv.__init__(self, scene_name=level.name, save_path=save_path, record_bubble_pcs=True,
                               bubble_left=use_cameras, bubble_right=use_cameras, wrap_data=True)
        RealArmEnv.__init__(self, *args, environment_level=level, vel=vel, **kwargs)
        SingleSceneCamerasBaseEnv.__init__(self, scene_name=level.name, save_path=save_path, wrap_data=True,
                                           record_scene_pcs=False)
        # force an orientation; if None, uses the rest orientation
        self.orientation = None
        # prevent saving gigantic arrays to csv
        self.downsample_info = downsample_info
        # listen for imprints (easier than working with the data directly...)
        imprint_names = [
            '/{}/imprint_filtered'.format(self.camera_parser_left.camera_name),
            '/{}/imprint_filtered'.format(self.camera_parser_right.camera_name)
        ]
        self.imprint_listeners = [
            Listener(topic_name=imprint, topic_type=Image, wait_for_data=False) for imprint in imprint_names
        ]

    def _get_med(self):
        med = BubbleMed(display_goals=False,
                        base_kwargs={'cartesian_impedance_controller_kwargs': {'pose_distance_fn': xy_pose_distance,
                                                                               'timeout_per_m': 40 / self.vel,
                                                                               'position_close_enough': 0.005}})
        return med

    def enter_cartesian_mode(self):
        self.robot.set_cartesian_impedance(self.vel, x_stiffness=1000, y_stiffness=5000, z_stiffnes=5000)

    def reset(self):
        # self.return_to_rest(self.robot.arm_group)
        self.state = self._obs()
        self.contact_detector.clear()
        return np.copy(self.state), None

    def _setup_robot_ros(self, residual_threshold=10., residual_precision=None):
        self._need_to_force_planar = False
        # adjust timeout according to velocity (at vel = 0.1 we expect 400s per 1m)
        self.robot = self.med
        self.vis.init_ros(world_frame="world")

        self.motion_status_input_lock = Lock()
        self._temp_wrenches = []
        # subscribe to status messages
        self.contact_ros_listener = rospy.Subscriber(self.robot.ns("motion_status"), MotionStatus,
                                                     self.contact_listener)

        # reset to rest position
        self.return_to_rest(self.robot.arm_group)

        self._calibrate_bubbles(open_before_calib=False)
        self.robot.gripper.move(0)

        self.last_ee_pos = self._observe_ee(return_z=True)
        self.REST_POS[2] = self.last_ee_pos[-1]
        self.state = self._obs()

        if residual_precision is None:
            residual_precision = np.diag([1, 1, 0, 1, 1, 1])

        self._contact_detector = ContactDetectorPokeRealArmBubble(residual_precision,
                                                                  self.camera_parser_left, self.camera_parser_right,
                                                                  self.EE_LINK_NAME)
        # parallel visualizer for ROS and pybullet
        self.vis.sim = None

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = self._observe_ee(return_z=True)
        return state

    # --- control (commonly overridden)
    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        dz = action[2] * self.MAX_PUSH_DIST
        return dx, dy, dz

    def _stop_push(self):
        imprint_threshold = 0.004
        imprint_num_threshold = 6
        for parser in [self.camera_parser_right, self.camera_parser_left]:
            imprint = self._get_imprint(parser)
            violating = imprint > imprint_threshold
            if violating.sum() >= imprint_num_threshold:
                return True
        return False

    def _get_imprint(self, camera_parser):
        # camera_parser = self.camera_parser_right
        depth_img = camera_parser.get_image_depth()
        def_img = process_bubble_img(depth_img)
        side = camera_parser.camera_name.split('_')[-1]
        ref_img = process_bubble_img(self.ref_obs[f'bubble_depth_img_{side}'])
        imprint = ref_img - def_img
        imprint = filter_depth_map(imprint, 20)
        return imprint

    def _do_action(self, action):
        self._clear_state_before_step()
        action = action['dxyz']
        success = True
        if action is not None:
            action = np.clip(action, *self.get_control_bounds())
            # normalize action such that the input can be within a fixed range
            self.last_ee_pos = self._observe_ee(return_z=True)
            dx, dy, dz = self._unpack_action(action)

            self._single_step_contact_info = {}
            orientation = self.orientation
            if orientation is None:
                orientation = copy.deepcopy(self.REST_ORIENTATION)
                if self._need_to_force_planar:
                    orientation[1] += np.pi / 4

            self.robot.move_delta_cartesian_impedance(self.ACTIVE_MOVING_ARM, dx=dx, dy=dy,
                                                      target_z=self.last_ee_pos[2] + dz,
                                                      stop_on_force_threshold=10,
                                                      stop_callback=self._stop_push,
                                                      blocking=True, step_size=0.025,
                                                      target_orientation=orientation)
            s = self.robot.cartesian.status
            if s.reached_joint_limit or s.reached_force_threshold or s.callback_stopped:
                success = False

        full_info = self.aggregate_info()
        info = {}
        for key, value in full_info.items():
            if not isinstance(value, np.ndarray) or len(value.shape) == 1:
                info[key] = value
            else:
                info[key] = value[::self.downsample_info]
        info['success'] = success

        return info


def filter_depth_map(depth_map, size=10):
    depth_map = depth_map.squeeze(-1)
    filtered_depth_map = uniform_filter(depth_map, size=size, mode='reflect')
    return filtered_depth_map


class RealPokeDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {RealPokeEnv: RealPokeLoader, }
        return loader_map.get(env_type, None)


class PokeBubbleDataset(BubbleDatasetBase):
    def __init__(self, *args, **kwargs):
        self.min_trajectory_index_for_reference = {}
        super().__init__(*args, **kwargs)

    def get_name(self):
        return 'poke_bubble_dataset'

    def _get_common_info(self, sample_code):
        dl_line = self.dl.iloc[sample_code]  # read the line from the datalegend with that
        # e.g. DRILL, DRILL_SLANTED
        task = dl_line['Scene']
        init_fc = int(dl_line['InitialStateFC'])
        final_fc = int(dl_line['FinalStateFC'])
        ref_fc = int(dl_line['ReferenceStateFC'])
        return task, init_fc, final_fc, ref_fc

    def _get_sample(self, sample_code):
        """
        Retruns the sample corresponding to the sample code sample_code
        :param sample_code:
        :return:
        """
        dl_line = self.dl.iloc[sample_code]  # read the line from the datalegend with that
        task, init_fc, final_fc, ref_fc = self._get_common_info(sample_code)
        info = eval(dl_line['Info'])
        seed = int(dl_line['seed_init'])
        poke = dl_line['TrajectoryIndex']
        # keeps incrementing so we need to keep track of the minimum trajectory index and take the offset
        min_ti = self.min_trajectory_index_for_reference.get(ref_fc, 1000000)
        self.min_trajectory_index_for_reference[ref_fc] = min(poke, min_ti)
        poke = poke - self.min_trajectory_index_for_reference[ref_fc] + 1

        # get z axis of wrist in world frame since we'll need to extend the point clouds along it to generate free space
        pose = self._get_tfs(final_fc, task, frame_id=f'med_kuka_link_ee', ref_id='med_base')[0]
        T = pose_to_matrix(pose)
        z_axis = T[:3, :3] @ np.array([0, 0, 1])
        sample = {
            'sample_code': sample_code,
            'task': task,
            'info': info,
            'seed': seed,
            'poke': poke,
            'wrist_z': z_axis,
        }
        return sample

    def get_bubble_info(self, sample_code):
        """More expensive bubble information to retrieve when it has relevant common info"""
        task, init_fc, final_fc, ref_fc = self._get_common_info(sample_code)
        sample = {}
        for camera in ['left', 'right']:
            data = self._get_bubble_sample_data(init_fc, final_fc, ref_fc, task, camera_name=camera)
            sample[camera] = data

            pose = self._get_tfs(final_fc, task, frame_id=f'pico_flexx_{camera}_optical_frame', ref_id='med_base')[0]
            T = pose_to_matrix(pose)
            # project depth images to world frame
            pts = project_depth_image(data['depth_final'].squeeze(-1), data['camera_info_depth']['K'], usvs=None)
            pts_homogenous = np.concatenate((pts, np.ones(pts.shape[:-1] + (1,))), axis=-1)
            pts_world = pts_homogenous.reshape(-1, 4)
            pts_world = T @ pts_world.transpose()
            pts_world = pts_world.transpose().reshape(*pts.shape[:-1], 4)
            pts_world = pts_world[..., :3]  # strip the 1 at the end
            sample[camera][f"pts_world"] = pts_world
            # can use despite it being points
            sample[camera][f"pts_world_filtered"] = process_bubble_img(pts_world)
        return sample

    def get_scene_info(self, sample_code, camera_idx=1):
        """More expensive bubble information to retrieve when it has relevant common info"""
        task, init_fc, final_fc, ref_fc = self._get_common_info(sample_code)
        camera_tfs = self._get_tfs(final_fc, task, frame_id=f'camera_{camera_idx}_depth_optical_frame',
                                   ref_id='med_base')
        sample = {
            'depth': self._load_scene_depth_img(final_fc, task, camera_idx),
            'K': self._load_scene_camera_info_depth(task, camera_idx, final_fc)['K'],
            'camera_tf': camera_tfs[0]
        }
        return sample
