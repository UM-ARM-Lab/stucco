import logging
import math
import pybullet as p
import time
import enum
import torch
import os
import random
import scipy.stats
import copy

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

from arm_pytorch_utilities import tensor_utils

import stucco.sdf
from stucco.env.pybullet_env import PybulletEnv, get_total_contact_force, make_box, state_action_color_pairs, \
    ContactInfo, make_cylinder, closest_point_on_surface
from stucco.env.env import TrajectoryLoader, handle_data_format_for_state_diff, EnvDataSource, InfoKeys, \
    PlanarPointToConfig
from stucco.env.panda import PandaJustGripperID
from stucco.env.pybullet_sim import PybulletSim
from stucco import cfg
from stucco import tracking
from stucco.defines import NO_CONTACT_ID
from stucco.detection import ContactDetector, ContactSensor
from stucco.sdf import ObjectFactory
from stucco import util

import pytorch_kinematics.transforms as tf

logger = logging.getLogger(__name__)

DIR = "poke"

kukaEndEffectorIndex = 6
pandaNumDofs = 7


class Levels(enum.IntEnum):
    # no clutter environments
    MUSTARD = 0
    CRACKER = 1
    COFFEE_CAN = 2
    BANANA = 3
    DRILL = 4
    HAMMER = 5
    DRILL_OPPOSITE = 6
    DRILL_SLANTED = 7
    DRILL_FALLEN = 8
    MUSTARD_SIDEWAYS = 9
    MUSTARD_FALLEN = 10
    MUSTARD_FALLEN_SIDEWAYS = 11
    HAMMER_1 = 12
    HAMMER_2 = 13


task_map = {str(c).split('.')[1]: c for c in Levels}
level_to_obj_map = {
    Levels.MUSTARD: "mustard",
    Levels.BANANA: "banana",
    Levels.DRILL: "drill",
    Levels.HAMMER: "hammer",
    Levels.HAMMER_1: "hammer",
    Levels.HAMMER_2: "hammer",
    Levels.DRILL_OPPOSITE: "drill",
    Levels.DRILL_SLANTED: "drill",
    Levels.DRILL_FALLEN: "drill",
    Levels.MUSTARD_SIDEWAYS: "mustard",
    Levels.MUSTARD_FALLEN: "mustard",
    Levels.MUSTARD_FALLEN_SIDEWAYS: "mustard",
}

DEFAULT_MOVABLE_RGBA = [0.8, 0.7, 0.3, 0.8]


class PokeLoader(TrajectoryLoader):
    @staticmethod
    def _info_names():
        return []

    def _process_file_raw_data(self, d):
        x = d['X']
        if self.config.predict_difference:
            y = PokeEnv.state_difference(x[1:], x[:-1])
        else:
            raise RuntimeError("Too hard to predict discontinuous normalized angles; use predict difference")

        xu, y, cc = self._apply_masks(d, x, y)

        return xu, y, cc


class DebugVisualization(enum.IntEnum):
    STATE = 2
    ACTION = 3
    REACTION_MINI_STEP = 4
    REACTION_IN_STATE = 5
    GOAL = 6
    INIT = 7
    FREE_VOXELS = 8
    STATE_TRANSITION = 9


class ReactionForceStrategy(enum.IntEnum):
    MAX_OVER_CONTROL_STEP = 0
    MAX_OVER_MINI_STEPS = 1
    AVG_OVER_MINI_STEPS = 2
    MEDIAN_OVER_MINI_STEPS = 3


class PybulletOracleContactSensor(ContactSensor):
    def __init__(self, robot_id, target_id, **kwargs):
        super(PybulletOracleContactSensor, self).__init__(**kwargs)
        self.robot_id = robot_id
        self.target_id = target_id
        self._cached_contact = None

    def observe_residual(self, residual):
        c = p.getContactPoints(self.robot_id, self.target_id())
        if len(c):
            self.in_contact = True
            self._cached_contact = c
        else:
            self.in_contact = False

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        if self._cached_contact is None:
            return None
        # assume only 1 contact
        pt = self._cached_contact[0][ContactInfo.POS_B]
        # caller expects it in link frame while we have it in global frame
        pt, pos, rot = tensor_utils.ensure_tensor(self.device, self.dtype, pt, pose[0], pose[1])
        link_to_current_tf = tf.Transform3d(pos=pos, rot=tf.xyzw_to_wxyz(rot), dtype=self.dtype, device=self.device)
        return link_to_current_tf.inverse().transform_points(pt.view(1, -1)).view(-1)


class PokeEnv(PybulletEnv):
    """To start with we have a fixed gripper orientation so the state is 3D position only"""
    nu = 3
    nx = 3
    MAX_FORCE = 30
    MAX_GRIPPER_FORCE = 30
    MAX_PUSH_DIST = 0.03
    OPEN_ANGLE = 0.055
    CLOSE_ANGLE = 0.0

    @property
    def robot_id(self):
        return self.gripperId

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
        dreaction = state[:, 3:] - other_state[:, 3:]
        return dpos, dreaction

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

    def __init__(self, goal=(0.25, 0.0, 0.2, 0.), init=(0., 0., 0.2),
                 environment_level=0, sim_step_wait=None, mini_steps=15, wait_sim_steps_per_mini_step=20,
                 debug_visualizations=None, dist_for_done=0.04, camera_dist=0.75,
                 contact_residual_threshold=1.,
                 contact_residual_precision=None,
                 reaction_force_strategy=ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS,
                 device="cpu",
                 dtype=torch.float,
                 sdf_resolution=0.025,
                 freespace_voxel_resolution=0.025,
                 clean_cache=False,
                 immovable_target=True,
                 **kwargs):
        """
        :param environment_level: what obstacles should show up in the environment
        :param sim_step_wait: how many seconds to wait between each sim step to show intermediate states
        (0.01 seems reasonable for visualization)
        :param mini_steps how many mini control steps to divide the control step into;
        more is better for controller and allows greater force to prevent sliding
        :param wait_sim_steps_per_mini_step how many sim steps to wait per mini control step executed;
        inversely proportional to mini_steps
        :param contact_residual_threshold magnitude threshold on the reaction residual (measured force and torque
        at end effector) for when we should consider to be in contact with an object
        :param contact_residual_precision if specified, the inverse of a matrix representing the expected measurement
        error of the contact residual; if left none default values for the environment is used. This is used to
        normalize the different dimensions of the residual so that they are around the same expected magnitude.
        Typically estimated from the training set, but for simulated environments can just pass in 1/max training value
        for normalization.
        :param reaction_force_strategy how to aggregate measured reaction forces over control step into 1 value
        :param kwargs:
        """
        super().__init__(**kwargs, default_debug_height=0.1, camera_dist=camera_dist)
        self._dd.toggle_3d(True)
        self.level = Levels(environment_level)
        self.sim_step_wait = sim_step_wait
        # as long as this is above a certain amount we won't exceed it in freespace pushing if we have many mini steps
        self.mini_steps = mini_steps
        self.wait_sim_step_per_mini_step = wait_sim_steps_per_mini_step
        self.reaction_force_strategy = reaction_force_strategy
        self.dist_for_done = dist_for_done

        # object IDs
        self.immovable = []
        self.movable = []

        # debug parameter for extruding objects so penetration is measured wrt the x-y plane
        self.extrude_objects_in_z = False

        # initial config
        self.goal = None
        self.init = None

        # poke target information
        self.obj_factory: YCBObjectFactory = None
        self.sdf: util.ObjectFrameSDF = None
        self.free_voxels: util.VoxelGrid = None

        self.target_model_name = None
        self._target_object_id = None
        self.target_pose = None
        self.ranges = None
        self.freespace_ranges = None
        self.dtype = dtype
        self.device = device
        self.sdf_resolution = sdf_resolution
        self.freespace_voxel_resolution = freespace_voxel_resolution
        self.z = 0.1
        self.clean_cache = clean_cache
        self.immovable_target = immovable_target

        self._debug_visualizations = {
            DebugVisualization.STATE: False,
            DebugVisualization.STATE_TRANSITION: False,
            DebugVisualization.ACTION: False,
            DebugVisualization.REACTION_MINI_STEP: False,
            DebugVisualization.REACTION_IN_STATE: False,
            DebugVisualization.GOAL: False,
            DebugVisualization.INIT: False,
            DebugVisualization.FREE_VOXELS: False,
        }
        if debug_visualizations is not None:
            self._debug_visualizations.update(debug_visualizations)
        self._contact_debug_names = []

        # avoid the spike at the start of each mini step from rapid acceleration
        self._steps_since_start_to_get_reaction = 5
        self._clear_state_between_control_steps()

        self.set_task_config(goal, init)
        self._setup_experiment()
        self._crt = contact_residual_threshold
        self._crp = contact_residual_precision
        self._contact_detector = self.create_contact_detector(self._crt, self._crp)
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()

    def target_object_id(self) -> int:
        return self._target_object_id

    @property
    def contact_detector(self) -> ContactDetector:
        return self._contact_detector

    def create_contact_detector(self, residual_threshold, residual_precision) -> ContactDetector:
        if residual_precision is None:
            residual_precision = np.diag([1, 1, 1, 50, 50, 50])
        contact_detector = ContactDetector(residual_precision)
        contact_detector.register_contact_sensor(PybulletOracleContactSensor(self.robot_id, self.target_object_id))
        # contact_detector.register_contact_sensor(
        #     PybulletResidualPlanarContactSensor("floating_gripper", residual_threshold,
        #                                         robot_id=self.robot_id,
        #                                         canonical_orientation=self.endEffectorOrientation,
        #                                         default_joint_config=[0, 0]))
        return contact_detector

    # --- initialization and task configuration
    def _clear_state_between_control_steps(self):
        self._sim_step = 0
        self._mini_step_contact = {'full': np.zeros((self.mini_steps + 1, 3)),
                                   'torque': np.zeros((self.mini_steps + 1, 3)),
                                   'mag': np.zeros(self.mini_steps + 1),
                                   'id': np.ones(self.mini_steps + 1) * NO_CONTACT_ID}
        self._contact_info = {}
        self._largest_contact = {}
        self._reaction_force = np.zeros(2)

    def _clear_state_before_step(self):
        self.contact_detector.clear_sensors()

    def set_task_config(self, goal=None, init=None):
        if goal is not None:
            self._set_goal(goal)
        if init is not None:
            self._set_init(init)

    def _set_goal(self, goal):
        # ignore the pusher position
        self.goal = np.array(goal)
        if self._debug_visualizations[DebugVisualization.GOAL]:
            self._dd.draw_point('goal', self.goal)

    def _set_init(self, init):
        # initial position of end effector
        self.init = init
        if self._debug_visualizations[DebugVisualization.INIT]:
            self._dd.draw_point('init', self.init, color=(0, 1, 0.2))

    def set_state(self, state, action=None):
        p.resetBasePositionAndOrientation(self.robot_id, (state[0], state[1], state[2]),
                                          self.endEffectorOrientation)
        self.state = state
        self._draw_state()
        if action is not None:
            self._draw_action(action, old_state=state)

    def visualize_rollouts(self, rollout, state_cmap='Blues_r', contact_cmap='Reds_r'):
        """In GUI mode, show how the sequence of states will look like"""
        if rollout is None:
            return
        if type(rollout) is tuple and len(rollout) == 3:
            states, contact_model_active, center_points = rollout
        else:
            states = rollout
            contact_model_active = np.zeros(len(states))
            center_points = [None]
        # assume states is iterable, so could be a bunch of row vectors
        T = len(states)
        if T > 0:
            smap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=T), cmap=state_cmap)
            cmap = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=T), cmap=contact_cmap)
            prev_pos = None
            for t in range(T):
                pos = self.get_ee_pos(states[t])
                rgba = cmap.to_rgba(t) if contact_model_active[t] else smap.to_rgba(t)
                self._dd.draw_point('rx{}.{}'.format(state_cmap, t), pos, rgba[:-1])
                if t > 0:
                    self._dd.draw_2d_line('tx{}.{}'.format(state_cmap, t), prev_pos, pos - prev_pos, scale=1,
                                          color=rgba[:-1])
                prev_pos = pos
        self._dd.clear_visualization_after('rx{}'.format(state_cmap), T)
        self._dd.clear_visualization_after('tx{}'.format(state_cmap), T)

        if center_points[0] is not None:
            obj_center_color_maps = ['Purples_r', 'Greens_r', 'Greys_r']
            # only consider the first sample (m = 0)
            center_points = [pt[:, 0] for pt in center_points]
            center_points = torch.stack(center_points)
            num_objs = center_points.shape[1]
            for j in range(num_objs):
                rollout = center_points[:, j]
                self.visualize_rollouts(rollout.cpu().numpy(),
                                        state_cmap=obj_center_color_maps[j % len(obj_center_color_maps)])
            # clear the other colors
            for j in range(num_objs, len(obj_center_color_maps)):
                self.visualize_rollouts([], state_cmap=obj_center_color_maps[j % len(obj_center_color_maps)])

    def visualize_goal_set(self, states):
        if states is None:
            return
        T = len(states)
        for t in range(T):
            pos = self.get_ee_pos(states[t])
            c = (t + 1) / (T + 1)
            self._dd.draw_point('gs.{}'.format(t), pos, (c, c, c))
        self._dd.clear_visualization_after('gs', T)

    def visualize_trap_set(self, trap_set):
        if trap_set is None:
            return
        T = len(trap_set)
        for t in range(T):
            c = (t + 1) / (T + 1)
            # decide whether we're given state and action or just state
            if len(trap_set[t]) == 2:
                state, action = trap_set[t]
                self._draw_action(action.cpu().numpy(), old_state=state.cpu().numpy(), debug=t + 1)
            else:
                state = trap_set[t]
            pose = self.get_ee_pos(state)
            self._dd.draw_point('ts.{}'.format(t), pose, (1, 0, c))
        self._dd.clear_visualization_after('ts', T)
        self._dd.clear_visualization_after('u', T + 1)

    def visualize_state_actions(self, base_name, states, actions, state_c, action_c, action_scale):
        if torch.is_tensor(states):
            states = states.cpu()
            if actions is not None:
                actions = actions.cpu()
        j = -1
        for j in range(len(states)):
            p = self.get_ee_pos(states[j])
            name = '{}.{}'.format(base_name, j)
            self._dd.draw_point(name, p, color=state_c)
            if actions is not None:
                # draw action
                name = '{}a.{}'.format(base_name, j)
                self._dd.draw_2d_line(name, p, actions[j], color=action_c, scale=action_scale)
        self._dd.clear_visualization_after(base_name, j + 1)
        self._dd.clear_visualization_after('{}a'.format(base_name), j + 1)

    def visualize_contact_set(self, contact_set: tracking.ContactSet):
        if isinstance(contact_set, tracking.ContactSetHard):
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(contact_set):
                for name in self._contact_debug_names:
                    self._dd.clear_visualizations(name)
                self._contact_debug_names = []
            for i, c in enumerate(contact_set):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                name = 'cp{}'.format(i)
                eigval, eigvec = torch.eig(c.cov[0], eigenvectors=True)
                yx_ratio = eigval[1, 0] / eigval[0, 0]
                rot = math.atan2(eigvec[1, 0], eigvec[0, 0])
                l = eigval[0, 0] * 100
                w = c.weight
                self._dd.draw_point(name, self.get_ee_pos(c.mu[0]), length=l.item(), length_ratio=yx_ratio, rot=rot,
                                    color=color)
                self._contact_debug_names[i].add(name)

                base_name = str(i)
                self.visualize_state_actions(base_name, c.points, c.actions, color, u_color, 0.1 * w)

                for j in range(len(c.points)):
                    self._contact_debug_names[i].add('{}{}'.format(base_name, j))
                    self._contact_debug_names[i].add('{}{}a'.format(base_name, j))
        elif isinstance(contact_set, tracking.ContactSetSoft):
            pts = contact_set.get_posterior_points()
            if pts is None:
                return
            groups = contact_set.get_hard_assignment(contact_set.p.hard_assignment_threshold)
            # clear all previous markers because we don't know which one was removed
            if len(self._contact_debug_names) > len(groups):
                for name in self._contact_debug_names:
                    self._dd.clear_visualizations(name)
                self._contact_debug_names = []
            for i, indices in enumerate(groups):
                color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
                if i >= len(self._contact_debug_names):
                    self._contact_debug_names.append(set())
                # represent the uncertainty of the center point
                base_name = str(i)
                self.visualize_state_actions(base_name, pts[indices], contact_set.acts[indices], color, u_color, 0.1)
                for j in range(len(pts[indices])):
                    self._contact_debug_names[i].add('{}{}'.format(base_name, j))
                    self._contact_debug_names[i].add('{}{}a'.format(base_name, j))

        # clean up any old visualization
        # for i in range(len(contact_set), len(self._contact_debug_names)):
        #     self._dd.clear_visualizations(self._contact_debug_names[i])
        # self._contact_debug_names = self._contact_debug_names[:len(contact_set)]

    def visualize_prediction_error(self, predicted_state):
        """In GUI mode, show the difference between the predicted state and the current actual state"""
        pred = self.get_ee_pos(predicted_state)
        c = (0.5, 0, 0.5)
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('ep', pred, c)

    def clear_debug_trajectories(self):
        self._dd.clear_transitions()

    def _draw_state(self):
        if self._debug_visualizations[DebugVisualization.STATE]:
            pos = self.get_ee_pos(self.state)
            self._dd.draw_point('state', pos)
        if self._debug_visualizations[DebugVisualization.REACTION_IN_STATE]:
            self._draw_reaction_force(self.state[3:6], 'sr', (0, 0, 0))

    def _draw_action(self, action, old_state=None, debug=0):
        if old_state is None:
            old_state = self._obs()
        start = old_state[:3]
        pointer = action
        if debug:
            self._dd.draw_2d_line('u.{}'.format(debug), start, pointer, (1, debug / 30, debug / 10), scale=0.2)
        else:
            self._dd.draw_2d_line('u', start, pointer, (1, 0, 0), scale=0.2)

    def _draw_reaction_force(self, r, name, color=(1, 0, 1)):
        start = self.get_ee_pos(self._obs())
        self._dd.draw_2d_line(name, start, r, size=np.linalg.norm(r), scale=0.03, color=color)

    # --- observing state from simulation
    def _obs(self):
        """Observe current state from simulator"""
        state = self._observe_ee(return_z=True)
        return state

    def _observe_ee(self, return_z=False, return_orientation=False):
        gripperPose = p.getBasePositionAndOrientation(self.robot_id)
        pos = gripperPose[0]
        if not return_z:
            pos = pos[:2]
        if return_orientation:
            pos = pos, gripperPose[1]
        return pos

    def open_gripper(self):
        p.setJointMotorControlArray(self.robot_id,
                                    [PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.OPEN_ANGLE, self.OPEN_ANGLE],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def close_gripper(self):
        p.setJointMotorControlArray(self.robot_id,
                                    [PandaJustGripperID.FINGER_A, PandaJustGripperID.FINGER_B],
                                    p.POSITION_CONTROL,
                                    targetPositions=[self.CLOSE_ANGLE, self.CLOSE_ANGLE],
                                    forces=[self.MAX_GRIPPER_FORCE, self.MAX_GRIPPER_FORCE])

    def _observe_reaction_force_torque(self):
        """Return representative reaction force for simulation steps up to current one since last control step"""
        if self.reaction_force_strategy is ReactionForceStrategy.AVG_OVER_MINI_STEPS:
            return self._mini_step_contact['full'].mean(axis=0), self._mini_step_contact['torque'].mean(axis=0)
        if self.reaction_force_strategy is ReactionForceStrategy.MEDIAN_OVER_MINI_STEPS:
            median_mini_step = np.argsort(self._mini_step_contact['mag'])[self.mini_steps // 2]
            return self._mini_step_contact['full'][median_mini_step], \
                   self._mini_step_contact['torque'][median_mini_step]
        if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_MINI_STEPS:
            max_mini_step = np.argmax(self._mini_step_contact['mag'])
            return self._mini_step_contact['full'][max_mini_step], self._mini_step_contact['torque'][max_mini_step]
        else:
            raise NotImplementedError("Not implemented max over control step reaction torque")

    def _observe_additional_info(self, info, visualize=True):
        reaction_force = [0, 0, 0]
        reaction_torque = [0, 0, 0]

        ee_pos = self._observe_ee(return_z=True)
        for objectId in self.objects:
            contactInfo = self.get_ee_contact_info(objectId)
            for i, contact in enumerate(contactInfo):
                f_contact = get_total_contact_force(contact, False)
                reaction_force = [sum(i) for i in zip(reaction_force, f_contact)]
                # torque wrt end effector position
                pos_vec = np.subtract(contact[ContactInfo.POS_A], ee_pos)
                r_contact = np.cross(pos_vec, f_contact)
                reaction_torque = np.add(reaction_torque, r_contact)

        self._observe_raw_reaction_force(info, reaction_force, reaction_torque, visualize)

    def _observe_info(self, visualize=True):
        info = {}

        self._observe_additional_info(info, visualize)
        self._sim_step += 1

        for key, value in info.items():
            if key not in self._contact_info:
                self._contact_info[key] = []
            self._contact_info[key].append(value)

    def _observe_raw_reaction_force(self, info, reaction_force, reaction_torque, visualize=True):
        # can estimate change in state only when in contact
        new_ee_pos, new_ee_orientation = self._observe_ee(return_z=True, return_orientation=True)
        pose = (new_ee_pos, new_ee_orientation)
        if self.contact_detector.observe_residual(np.r_[reaction_force, reaction_torque], pose):
            dx = np.subtract(new_ee_pos, self.last_ee_pos)
            # for step size resolution issues, explicitly say there is no dx during collision
            if self.immovable_target:
                dx = np.zeros_like(dx)
            self.contact_detector.observe_dx(dx)
            info[InfoKeys.DEE_IN_CONTACT] = dx
        self.last_ee_pos = new_ee_pos

        # save end effector pose
        info[InfoKeys.HIGH_FREQ_EE_POSE] = np.r_[new_ee_pos, new_ee_orientation]

        # save reaction force
        info[InfoKeys.HIGH_FREQ_REACTION_T] = reaction_torque
        name = InfoKeys.HIGH_FREQ_REACTION_F
        info[name] = reaction_force
        reaction_force_size = np.linalg.norm(reaction_force)
        # see if we should save it as the reaction force for this mini-step
        mini_step, step_since_start = divmod(self._sim_step, self.wait_sim_step_per_mini_step)

        # detect what we are in contact with
        for bodyId in self.movable + self.immovable:
            contactInfo = self.get_ee_contact_info(bodyId)
            # assume at most single body in contact
            if len(contactInfo):
                self._mini_step_contact['id'][mini_step] = bodyId
                pt = contactInfo[0][ContactInfo.POS_A]
                info[InfoKeys.HIGH_FREQ_CONTACT_POINT] = pt
                break
        else:
            info[InfoKeys.HIGH_FREQ_CONTACT_POINT] = [0, 0, 0]

        if step_since_start is self._steps_since_start_to_get_reaction:
            self._mini_step_contact['full'][mini_step] = reaction_force
            self._mini_step_contact['torque'][mini_step] = reaction_torque
            self._mini_step_contact['mag'][mini_step] = reaction_force_size
            if self.reaction_force_strategy is not ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_MINI_STEP] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))
        # update our running count of max force
        if reaction_force_size > self._largest_contact.get(name, 0):
            self._largest_contact[name] = reaction_force_size
            self._reaction_force = reaction_force
            if self.reaction_force_strategy is ReactionForceStrategy.MAX_OVER_CONTROL_STEP and \
                    self._debug_visualizations[DebugVisualization.REACTION_MINI_STEP] and visualize:
                self._draw_reaction_force(reaction_force, name, (1, 0, 1))

    def _aggregate_info(self):
        info = {key: np.stack(value, axis=0) for key, value in self._contact_info.items() if len(value)}
        info[InfoKeys.LOW_FREQ_REACTION_F], info[InfoKeys.LOW_FREQ_REACTION_T] = self._observe_reaction_force_torque()
        name = InfoKeys.DEE_IN_CONTACT
        if name in info:
            info[name] = info[name].sum(axis=0)
        else:
            info[name] = np.zeros(3)
        most_frequent_contact = scipy.stats.mode(self._mini_step_contact['id'])
        info[InfoKeys.CONTACT_ID] = int(most_frequent_contact[0])

        # ground truth object information
        if len(self.movable + self.immovable):
            for obj_id in self.movable + self.immovable:
                pose = p.getBasePositionAndOrientation(obj_id)
                c = p.getClosestPoints(obj_id, self.robot_id, 100000)
                info[f"obj{obj_id}pose"] = np.concatenate([pose[0], pose[1]])
                # for multi-link bodies, will return 1 per combination; store the min
                info[f"obj{obj_id}distance"] = min(cc[ContactInfo.DISTANCE] for cc in c)

        return info

    def get_ee_contact_info(self, bodyId):
        return p.getContactPoints(self.robot_id, bodyId)

    # --- control helpers (rarely overridden)
    def evaluate_cost(self, state, action=None):
        diff = self.get_ee_pos(state) - self.get_ee_pos(self.goal)
        dist = np.linalg.norm(diff)
        done = dist < self.dist_for_done
        return (dist * 10) ** 2, done

    def _finish_action(self, old_state, action):
        """Evaluate action after finishing it; step should not modify state after calling this"""
        self.state = np.array(self._obs())

        # track trajectory
        prev_block = self.get_ee_pos(old_state)
        new_block = self.get_ee_pos(self.state)
        if self._debug_visualizations[DebugVisualization.STATE_TRANSITION]:
            self._dd.draw_transition(prev_block, new_block)

        # render current pose
        self._draw_state()

        cost, done = self.evaluate_cost(self.state, action)
        # summarize information per sim step into information for entire control step
        info = self._aggregate_info()

        # prepare for next control step
        self._clear_state_between_control_steps()

        self._occupy_current_config_as_freespace()

        return cost, done, info

    def _occupy_current_config_as_freespace(self):
        # occupy free space with swept volume (transform robot interior points to current configuration)
        pose = p.getBasePositionAndOrientation(self.robot_id)
        link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        robot_interior_points_world = link_to_current_tf.transform_points(self.robot_interior_points_orig)
        self.free_voxels[robot_interior_points_world] = 1

        if self._debug_visualizations[DebugVisualization.FREE_VOXELS]:
            i = 0
            free_space_world_frame_points, _ = self.free_voxels.get_known_pos_and_values()
            for i, pt in enumerate(free_space_world_frame_points):
                self.vis.draw_point(f"fspt.{i}", pt, color=(1, 0, 1), scale=2, length=0.003)
            self.vis.clear_visualization_after("fspt", i + 1)

    # --- control (commonly overridden)
    def _move_and_wait(self, eePos, steps_to_wait=50):
        # execute the action
        self.last_ee_pos = self._observe_ee(return_z=True)
        self._move_pusher(eePos)
        p.stepSimulation()
        for _ in range(steps_to_wait):
            self._observe_info()
            p.stepSimulation()
            if self.mode is p.GUI and self.sim_step_wait:
                time.sleep(self.sim_step_wait)
        self._observe_info()

    def _unpack_action(self, action):
        dx = action[0] * self.MAX_PUSH_DIST
        dy = action[1] * self.MAX_PUSH_DIST
        dz = action[2] * self.MAX_PUSH_DIST
        return dx, dy, dz

    def step(self, action):
        self._clear_state_before_step()

        action = np.clip(action, *self.get_control_bounds())
        # normalize action such that the input can be within a fixed range
        old_state = np.copy(self.state)
        dx, dy, dz = self._unpack_action(action)

        ee_pos = self.get_ee_pos(old_state)
        final_ee_pos = np.array((ee_pos[0] + dx, ee_pos[1] + dy, ee_pos[2] + dz))

        if self._debug_visualizations[DebugVisualization.ACTION]:
            self._draw_action(action)
            self._dd.draw_point('final eepos', final_ee_pos, color=(1, 0.5, 0.5))

        # execute push with mini-steps
        for step in range(self.mini_steps):
            intermediate_ee_pos = interpolate_pos(ee_pos, final_ee_pos, (step + 1) / self.mini_steps)
            self._move_and_wait(intermediate_ee_pos, steps_to_wait=self.wait_sim_step_per_mini_step)

        cost, done, info = self._finish_action(old_state, action)

        return np.copy(self.state), -cost, done, info

    def _move_pusher(self, end):
        p.changeConstraint(self.gripperConstraint, end, self.endEffectorOrientation, maxForce=self.MAX_FORCE)

    def _setup_experiment(self):
        # set gravity
        p.setGravity(0, 0, -10)
        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self.set_camera_position([0.3, 0.1, 0.1], -40, -50)

        self._setup_gripper()
        self._setup_objects()

        self.state = self._obs()
        self._draw_state()

    def _adjust_mass_and_visual(self, obj, m):
        # adjust the mass of a pybullet object and indicate it via the color
        p.changeVisualShape(obj, -1, rgbaColor=[1 - m / 3, 0.8 - m / 3, 0.2, 0.8])
        p.changeDynamics(obj, -1, mass=m)

    def _adjust_box_dynamics(self, obj):
        p.changeDynamics(obj, -1, lateralFriction=0.8, spinningFriction=0.05, rollingFriction=0.01)

    def _setup_gripper(self):
        self.endEffectorOrientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        # use a floating gripper
        self.gripperId = p.loadURDF(os.path.join(cfg.URDF_DIR, "panda_gripper.urdf"), self.init,
                                    self.endEffectorOrientation)

        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_A, lateralFriction=2)
        p.changeDynamics(self.gripperId, PandaJustGripperID.FINGER_B, lateralFriction=2)

        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.init, childFrameOrientation=self.endEffectorOrientation)

        self.close_gripper()
        self._make_robot_translucent(self.gripperId)

    def create_target_obj(self, target_pos, target_rot, flags, immovable=False, mass=2):
        self.z = 0.1
        self.target_model_name = level_to_obj_map.get(self.level)

        self.obj_factory = obj_factory_map(self.target_model_name)
        # reset target and robot to their object frame to create the SDF
        # self.target_pose = target_pos, target_rot
        self._target_object_id, self.ranges = self.obj_factory.make_collision_obj(self.z)
        # initialize object at intended target, then we get to see what its achievable pose is
        p.resetBasePositionAndOrientation(self._target_object_id, target_pos, target_rot)
        for _ in range(1000):
            p.stepSimulation()
        p.changeDynamics(self._target_object_id, -1, mass=0 if immovable else mass)
        self.target_pose = p.getBasePositionAndOrientation(self._target_object_id)
        self.draw_mesh("base_object", self.target_pose, (1.0, 1.0, 0., 0.5), object_id=self.vis.USE_DEFAULT_ID_FOR_NAME)

        # ranges is in object frame, centered on 0; our experiment workspace takes on x > 0 and z > 0 mostly
        if self.level in [Levels.HAMMER, Levels.HAMMER_2]:
            self.freespace_ranges = np.array([[-0.1, 0.5],
                                              [-0.3, 0.5],
                                              [-0.075, 0.4]])
        else:
            self.freespace_ranges = np.array([[-0.1, 0.5],
                                              [-0.3, 0.3],
                                              [-0.075, 0.625]])
        self._create_sdfs()

    def draw_mesh(self, *args, **kwargs):
        return self.obj_factory.draw_mesh(self.vis, *args, **kwargs)

    def _create_sdfs(self):
        rob_pos, rob_rot = p.getBasePositionAndOrientation(self.robot_id)
        target_pos, target_rot = self.target_pose

        p.resetBasePositionAndOrientation(self._target_object_id, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)
        p.resetBasePositionAndOrientation(self.robot_id, self.LINK_FRAME_POS, self.LINK_FRAME_ORIENTATION)

        # SDF for the object
        obj_frame_sdf = stucco.sdf.MeshSDF(self.obj_factory)
        self.target_sdf = stucco.sdf.CachedSDF(self.obj_factory.name, self.sdf_resolution, self.ranges,
                                               obj_frame_sdf, device=self.device, clean_cache=self.clean_cache)
        if self.clean_cache:
            # display the voxels created for this sdf
            interior_pts = self.target_sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < 0.0)
            for i, pt in enumerate(interior_pts):
                self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
            input("interior SDF points for target object (press enter to confirm)")

        # SDF for the robot (used for filling up freespace voxels)
        # should be fine to use the actual robot ID for the ground truth SDF since we won't be querying outside of the
        # SDF range (and thus need to actually use the GT lookup)
        robot_frame_sdf = stucco.sdf.PyBulletNaiveSDF(self.robot_id)
        self.robot_sdf = stucco.sdf.CachedSDF("floating_gripper", 0.01, self.ranges / 3,
                                              robot_frame_sdf, device=self.device, clean_cache=self.clean_cache)
        self.robot_interior_points_orig = self.robot_sdf.get_filtered_points(lambda voxel_sdf: voxel_sdf < -0.01)

        if self.clean_cache:
            for i, pt in enumerate(self.robot_interior_points_orig):
                self.vis.draw_point(f"mipt.{i}", pt, color=(0, 1, 1), length=0.003, scale=4)
            self.vis.clear_visualization_after("mipt", i + 1)
            input("interior SDF points for robot (press enter to confirm)")
            self.vis.clear_visualization_after("mipt", 0)

        self.free_voxels = util.VoxelGrid(self.freespace_voxel_resolution, self.freespace_ranges, device=self.device)

        # register floor as freespace

        floor_range = self.ranges.copy()
        # having the whole sdf's floor takes too long to draw (pybullet takes a long time to draw)
        # so for debugging/visualization use the following lines; otherwise uncomment it and use the whole floor
        floor_offset = 0.25
        floor_range[0, 0] = self.target_pose[0][0] - floor_offset
        floor_range[0, 1] = self.target_pose[0][0] + floor_offset
        floor_range[1, 0] = self.target_pose[0][1] - floor_offset
        floor_range[1, 1] = self.target_pose[0][1] + floor_offset

        floor_range[2, 0] = -self.freespace_voxel_resolution * 3
        floor_range[2, 1] = -self.freespace_voxel_resolution * 2
        floor_coord, floor_pts = util.get_coordinates_and_points_in_grid(self.freespace_voxel_resolution, floor_range,
                                                                         dtype=self.dtype, device=self.device)
        self.free_voxels[floor_pts] = 1

        # restore robot pose
        p.resetBasePositionAndOrientation(self.robot_id, rob_pos, rob_rot)
        p.resetBasePositionAndOrientation(self._target_object_id, target_pos, target_rot)

    def _setup_objects(self):
        self.immovable = []
        self.movable = []
        z = 0.1
        h = 2 if self.extrude_objects_in_z else 0.15
        separation = 0.7

        # make walls
        self.immovable.append(make_box([0.7, 0.1, h], [1.1, 0, z], [0, 0, -np.pi / 2]))
        self.immovable.append(make_box([0.7, 0.1, h], [0.5, -separation, z], [0, 0, 0]))
        self.immovable.append(make_box([0.7, 0.1, h], [0.5, separation, z], [0, 0, 0]))

        flags = p.URDF_USE_INERTIA_FROM_FILE
        target_pos = [self.goal[0], self.goal[1], self.goal[2]]
        # all targets are upright
        target_rot = self.goal[3:]
        if len(target_rot) == 1:
            target_rot = p.getQuaternionFromEuler([0, 0, target_rot[0]])
        elif len(target_rot) == 3:
            target_rot = p.getQuaternionFromEuler(target_rot)

        self.create_target_obj(target_pos, target_rot, flags, immovable=self.immovable_target)
        p.changeDynamics(self.planeId, -1, lateralFriction=0.6, spinningFriction=0.8)
        self.movable.append(self._target_object_id)

        for objId in self.immovable:
            p.changeVisualShape(objId, -1, rgbaColor=[0.2, 0.2, 0.2, 0.8])
        self.objects = self.immovable + self.movable

    def reset(self):
        for _ in range(1000):
            p.stepSimulation()

        self.open_gripper()
        if self.gripperConstraint:
            p.removeConstraint(self.gripperConstraint)

        for obj in self.immovable + self.movable:
            p.removeBody(obj)
        self._setup_objects()

        p.resetBasePositionAndOrientation(self.gripperId, self.init, self.endEffectorOrientation)
        self.gripperConstraint = p.createConstraint(self.gripperId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0],
                                                    self.init, childFrameOrientation=self.endEffectorOrientation)
        self.close_gripper()

        # set robot init config
        self._clear_state_between_control_steps()
        # start at rest
        for _ in range(1000):
            p.stepSimulation()
        self.state = self._obs()
        if self._debug_visualizations[DebugVisualization.STATE]:
            self._dd.draw_point('x0', self.get_ee_pos(self.state), color=(0, 1, 0))

        self.contact_detector.clear()
        # recreate SDFs so we don't have stale data
        self._create_sdfs()
        return np.copy(self.state)


def interpolate_pos(start, end, t):
    return t * end + (1 - t) * start


class ExperimentRunner(PybulletSim):
    def __init__(self, env: PokeEnv, ctrl, save_dir=DIR, **kwargs):
        reaction_dim = 3
        super(ExperimentRunner, self).__init__(env, ctrl, save_dir=save_dir, reaction_dim=reaction_dim, **kwargs)


class PokeDataSource(EnvDataSource):

    @staticmethod
    def _default_data_dir():
        return DIR

    @staticmethod
    def _loader_map(env_type):
        loader_map = {PokeEnv: PokeLoader, }
        return loader_map.get(env_type, None)


def pt_to_config_dist(env, max_robot_radius, configs, pts):
    M = configs.shape[0]
    N = pts.shape[0]
    dist = torch.zeros((M, N), dtype=pts.dtype, device=pts.device)

    orig_pos, orig_orientation = p.getBasePositionAndOrientation(env.robot_id)
    z = orig_pos[2]

    # to speed up distance checking, we compute distance from center of robot config to point
    # and avoid the expensive check of distance to surface for those that are too far away
    center_dist = torch.cdist(configs[:, :2].view(-1, 2), pts[:, :2].view(-1, 2))

    # if visualize:
    #     for i, pt in enumerate(pts):
    #         env._dd.draw_point(f't{i}', pt, color=(1, 0, 0), height=z)

    for i in range(M):
        p.resetBasePositionAndOrientation(env.robot_id, [configs[i][0], configs[i][1], z], orig_orientation)
        for j in range(N):
            if center_dist[i, j] > max_robot_radius:
                # just have to report something > 0
                dist[i, j] = 1
            else:
                closest = closest_point_on_surface(env.robot_id, [pts[j][0], pts[j][1], z])
                dist[i, j] = closest[ContactInfo.DISTANCE]

    p.resetBasePositionAndOrientation(env.robot_id, orig_pos, orig_orientation)
    return dist


# TODO remove old SDF handler
class ArmPointToConfig(PlanarPointToConfig):
    def __init__(self, env):
        # try loading cache
        fullname = os.path.join(cfg.DATA_DIR, f'arm_point_to_config.pkl')
        if os.path.exists(fullname):
            super(ArmPointToConfig, self).__init__(*torch.load(fullname))
        else:
            import trimesh
            hand = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/hand.obj"))
            lf = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/finger.obj"))
            lf.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0.0584]))
            rf = trimesh.load(os.path.join(cfg.ROOT_DIR, "meshes/collision/finger.obj"))
            rf.apply_transform(
                trimesh.transformations.concatenate_matrices(trimesh.transformations.euler_matrix(0, 0, np.pi),
                                                             trimesh.transformations.translation_matrix(
                                                                 [0, 0, 0.0584])))
            mesh = trimesh.util.concatenate([hand, lf, rf])
            # TODO get resting orientation from environment?
            mesh.apply_transform(trimesh.transformations.euler_matrix(0, np.pi / 2, 0))
            # cache points inside bounding box of robot to accelerate lookup
            min_x, min_y = mesh.bounding_box.bounds[0, :2]
            max_x, max_y = mesh.bounding_box.bounds[1, :2]
            cache_resolution = 0.001
            # create mesh grid
            x = np.arange(min_x, max_x + cache_resolution, cache_resolution)
            y = np.arange(min_y, max_y + cache_resolution, cache_resolution)
            cache_y_len = len(y)

            orig_pos, orig_orientation = p.getBasePositionAndOrientation(env.robot_id)
            p.resetBasePositionAndOrientation(env.robot_id, [0, 0, 0], orig_orientation)
            d = np.zeros((len(x), len(y)))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    closest = closest_point_on_surface(env.robot_id, [xi, yj, 0])
                    d[i, j] = closest[ContactInfo.DISTANCE]
            d_cache = d.reshape(-1)
            data = [d_cache, min_x, min_y, max_x, max_y, cache_resolution, cache_y_len]
            torch.save(data, fullname)
            super(ArmPointToConfig, self).__init__(*data)


class YCBObjectFactory(ObjectFactory):
    def __init__(self, name, ycb_name, ranges=np.array([[-.15, .15], [-.15, .15], [-0.15, .4]]), **kwargs):
        super(YCBObjectFactory, self).__init__(name, **kwargs)
        self.ycb_name = ycb_name
        self.ranges = ranges * self.scale

    def make_collision_obj(self, z, rgba=None):
        obj_id = p.loadURDF(os.path.join(cfg.URDF_DIR, self.ycb_name, "model.urdf"),
                            [0., 0., z * 3],
                            p.getQuaternionFromEuler([0, 0, -1]), globalScaling=self.scale, **self.other_load_kwargs)
        if rgba is not None:
            p.changeVisualShape(obj_id, -1, rgbaColor=rgba)
        return obj_id, self.ranges

    def get_mesh_resource_filename(self):
        return os.path.join(cfg.URDF_DIR, self.ycb_name, "textured_simple_reoriented.obj")

    def get_mesh_high_poly_resource_filename(self):
        return os.path.join(cfg.URDF_DIR, self.ycb_name, "textured_simple_reoriented.obj")


def obj_factory_map(obj_name):
    if obj_name == "mustard":
        return YCBObjectFactory("mustard", "YcbMustardBottle",
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, 1.57 - 0.1]),
                                vis_frame_pos=[-0.005, -0.005, 0.015])
    if obj_name == "banana":
        return YCBObjectFactory("banana", "YcbBanana", ranges=np.array([[-.075, .075], [-.075, .075], [-0.1, .15]]),
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, 0]),
                                vis_frame_pos=[-.01, 0.0, -.01])
    if obj_name == "drill":
        return YCBObjectFactory("drill", "YcbPowerDrill", ranges=np.array([[-.075, .125], [-.075, .075], [-0.08, .15]]),
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, -0.6]),
                                vis_frame_pos=[-0.002, -0.011, -.06])
    if obj_name == "hammer":
        # TODO seems to be a problem with the hammer experiment
        return YCBObjectFactory("hammer", "YcbHammer", ranges=np.array([[-.065, .065], [-.095, .095], [-0.1, .1]]),
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, 0]),
                                vis_frame_pos=[-0.02, 0.01, -0.01])
    if obj_name == "box":
        # TODO seems to be a problem with the hammer experiment
        return YCBObjectFactory("box", "YcbCrackerBox", ranges=np.array([[-.075, .125], [-.075, .075], [-0.08, .15]]),
                                vis_frame_rot=p.getQuaternionFromEuler([0, 0, 0]),
                                vis_frame_pos=[-0.03, 0.01, 0.02])
    # TODO create the other object factories
