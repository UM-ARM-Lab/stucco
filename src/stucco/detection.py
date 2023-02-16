import torch
import math
import abc
import typing
from collections import deque

import pytorch_kinematics.transforms as tf
from arm_pytorch_utilities import math_utils, tensor_utils
import numpy as np

point = torch.tensor
points = torch.tensor
normals = torch.tensor


class ContactSensor:
    """An independent sensor that holds sensor related information during an action"""

    def __init__(self, dtype=torch.float, device='cpu'):
        self.in_contact = False
        self.dx = []
        self.dtype = dtype
        self.device = device

    def clear(self):
        self.in_contact = False
        self.dx = []

    def observe_residual(self, residual):
        pass

    def get_dx(self):
        dx = np.sum(self.dx, axis=0)
        return torch.tensor(dx, dtype=self.dtype, device=self.device)

    def observe_dx(self, dx):
        if self.in_contact:
            self.dx.append(dx)

    @abc.abstractmethod
    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        """Return contact point in link frame that most likely explains the observed residual"""


class ResidualPlanarContactSensor(ContactSensor):
    """Contact sensor based on generalized momentum residual measurements only"""

    def __init__(self, surface_points, surface_normals, residual_threshold, max_friction_cone_angle=50 * math.pi / 180,
                 dtype=torch.float, device='cpu'):
        """
        :param surface_points: N x 3 sampled points on the robot surface in link frame
        :param surface_normals: N x 3 corresponding surface normals to the surface points, only necessary for
        visualization
        :param residual_threshold: contact threshold for residual^T sigma_meas^-1 residual to count as being in contact
        :param max_friction_cone_angle: max angular difference (radian) from surface normal that a contact is possible
        note that 90 degrees is the weakest assumption that the force is from a push and not a pull
        """
        self._cached_points = surface_points
        self._cached_normals = surface_normals

        self._residual_threshold = residual_threshold
        self.max_friction_cone_angle = max_friction_cone_angle

        super().__init__(dtype=dtype, device=device)

    def observe_residual(self, residual):
        self.in_contact = residual > self._residual_threshold

    def get_jacobian(self, locations, q=None):
        """Get J^T in the equation: wrench at end effector = J^T * wrench at contact point.
        In general the Jacobian is dependent on configuration.

        Locations are specified wrt the end effector frame.
        For planar robots, this kind of Jacobian is configuration independent"""

        return torch.stack(
            [torch.tensor([[1., 0.], [0., 1.], [-loc[1], loc[0]]], device=self.device, dtype=self.dtype) for loc in
             locations])

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        # 2D
        link_frame_pts, pts, normals = self.sample_robot_surface_points(pose, visualizer=visualizer)
        F_c = ee_force_torque[:, :2]
        T_ee = ee_force_torque[:, -1]

        while True:
            # reject points where force is sufficiently away from surface normal
            from_normal = math_utils.angle_between(normals[:, :2], -F_c)
            valid = from_normal < self.max_friction_cone_angle
            # validity has to hold across all experienced forces
            valid = valid.all(dim=1)
            # remove a single element of the constraint if we don't have any satisfying them
            if valid.any():
                break
            else:
                F_c = F_c[:-1]
                T_ee = T_ee[:-1]

        # no valid point
        if F_c.numel() == 0:
            return None

        pts = pts[valid]
        link_frame_pts = link_frame_pts[valid]

        # NOTE: if our system is able to rotate, would have to transform points by rotation too
        # get relative to end effector origin
        rel_pts = pts - pose[0]
        J = self.get_jacobian(rel_pts, q=q)
        # J_{r_c}^T F_c
        expected_residual = J @ F_c.transpose(-1, -2)

        # the below is the case for full residual; however we can shortcut since we only need to compare torque
        # error = ee_force_torque - expected_residual
        # combined_error = linalg.batch_quadratic_product(error, self.residual_precision)

        error = expected_residual[:, -1] - T_ee
        # don't have to worry about normalization since it's just the torque dimension
        combined_error = error.abs().sum(dim=1)

        min_err_i = torch.argmin(combined_error)

        if visualizer is not None:
            # also draw some other likely points
            likely_pt_index = torch.argsort(combined_error)
            for i in reversed(range(1, min(10, len(pts)))):
                pt = pts[likely_pt_index[i]]
                visualizer.draw_point(f'likely.{i}', pt, height=pt[2] + 0.001,
                                      color=(0, 1 - 0.8 * i / 10, 0))
            visualizer.draw_point(f'most likely contact', pts[min_err_i], color=(0, 1, 0), scale=2)
            visualizer.draw_2d_line('reaction', pts[min_err_i], ee_force_torque.mean(dim=0)[:3], color=(0, 0.2, 1.0),
                                    scale=0.2)

        return link_frame_pts[min_err_i]

    def sample_robot_surface_points(self, pose, visualizer=None) -> typing.Tuple[points, points, normals]:
        """Get points on the surface of the robot that could be possible contact locations
        pose[0] and pose[1] are the position and orientation (quaternion) of the end effector, respectively.
        Also return the correspnoding surface normals for each of the points.
        """
        if self._cached_points.dtype != self.dtype or self._cached_points.device != self.device:
            self._cached_points = self._cached_points.to(device=self.device, dtype=self.dtype)
            self._cached_normals = self._cached_normals.to(device=self.device, dtype=self.dtype)

        link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(
            tensor_utils.ensure_tensor(self.device, self.dtype, pose[1])), dtype=self.dtype, device=self.device)
        pts = link_to_current_tf.transform_points(self._cached_points)
        normals = link_to_current_tf.transform_normals(self._cached_normals)
        if visualizer is not None:
            for i, pt in enumerate(pts):
                visualizer.draw_point(f't.{i}', pt, color=(1, 0, 0), height=pt[2])
                visualizer.draw_2d_line(f'n.{i}', pt, normals[i], color=(0.5, 0, 0), size=2., scale=0.1)

        return self._cached_points, pts, normals


class ContactDetector:
    """Detect and isolate contacts given some form of generalized momentum residual measurements, see
    https://ieeexplore.ieee.org/document/7759743 (Localizing external contact using proprioceptive sensors).

    We additionally assume access to force torque sensors at the end effector, which is our residual."""

    def __init__(self, residual_precision, window_size=5, dtype=torch.float, device='cpu'):
        """
        :param residual_precision: sigma_meas^-1 matrix that scales the different residual dimensions based on their
        expected precision
        """

        self.residual_precision = residual_precision
        self.observation_history = deque(maxlen=500)
        self._window_size = window_size
        self.dtype = dtype
        self.device = device

        self.sensors: typing.List[ContactSensor] = []

    def register_contact_sensor(self, sensor: ContactSensor):
        sensor.dtype = self.dtype
        sensor.device = self.device
        self.sensors.append(sensor)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
            for sensor in self.sensors:
                sensor.device = device
        if dtype is not None:
            self.dtype = dtype
            for sensor in self.sensors:
                sensor.dtype = dtype

    def observe_residual(self, ee_force_torque, pose=None):
        """Returns whether this residual implies we are currently in contact and track its location if given pose"""
        epsilon = ee_force_torque.T @ self.residual_precision @ ee_force_torque

        in_contact = False

        for sensor in self.sensors:
            sensor.observe_residual(epsilon)
            if sensor.in_contact:
                in_contact = True

        self.observation_history.append((in_contact, ee_force_torque, pose))

        return in_contact

    def observe_dx(self, dx):
        for sensor in self.sensors:
            sensor.observe_dx(dx)

    def clear_sensors(self):
        for sensor in self.sensors:
            sensor.clear()

    def __len__(self):
        return len(self.observation_history)

    def clear(self):
        self.observation_history.clear()
        self.clear_sensors()

    def isolate_contact(self, ee_force_torque, pose, q=None, visualizer=None):
        """Return contact points in link frame that most likely explains the observed residual,
        and the change in config associated with that contact"""
        pts = torch.stack(
            [sensor.isolate_contact(ee_force_torque, pose, q=q, visualizer=visualizer) for sensor in self.sensors])
        dx = torch.stack([sensor.get_dx() for sensor in self.sensors])
        return pts, dx

    def in_contact(self):
        """Whether our last observed residual indicates that we are currently in contact"""
        if len(self.observation_history) == 0:
            return False
        for i in range(self._window_size):
            index = -1 - i
            # out of range
            if index < -len(self.observation_history):
                return False
            in_contact, ee_force_torque, prev_pose = self.observation_history[index]
            if in_contact:
                return True
        return False

    def get_last_contact_location(self, pose=None, **kwargs):
        """Get last contact point given the current end effector pose"""
        if len(self.observation_history) == 0:
            return None, None

        # allow for being in contact anytime within the latest window
        start_i = -1
        for i in range(0, min(len(self.observation_history), self._window_size)):
            in_contact, ee_force_torque, prev_pose = self.observation_history[-1 - i]
            if in_contact:
                if pose is None:
                    pose = prev_pose
                break
            start_i -= 1
        else:
            return None, None

        # use history of points to handle jitter
        ee_ft = [ee_force_torque]
        pp = [prev_pose]
        for i in range(2, self._window_size):
            from_end_i = start_i - i + 1
            if -from_end_i > len(self.observation_history):
                break
            in_contact, ee_force_torque, prev_pose = self.observation_history[from_end_i]
            # look back until we leave contact
            if not in_contact:
                break
            ee_ft.append(ee_force_torque)
            pp.append(prev_pose)
        ee_ft = torch.tensor(np.stack(ee_ft), dtype=self.dtype, device=self.device)
        # assume we don't move that much in a short amount of time and we can just use the latest pose
        pp = tuple(torch.tensor(p, dtype=self.dtype, device=self.device) for p in pp[0])
        # pos = torch.tensor([p[0] for p in pp], dtype=self.dtype)
        # orientation = torch.tensor([p[1] for p in pp], dtype=self.dtype)
        # pp = (pos, orientation)

        # in link frame
        last_contact_point, dx = self.isolate_contact(ee_ft, pp, **kwargs)
        if last_contact_point is None:
            return None, None

        link_to_current_tf = tf.Transform3d(pos=pose[0], rot=tf.xyzw_to_wxyz(torch.tensor(pose[1])),
                                            dtype=self.dtype, device=self.device)
        return link_to_current_tf.transform_points(last_contact_point), dx
