import pybullet as p
from arm_pytorch_utilities import tensor_utils

from pytorch_kinematics import transforms as tf
from stucco.detection import ContactSensor
import enum


class ContactInfo(enum.IntEnum):
    """Semantics for indices of a contact info from getContactPoints"""
    LINK_A = 3
    LINK_B = 4
    POS_A = 5
    POS_B = 6
    NORMAL_DIR_B = 7
    DISTANCE = 8
    NORMAL_MAG = 9
    LATERAL1_MAG = 10
    LATERAL1_DIR = 11
    LATERAL2_MAG = 12
    LATERAL2_DIR = 13


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
