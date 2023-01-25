import enum
import itertools

import numpy as np
import torch
from arm_pytorch_utilities.controller import Controller
from stucco import detection, tracking


class PokingController(Controller):
    class Mode(enum.Enum):
        GO_TO_NEXT_TARGET = 0
        PUSH_FORWARD = 1
        RETURN_BACKWARD = 2
        DONE = 3

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet,
                 y_order=(0, 0.2, 0.3, -0.2, -0.3), z_order=(0.05, 0.15, 0.25, 0.325, 0.4, 0.5), x_rest=-0.05,
                 Kp=30,
                 push_forward_count=10, nu=3, dim=3, goal_tolerance=3e-4):
        super().__init__()

        self.x_rest = x_rest
        self._all_targets = list(itertools.product(y_order, z_order))
        self.target_yz = self._all_targets
        self.push_forward_count = push_forward_count
        self.mode = self.Mode.GO_TO_NEXT_TARGET
        self.kp = Kp

        self.goal_tolerance = goal_tolerance

        # primitive state machine where we push forward for push_forward_count, then go back to x_rest
        self.push_i = 0
        self.i = 0
        self.current_target = None

        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.dim = dim
        self.contact_indices = []

        self.nu = nu

        self.x_history = []
        self.u_history = []

    def reset(self):
        self.target_yz = self._all_targets
        self.mode = self.Mode.GO_TO_NEXT_TARGET
        self.contact_indices = []
        self.x_history = []
        self.u_history = []
        self.push_i = 0
        self.i = 0
        self.current_target = None

    def update(self, obs, info, visualizer=None):
        if self.contact_detector.in_contact():
            self.contact_indices.append(self.i)

        x = self.x_history[-1][:self.dim]
        pt, dx = self.contact_detector.get_last_contact_location(visualizer=visualizer)

        if info is not None:
            info['u'] = torch.tensor(self.u_history[-1])
            self.contact_set.update(x, dx, pt, info=info)

    def done(self):
        return len(self.target_yz) == 0

    def command(self, obs, info=None, visualizer=None):
        u = [0 for _ in range(self.nu)]
        if info is None:
            return u
        self.x_history.append(obs)

        if len(self.x_history) > 1:
            self.update(obs, info, visualizer=visualizer)

        if self.done():
            self.mode = self.Mode.DONE
        else:
            if self.mode == self.Mode.GO_TO_NEXT_TARGET:
                # go to next target proportionally
                target = self.target_yz[0]
                diff = np.array(target) - np.array(obs[1:])
                # TODO clamp?
                u[1] = diff[0] * self.kp
                u[2] = diff[1] * self.kp

                if np.linalg.norm(diff) < self.goal_tolerance:
                    self.mode = self.Mode.PUSH_FORWARD
                    self.push_i = 0

            # if we don't have a current target, find the next
            elif self.mode == self.Mode.PUSH_FORWARD:
                u[0] = 1.
                self.push_i += 1
                if self.push_i >= self.push_forward_count or np.linalg.norm(info['reaction']) > 5:
                    self.mode = self.Mode.RETURN_BACKWARD
            elif self.mode == self.Mode.RETURN_BACKWARD:
                diff = self.x_rest - obs[0]
                u[0] = diff * self.kp

                if abs(diff) < self.goal_tolerance:
                    self.mode = self.Mode.GO_TO_NEXT_TARGET
                    self.target_yz = self.target_yz[1:]
                    u = None

            self.i += 1

        if u is not None:
            self.u_history.append(u)
        return u
