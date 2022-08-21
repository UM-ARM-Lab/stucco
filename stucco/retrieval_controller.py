import abc
import typing

import numpy as np
import pybullet as p
import torch
from arm_pytorch_utilities import rand
from arm_pytorch_utilities.math_utils import angular_diff
from arm_pytorch_utilities.controller import Controller
from pynput import keyboard

from stucco import detection, tracking
from stucco import exploration
from stucco.defines import NO_CONTACT_ID
from stucco import cfg
from stucco.env.pybullet_env import closest_point_on_surface, ContactInfo, state_action_color_pairs

from stucco.baselines.cluster import process_labels_with_noise
from stucco.baselines.gmphd import GMPHDWrapper

from arm_pytorch_utilities.draw import clear_ax_content
from datetime import datetime
import matplotlib.pyplot as plt
import os
import subprocess
import glob

from stucco.util import move_figure
from stucco.env.env import Visualizer, InfoKeys


class RetrievalController(Controller):

    def __init__(self, contact_detector: detection.ContactDetector, nu, dynamics, cost_to_go,
                 contact_set: tracking.ContactSetHard, u_min, u_max, num_samples=100,
                 walk_length=3):
        super().__init__()
        self.contact_detector = contact_detector
        self.nu = nu
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics = dynamics
        self.cost = cost_to_go
        self.num_samples = num_samples

        self.max_walk_length = walk_length
        self.remaining_random_actions = 0

        self.x_history = []
        self.u_history = []

        self.contact_set = contact_set

    def command(self, obs, info=None):
        d = self.dynamics.device
        dtype = self.dynamics.dtype

        self.x_history.append(obs)

        if self.contact_detector.in_contact():
            self.remaining_random_actions = self.max_walk_length
            x = self.x_history[-1][:2]
            pt, dx = self.contact_detector.get_last_contact_location()
            info['u'] = torch.tensor(self.u_history[-1][:2])
            self.contact_set.update(x, dx, pt, info=info)

        if self.remaining_random_actions > 0:
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=self.nu)
            self.remaining_random_actions -= 1
        else:
            # take greedy action if not in contact
            state = torch.from_numpy(obs).to(device=d, dtype=dtype).repeat(self.num_samples, 1)
            u = np.random.uniform(low=self.u_min, high=self.u_max, size=(self.num_samples, self.nu))
            u = torch.from_numpy(u).to(device=d, dtype=dtype)

            next_state = self.dynamics(state, u)
            costs = self.cost(torch.from_numpy(self.goal).to(device=d, dtype=dtype), next_state)
            min_i = torch.argmin(costs)
            u = u[min_i].cpu().numpy()

        self.u_history.append(u)
        return u


class RetrievalPredeterminedController(Controller):

    def __init__(self, controls, nu=None):
        super().__init__()
        self.controls = controls
        self.i = 0
        self.nu = nu or len(self.controls[0])

        self.x_history = []
        self.u_history = []

    def done(self):
        return self.i >= len(self.controls)

    def insert_next_controls(self, controls):
        self.controls = self.controls[:self.i] + controls + self.controls[self.i:]

    @abc.abstractmethod
    def update(self, obs, info, visualizer=None):
        pass

    def command(self, obs, info=None, visualizer=None):
        self.x_history.append(obs)

        if len(self.x_history) > 1:
            self.update(obs, info, visualizer=visualizer)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = self.controls[self.i]
            self.i += 1

        self.u_history.append(u)
        return u


class OursRetrievalPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet, controls,
                 nu=None):
        super().__init__(controls, nu=nu)
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.contact_indices = []

    def update(self, obs, info, visualizer=None):
        if self.contact_detector.in_contact():
            self.contact_indices.append(self.i)

        x = self.x_history[-1][:2]
        pt, dx = self.contact_detector.get_last_contact_location(visualizer=visualizer)
        # from stucco.tracking import InfoKeys
        # if dx is not None:
        #     assert torch.allclose(dx, torch.tensor(info[InfoKeys.DEE_IN_CONTACT][:2], dtype=dx.dtype, device=dx.device))
        info['u'] = torch.tensor(self.u_history[-1])
        self.contact_set.update(x, dx, pt, info=info)


class OursRetrievalICPEVController(Controller):
    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet,
                 policy: exploration.ShapeExplorationPolicy, method, u_mag_max, max_steps=200, vis: Visualizer = None,
                 real_to_control_magnitude=1):

        self.u_mag_max = u_mag_max
        self.max_steps = max_steps
        self.method = method
        self.vis = vis
        self.real_to_control_magnitude = real_to_control_magnitude

        self.policy = policy
        self.next_target = None
        self.next_target_close_enough = 0.01  # points and normals
        self.x_history = []
        self.n_history = []

        self.u_history = []

        super().__init__()
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.contact_indices = []

    def done(self):
        return len(self.x_history) >= self.max_steps

    def command(self, obs, info=None, visualizer=None):
        z = 0.1
        # TODO add device and dtype
        x = torch.tensor(obs)
        # get surface normal
        self.x_history.append(obs)
        if info is None:
            self.n_history.append(np.zeros_like(obs))
        else:
            n = info[InfoKeys.LOW_FREQ_REACTION_F]
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
            self.n_history.append(n)

        if len(self.x_history) > 1:
            self.update(obs, info, visualizer=visualizer)

        if self.next_target is not None:
            # decide if we're close enough to target
            diff = self.next_target - x
            going_to_previous_target = diff.norm() > self.next_target_close_enough
            if going_to_previous_target:
                u = diff / self.real_to_control_magnitude * 1.2  # need a little kick
                return u

        # else figure out next target
        if info is None:
            u = np.random.normal(np.zeros_like(obs), np.ones_like(obs) * self.u_mag_max)
        else:
            # TODO only pass in xs and df of best guess for ICP
            # TODO if ICP results are not good enough take a random action
            xs = self.x_history
            df = self.n_history
            t = len(self.x_history)
            self.policy.start_step(xs, df)
            target_dx = self.policy.get_next_dx(xs, df, t)
            self.policy.end_step(xs, df, t)

            target_x = x + target_dx[:2]
            self.next_target = target_x
            if self.vis is not None:
                self.vis.draw_point("target point", [target_x[0], target_x[1], z], color=(0.5, 0.2, 0.8))

            # TODO convert target to a sequence of cached controls; first take a step in the normal direction,
            n = self.n_history[-1]
            u = n * self.u_mag_max

        self.u_history.append(u)
        return u

    def update(self, obs, info, visualizer=None):
        if self.contact_detector.in_contact():
            self.contact_indices.append(len(self.x_history) - 1)

        x = self.x_history[-1][:2]
        pt, dx = self.contact_detector.get_last_contact_location(visualizer=visualizer)
        # if dx is not None:
        #     assert torch.allclose(dx, torch.tensor(info[InfoKeys.DEE_IN_CONTACT][:2], dtype=dx.dtype, device=dx.device))
        info['u'] = torch.tensor(self.u_history[-1])
        self.contact_set.update(x, dx, pt, info=info)


def rot_2d_mat_to_angle(T):
    """T: bx3x3 homogenous transforms or bx2x2 rotation matrices"""
    return torch.atan2(T[:, 1, 0], T[:, 0, 0])


def sample_model_points(object_id=None, num_points=100, reject_too_close=0.002, force_z=None, mid_z=0, seed=0, name="",
                        sample_in_order=False, clean_cache=False, random_sample_sigma=0.1, vis: Visualizer = None,
                        restricted_points=[], other_rejection_criteria=None, device="cpu"):
    fullname = os.path.join(cfg.DATA_DIR, f'model_points_cache.pkl')
    if os.path.exists(fullname):
        cache = torch.load(fullname)
        if name not in cache:
            cache[name] = {}
        if seed not in cache[name]:
            cache[name][seed] = {}
        if not clean_cache and num_points in cache[name][seed]:
            res = cache[name][seed][num_points]
            return (v.to(device=device) for v in res)
    else:
        cache = {name: {seed: {}}}

    if object_id is None:
        raise RuntimeError(f"Expect model points to be cached for {name} {seed} {num_points}")

    def retrieve_valid_surface_pt(tester_pos, points):
        closest = closest_point_on_surface(object_id, tester_pos)
        pt = closest[ContactInfo.POS_A]
        normal = closest[ContactInfo.NORMAL_DIR_B]
        if force_z is not None:
            pt = (pt[0], pt[1], force_z)
            normal = (normal[0], normal[1], 0)
        if other_rejection_criteria is not None and other_rejection_criteria(pt, normal):
            return None, None
        if len(points) > 0:
            d = np.subtract(points + restricted_points, pt)
            d = np.linalg.norm(d, axis=1)
            if np.any(d < reject_too_close):
                return None, None
        return pt, normal

    with rand.SavedRNG():
        rand.seed(seed)
        orig_pos, orig_orientation = p.getBasePositionAndOrientation(object_id)
        # first reset to canonical location
        canonical_pos = [0, 0, 0]
        p.resetBasePositionAndOrientation(object_id, canonical_pos, [0, 0, 0, 1])

        # box is rectilinear, so can just get bounding box
        aabb_min, aabb_max = p.getAABB(object_id)
        bb = np.zeros((4, 3))
        # 2D planar bounding box
        bb[0] = [aabb_min[0], aabb_min[1], 1]
        bb[1] = [aabb_min[0], aabb_max[1], 1]
        bb[2] = [aabb_max[0], aabb_max[1], 1]
        bb[3] = [aabb_max[0], aabb_min[1], 1]

        points = []
        normals = []
        if sample_in_order:
            r = max(aabb_max[0], aabb_max[1])
            # sample evenly in terms of angles, but leave out the section in between the fingers
            leave_out = 0
            start_angle = 0
            angles = np.linspace(start_angle + leave_out, np.pi * 2 - leave_out + start_angle, num_points)

            for angle in angles:
                tester_pos = [np.cos(angle) * r, np.sin(angle) * r, 0]
                pt, normal = retrieve_valid_surface_pt(tester_pos, points)
                if pt is None:
                    continue
                points.append(pt)
                normals.append(normal)
        else:
            while len(points) < num_points:
                tester_pos = np.random.randn(3) * random_sample_sigma
                if force_z is not None:
                    tester_pos[2] = force_z
                else:
                    tester_pos[2] += mid_z
                pt, normal = retrieve_valid_surface_pt(tester_pos, points)
                if pt is None:
                    continue
                if vis is not None:
                    vis.draw_point(f"tpt", tester_pos, color=(1, 0, 0), length=0.005)
                    vis.draw_point(f"mpt.{len(points)}", pt, color=(0, 0, 1), length=0.003)
                    vis.draw_2d_line(f"mn.{len(points)}", pt, normal, color=(0, 0, 0), size=2., scale=0.03)
                points.append(pt)
                normals.append(normal)

        if vis is not None:
            input('enter to finish visualization')
            vis.clear_visualizations()

    p.resetBasePositionAndOrientation(object_id, orig_pos, orig_orientation)

    # if sampled in order, the ordered points themselves can be used for visualization, otherwise use the bounding box
    points = torch.tensor(points)
    normals = torch.tensor(normals)
    if sample_in_order:
        # reduce fidelity to speed up drawing of estimated pose
        bb = points.clone()[::3]
        bb = torch.cat((bb, points[0].view(1, -1)))
        bb[:, -1] = 1
    else:
        bb = torch.tensor(bb)

    cache[name][seed][num_points] = points, normals, bb
    torch.save(cache, fullname)

    return points.to(device=device), normals.to(device=device), bb.to(device=device)


def pose_error(target_pose, guess_pose):
    # mirrored, so being off by 180 degrees is fine
    yaw_error = min(abs(angular_diff(target_pose[-1], guess_pose[-1])),
                    abs(angular_diff(target_pose[-1] + np.pi, guess_pose[-1])))
    pos_error = np.linalg.norm(np.subtract(target_pose[:2], guess_pose[:2]))
    return pos_error, yaw_error


class TrackingMethod:
    """Common interface for each tracking method including ours and baselines"""

    @abc.abstractmethod
    def __iter__(self):
        """Iterating over this provides a set of contact points corresponding to an object"""

    @abc.abstractmethod
    def create_controller(self, controls):
        """Return a predetermined controller that updates the method when querying for a command"""

    @abc.abstractmethod
    def visualize_contact_points(self, env):
        """Render the tracked contact points in the given environment"""

    @abc.abstractmethod
    def get_labelled_moved_points(self, labels=None):
        """Return the final position of the tracked points as well as their object label"""

    def register_transforms(self, T, best_T):
        pass


class SoftTrackingIterator:
    def __init__(self, pts, to_iter):
        self.pts = pts
        self.to_iter = to_iter

    def __next__(self):
        indices = next(self.to_iter)
        return self.pts[indices]


class OurTrackingMethod(TrackingMethod):
    def __init__(self, env):
        self.env = env
        self.ctrl = None

    @property
    @abc.abstractmethod
    def contact_set(self) -> tracking.ContactSet:
        """Return some contact set"""

    def visualize_contact_points(self, env):
        env.visualize_contact_set(self.contact_set)

    def create_controller(self, controls):
        self.ctrl = OursRetrievalPredeterminedController(self.env.contact_detector, self.contact_set, controls)
        return self.ctrl


class OurSoftTrackingMethod(OurTrackingMethod):
    def __init__(self, env, contact_params, pt_to_config):
        self.contact_params = contact_params
        self._contact_set = tracking.ContactSetSoft(pt_to_config, self.contact_params)
        super(OurSoftTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetSoft:
        return self._contact_set

    def __iter__(self):
        pts = self.contact_set.get_posterior_points()
        to_iter = self.contact_set.get_hard_assignment(self.contact_set.p.hard_assignment_threshold)
        return SoftTrackingIterator(pts, iter(to_iter))

    def get_labelled_moved_points(self, labels=None):
        contact_pts = self.contact_set.get_posterior_points()
        if labels is not None:
            groups = self.contact_set.get_hard_assignment(self.contact_params.hard_assignment_threshold)
            contact_indices = torch.tensor(self.ctrl.contact_indices, device=groups[0].device)
            # self.ctrl.contact_indices = []

            for group_id, group in enumerate(groups):
                labels[contact_indices[group].cpu().numpy()] = group_id + 1

        return labels, contact_pts


class OurSoftTrackingWithRummagingMethod(OurSoftTrackingMethod):
    def __init__(self, *args, policy_factory: typing.Callable[..., exploration.ShapeExplorationPolicy] = None,
                 **kwargs):
        self.policy_factory = policy_factory
        super(OurSoftTrackingWithRummagingMethod, self).__init__(*args, **kwargs)

    def create_controller(self, controls):
        self.policy = self.policy_factory()
        self.ctrl = OursRetrievalICPEVController(self.env.contact_detector, self.contact_set, self.policy, self, 1,
                                                 max_steps=200, vis=self.env.vis,
                                                 real_to_control_magnitude=self.env.MAX_PUSH_DIST)
        return self.ctrl

    def register_transforms(self, T, best_tsf_guess):
        self.policy.register_transforms(T, best_tsf_guess)


class HardTrackingIterator:
    def __init__(self, contact_objs):
        self.contact_objs = contact_objs

    def __next__(self):
        object: tracking.ContactObject = next(self.contact_objs)
        return object.points


class OurHardTrackingMethod(OurTrackingMethod):
    def __init__(self, env, contact_params, hard_contact_params):
        self.contact_params = contact_params
        self.hard_contact_params = hard_contact_params
        self._contact_set = tracking.ContactSetHard(self.contact_params, hard_params=hard_contact_params,
                                                    contact_object_factory=self.create_contact_object)
        super(OurHardTrackingMethod, self).__init__(env)

    @property
    def contact_set(self) -> tracking.ContactSetHard:
        return self._contact_set

    def __iter__(self):
        return HardTrackingIterator(iter(self.contact_set))

    def create_contact_object(self):
        return tracking.ContactUKF(None, self.contact_params, self.hard_contact_params)


class SklearnPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, online_method, contact_detector: detection.ContactDetector, controls, nu=None):
        super().__init__(controls, nu=nu)
        self.online_method = online_method
        self.contact_detector = contact_detector
        self.in_contact = []

    def update(self, obs, info, visualizer=None):
        contact_point, dobj = self.contact_detector.get_last_contact_location(visualizer=visualizer)
        if contact_point is not None:
            self.in_contact.append(True)
            contact_point = contact_point.cpu().numpy()[:, :2]
            dobj = dobj.cpu().numpy()
            self.online_method.update(contact_point - dobj, self.u_history[-1], dobj)
        else:
            self.in_contact.append(False)


class CommonBaselineTrackingMethod(TrackingMethod):
    @property
    @abc.abstractmethod
    def method(self):
        """Get method that can provide final_labels() and moved_data()"""

    def __iter__(self):
        moved_pt_labels = process_labels_with_noise(self.method.final_labels())
        moved_pts = self.method.moved_data()
        # labels[valid] = moved_pt_labels
        groups = []
        for i, obj_id in enumerate(np.unique(moved_pt_labels)):
            if obj_id == NO_CONTACT_ID:
                continue

            indices = moved_pt_labels == obj_id
            groups.append(moved_pts[indices])

        return iter(groups)

    def visualize_contact_points(self, env):
        moved_pt_labels = process_labels_with_noise(self.method.final_labels())
        moved_pts = self.method.moved_data()
        i = 0
        for i, obj_id in enumerate(np.unique(moved_pt_labels)):
            if obj_id == NO_CONTACT_ID:
                continue

            indices = moved_pt_labels == obj_id
            color, u_color = state_action_color_pairs[i % len(state_action_color_pairs)]
            base_name = str(i)
            env.visualize_state_actions(base_name, moved_pts[indices], None, color, u_color, 0)
        for j in range(i + 1, 10):
            env.visualize_state_actions(str(j), [], None, None, None, 0)


class SklearnTrackingMethod(CommonBaselineTrackingMethod):
    def __init__(self, env, online_class, method, inertia_ratio=0.5, **kwargs):
        self.env = env
        self.online_method = online_class(method(**kwargs), inertia_ratio=inertia_ratio)
        self.ctrl: typing.Optional[SklearnPredeterminedController] = None

    @property
    def method(self):
        return self.online_method

    def create_controller(self, controls):
        self.ctrl = SklearnPredeterminedController(self.online_method, self.env.contact_detector, controls, nu=2)
        return self.ctrl

    def get_labelled_moved_points(self, labels=None):
        moved_pts = self.method.moved_data()
        if labels is not None:
            labels[1:][self.ctrl.in_contact] = process_labels_with_noise(self.method.final_labels())
        return labels, moved_pts


class PHDPredeterminedController(RetrievalPredeterminedController):

    def __init__(self, g, contact_detector: detection.ContactDetector, controls, nu=None):
        super().__init__(controls, nu=nu)
        self.g = g
        self.contact_detector = contact_detector
        self.in_contact = []

    def update(self, obs, info, visualizer=None):
        contact_point, dobj = self.contact_detector.get_last_contact_location(visualizer=visualizer)
        if contact_point is not None:
            self.in_contact.append(True)
            contact_point = contact_point.cpu().numpy()[:, :2]
            dobj = dobj.cpu().numpy()
            pt = contact_point - dobj
            self.g.update(pt, dobj)
        else:
            self.in_contact.append(False)


class PHDFilterTrackingMethod(CommonBaselineTrackingMethod):
    def __init__(self, env, **kwargs):
        self.env = env
        self.g = GMPHDWrapper(**kwargs, bounds=(-1, -1, 1, 1))
        self.ctrl = None

        self.i = 0
        self.tmp_save_folder = os.path.join(cfg.DATA_DIR, "gmphd")
        os.makedirs(self.tmp_save_folder, exist_ok=True)
        self.f = None

    @property
    def method(self):
        return self.g

    def create_controller(self, controls):
        self.ctrl = PHDPredeterminedController(self.g, self.env.contact_detector, controls, nu=2)
        return self.ctrl

    def visualize_contact_points(self, env):
        super(PHDFilterTrackingMethod, self).visualize_contact_points(env)
        if self.f is None:
            self.f = plt.figure(figsize=(8, 4))
            move_figure(self.f, 0, 1000)

            self.ax = plt.gca()
            self.ax.set_ylim(0., 0.8)
            self.ax.set_xlim(-0.8, 0.8)
            self.ax.set_aspect('equal')
            plt.ion()
            plt.show()

        # visualize the intensity
        clear_ax_content(self.ax)

        x = np.linspace(0., 0.8)
        y = np.linspace(-.8, .8)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = self.g.g.gmmeval(XX.reshape(-1, 2, 1))
        Z = np.stack(Z).astype(float).reshape(X.shape)
        # axes corresponds to the debug camera in pybullet
        plt.contour(-Y, X, Z)
        # axes corresponding to rviz for real robot
        # plt.contour(Y, 1 - X, Z)

        self.f.canvas.draw()
        plt.pause(0.0001)
        plt.savefig(self.tmp_save_folder + "/file{:02d}.png".format(self.i))
        self.i += 1

    def get_labelled_moved_points(self, labels=None):
        moved_pts = self.method.moved_data()
        if labels is not None:
            labels[1:][self.ctrl.in_contact] = process_labels_with_noise(self.method.final_labels())

        # save intensity plot images to video
        plt.close(self.f)
        os.chdir(self.tmp_save_folder)
        subprocess.call([
            'ffmpeg', '-framerate', '4', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.mp4"
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

        return labels, moved_pts


class KeyboardDirPressed:
    def __init__(self):
        self._dir = [0, 0]
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.calibrate = False
        self.esc = False

    @property
    def dir(self):
        return self._dir

    def on_press(self, key):
        if key == keyboard.Key.down:
            self.dir[1] = -1
        elif key == keyboard.Key.left:
            self.dir[0] = -1
        elif key == keyboard.Key.up:
            self.dir[1] = 1
        elif key == keyboard.Key.right:
            self.dir[0] = 1
        elif key == keyboard.Key.shift:
            self.calibrate = True
        elif key == keyboard.Key.esc:
            self.esc = True

    def on_release(self, key):
        if key in [keyboard.Key.down, keyboard.Key.up]:
            self.dir[1] = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.dir[0] = 0
        elif key == keyboard.Key.shift:
            self.calibrate = False


class KeyboardController(Controller):

    def __init__(self, contact_detector: detection.ContactDetector, contact_set: tracking.ContactSet, nu=2):
        super().__init__()
        self.pushed = KeyboardDirPressed()
        self.contact_detector = contact_detector
        self.contact_set = contact_set
        self.nu = nu

        self.x_history = []
        self.u_history = []

    def done(self):
        return self.pushed.esc

    @abc.abstractmethod
    def update(self, obs, info):
        x = self.x_history[-1][:2]
        info['u'] = torch.tensor(self.u_history[-1])
        pt, dx = self.contact_detector.get_last_contact_location()
        self.contact_set.update(x, dx, pt, info=info)

    def command(self, obs, info=None):
        self.x_history.append(obs)

        if self.done():
            u = [0 for _ in range(self.nu)]
        else:
            u = tuple(self.pushed.dir)

        if len(self.x_history) > 1 and self.u_history[-1] != (0, 0):
            self.update(obs, info)

        self.u_history.append(u)
        return u
