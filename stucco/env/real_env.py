# utilities for real environments that are typically using ROS
import copy
import os.path
import typing
from datetime import datetime

import rospy
from geometry_msgs.msg import Point
from rospy import ServiceException
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from window_recorder.recorder import WindowRecorder

from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest
from stucco import cfg
from stucco.env.env import Visualizer
import logging

from stucco.env.pybullet_env import DebugDrawer

logger = logging.getLogger(__name__)


class VideoLogger:
    def __init__(self, window_names=("RViz*", "RViz"), log_rviz=True, log_external_video=True):
        self.log_rviz = log_rviz
        self.log_external_video = log_external_video
        if self.log_rviz:
            self.wr = WindowRecorder(window_names=window_names, name_suffix="rviz", frame_rate=30.0,
                                     save_dir=cfg.VIDEO_DIR)

    def __enter__(self):
        logger.info("Start recording videos")
        if self.log_external_video:
            srv_name = "video_recorder"
            rospy.wait_for_service(srv_name)
            self.srv_video = rospy.ServiceProxy(srv_name, TriggerVideoRecording)
            self.req = TriggerVideoRecordingRequest()
            self.req.filename = os.path.join(cfg.VIDEO_DIR,
                                             '{}_robot.mp4'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
            self.req.timeout_in_sec = 3600
            self.req.record = True
            self.srv_video(self.req)
        if self.log_rviz:
            self.wr.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stop recording videos")
        # stop logging video
        if self.log_rviz:
            self.wr.__exit__()
        if self.log_external_video:
            if self.srv_video is not None:
                self.req.record = False
                # for some reason service will accept the request but not give a response... ignore for now
                try:
                    self.srv_video(self.req)
                except ServiceException:
                    pass


class DebugRvizDrawer(Visualizer):
    BASE_SCALE = 0.005

    def __init__(self, action_scale=0.1, max_nominal_model_error=20, world_frame="victor_root"):
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=0)
        self.action_scale = action_scale
        self.max_nom_model_error = max_nominal_model_error
        self.world_frame = world_frame
        self._ns = {}

    def _extract_ns_id_from_name(self, name):
        tokens = name.split('.')
        id = int(tokens[1]) if len(tokens) == 2 else 0
        return tokens[0], id

    def draw_point(self, name, point, color=(0, 0, 0), length=0.01, length_ratio=1, rot=0, height=None, label=None,
                   scale=1):
        ns, this_id = self._extract_ns_id_from_name(name)
        marker = self.make_marker(ns, scale=self.BASE_SCALE * scale, id=this_id)
        z = height if height is not None else point[2]

        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = z
        c = ColorRGBA()
        c.a = 1
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        marker.colors.append(c)
        marker.points.append(p)
        self.marker_pub.publish(marker)
        if label is not None:
            self.draw_text(f"{ns}_text", label, [point[0], point[1], z], id=this_id, absolute_pos=True)

        return p

    def draw_2d_pose(self, name, pose, color=(0, 0, 0), length=0.15 / 2, height=None):
        pass

    def draw_2d_line(self, name, start, diff, color=(0, 0, 0), size=2., scale=0.4, arrow=True):
        ns, this_id = self._extract_ns_id_from_name(name)
        marker = self.make_marker(ns, marker_type=Marker.ARROW if arrow else Marker.LINE_LIST, id=int(this_id))
        z = start[2] if len(start) > 2 else 0

        p = Point()
        p.x = start[0]
        p.y = start[1]
        p.z = z
        marker.points.append(p)
        p = Point()
        p.x = start[0] + diff[0] * scale
        p.y = start[1] + diff[1] * scale
        p.z = start[2] + diff[2] * scale if len(diff) > 2 else z
        marker.points.append(p)

        c = ColorRGBA()
        c.a = 1
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        if arrow:
            marker.color = c
        else:
            marker.colors.append(c)
            marker.colors.append(c)
        self.marker_pub.publish(marker)
        return p

    def make_marker(self, ns, scale=BASE_SCALE, marker_type=Marker.POINTS, adding_ns=True, id=0):
        marker = Marker()
        marker.ns = ns
        marker.id = id
        if adding_ns:
            if ns not in self._ns:
                self._ns[ns] = set()
            self._ns[ns].add(id)
        elif ns in self._ns:
            if ns in self._ns:
                self._ns[ns].remove(id)
                if len(self._ns[ns]) == 0:
                    del self._ns[ns]
        marker.header.frame_id = self.world_frame
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        return marker

    def draw_state(self, state, time_step, nominal_model_error=0, action=None, height=None):
        z = height if height is not None else state[2]
        p = self.draw_point("state_trajectory.{}".format(time_step), state, (
            0, 1.0 * max(0, self.max_nom_model_error - nominal_model_error) / self.max_nom_model_error, 0),
                            height=height)
        if action is not None:
            action_marker = self.make_marker("action", marker_type=Marker.LINE_LIST)
            action_marker.points.append(p)
            p = Point()
            p.x = state[0] + action[0] * self.action_scale
            p.y = state[1] + action[1] * self.action_scale
            p.z = z
            action_marker.points.append(p)

            c = ColorRGBA()
            c.a = 1
            c.r = 1
            c.g = 0
            c.b = 0
            action_marker.colors.append(c)
            action_marker.colors.append(c)
            self.marker_pub.publish(action_marker)

    def draw_goal(self, goal):
        marker = self.make_marker("goal", scale=self.BASE_SCALE * 2)
        p = Point()
        p.x = goal[0]
        p.y = goal[1]
        p.z = goal[2]
        c = ColorRGBA()
        c.a = 1
        c.r = 1
        c.g = 0.8
        c.b = 0
        marker.colors.append(c)
        marker.points.append(p)
        self.marker_pub.publish(marker)

    def draw_rollouts(self, rollouts):
        if rollouts is None:
            return
        marker = self.make_marker("rollouts")
        # assume states is iterable, so could be a bunch of row vectors
        T = len(rollouts)
        for t in range(T):
            cc = (t + 1) / (T + 1)
            p = Point()
            p.x = rollouts[t][0]
            p.y = rollouts[t][1]
            p.z = rollouts[t][2]
            c = ColorRGBA()
            c.a = 1
            c.r = 0
            c.g = cc
            c.b = cc
            marker.colors.append(c)
            marker.points.append(p)
        self.marker_pub.publish(marker)

    def draw_trap_set(self, trap_set):
        if trap_set is None:
            return
        state_marker = self.make_marker("trap_state", scale=self.BASE_SCALE * 2)
        action_marker = self.make_marker("trap_action", marker_type=Marker.LINE_LIST)

        T = len(trap_set)
        for t in range(T):
            action = None
            if len(trap_set[t]) == 2:
                state, action = trap_set[t]
            else:
                state = trap_set[t]

            p = Point()
            p.x = state[0]
            p.y = state[1]
            p.z = state[2]
            state_marker.points.append(p)
            if action is not None:
                action_marker.points.append(p)
                p = Point()
                p.x = state[0] + action[0] * self.action_scale
                p.y = state[1] + action[1] * self.action_scale
                p.z = state[2]
                action_marker.points.append(p)

            cc = (t + 1) / (T + 1)
            c = ColorRGBA()
            c.a = 1
            c.r = 1
            c.g = 0
            c.b = cc
            state_marker.colors.append(c)
            if action is not None:
                action_marker.colors.append(c)
                action_marker.colors.append(c)

        self.marker_pub.publish(state_marker)
        self.marker_pub.publish(action_marker)

    def clear_visualizations(self, names=None):
        if names is None:
            names = list(self._ns)
        for name in names:
            self.clear_markers(name)

    def clear_visualization_after(self, prefix, index):
        ns = self._ns.get(prefix, None)
        if ns not in self._ns:
            return
        ns = copy.deepcopy(ns)

        while index in ns:
            self.clear_markers(prefix, delete_all=False, id=index)
            index += 1

    def clear_markers(self, ns, delete_all=True, id=0):
        marker = self.make_marker(ns, adding_ns=False, id=id)
        marker.action = Marker.DELETEALL if delete_all else Marker.DELETE
        self.marker_pub.publish(marker)

    def draw_text(self, label, text, offset, left_offset=0, scale=5, id=0, absolute_pos=False):
        marker = self.make_marker(label, marker_type=Marker.TEXT_VIEW_FACING, scale=self.BASE_SCALE * scale)
        marker.id = id
        marker.text = text

        if absolute_pos:
            marker.pose.position.x = offset[0]
            marker.pose.position.y = offset[1]
            marker.pose.position.z = offset[2] if len(offset) > 2 else 1
        else:
            marker.pose.position.x = 1.4 + offset * self.BASE_SCALE * 6
            marker.pose.position.y = 0.4 + left_offset * 0.5
            marker.pose.position.z = 1
        marker.pose.orientation.w = 1

        marker.color.a = 1
        marker.color.r = 0.8
        marker.color.g = 0.3

        self.marker_pub.publish(marker)

    def draw_transition(self, x, new_x):
        pass

    def draw_mesh(self, name, model, pose, rgba=(0, 0, 0, 1.), scale=1., object_id=None, vis_frame_pos=(0, 0, 0),
                  vis_frame_rot=(0, 0, 0, 1)):
        if object_id is None:
            if name not in self._ns:
                object_id = 0
            else:
                object_id = max(self._ns[name]) + 1

        marker = self.make_marker(name, marker_type=Marker.MESH_RESOURCE, scale=scale, id=object_id)
        # sanitize resource link
        marker.mesh_resource = cfg.ensure_rviz_resource_path(model)
        marker.mesh_use_embedded_materials = True

        pos, rot = pose
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        # rot is given in xyzw
        marker.pose.orientation.x = rot[0]
        marker.pose.orientation.y = rot[1]
        marker.pose.orientation.z = rot[2]
        marker.pose.orientation.w = rot[3]

        self.marker_pub.publish(marker)
        rospy.sleep(0.1)
        self.marker_pub.publish(marker)


class CombinedVisualizer(Visualizer):
    def __init__(self):
        self.ros: typing.Optional[DebugRvizDrawer] = None
        self.sim: typing.Optional[DebugDrawer] = None

    def init_sim(self, *args, **kwargs):
        self.sim = DebugDrawer(*args, **kwargs)

    def init_ros(self, *args, **kwargs):
        self.ros = DebugRvizDrawer(*args, **kwargs)

    def draw_point(self, *args, **kwargs):
        if self.sim is not None:
            self.sim.draw_point(*args, **kwargs)
        if self.ros is not None:
            self.ros.draw_point(*args, **kwargs)

    def draw_2d_line(self, *args, **kwargs):
        if self.sim is not None:
            self.sim.draw_2d_line(*args, **kwargs)
        if self.ros is not None:
            self.ros.draw_2d_line(*args, **kwargs)

    def draw_2d_pose(self, *args, **kwargs):
        if self.sim is not None:
            self.sim.draw_2d_pose(*args, **kwargs)
        if self.ros is not None:
            self.ros.draw_2d_pose(*args, **kwargs)

    def clear_visualizations(self, names=None):
        if self.sim is not None:
            self.sim.clear_visualizations(names)
        if self.ros is not None:
            self.ros.clear_visualizations(names)

    def clear_visualization_after(self, prefix, index):
        if self.sim is not None:
            self.sim.clear_visualization_after(prefix, index)
        if self.ros is not None:
            self.ros.clear_visualization_after(prefix, index)

    def draw_transition(self, x, new_x):
        if self.sim is not None:
            self.sim.draw_transition(x, new_x)
        if self.ros is not None:
            self.ros.draw_transition(x, new_x)

    def draw_mesh(self, *args, **kwargs):
        if self.sim is not None:
            self.sim.draw_mesh(*args, **kwargs)
        if self.ros is not None:
            self.ros.draw_mesh(*args, **kwargs)
