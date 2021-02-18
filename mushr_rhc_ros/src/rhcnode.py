#!/usr/bin/env python

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import cProfile
import os
import signal
import threading
import math

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from std_msgs.msg import ColorRGBA, Empty
from std_srvs.srv import Empty as SrvEmpty
from mushr_rhc_ros.srv import FollowPath
from mushr_rhc_ros.msg import XYHVPath, XYHV
from visualization_msgs.msg import Marker

import logger
import parameters
import rhcbase
import rhctensor
import utils


class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)

        self.reset_lock = threading.Lock()
        self.inferred_pose_lock = threading.Lock()
        self.pose_dist_lock = threading.Lock()
        self._inferred_pose = None
        self._pose_dist = None

        self.cur_rollout = self.cur_rollout_ip = None
        self.traj_pub_lock = threading.Lock()

        self.task_event = threading.Event()
        self.map_metadata_event = threading.Event()
        self.ready_event = threading.Event()
        self.pose_dist_event = threading.Event()
        self.events = [self.task_event, self.map_metadata_event,
                       self.ready_event, self.pose_dist_event]
        self.run = True

        self.do_profile = self.params.get_bool("profile", default=False)
        self.do_path_viz = self.params.get_bool("~do_path_viz", default=True)
        self.pose_from_dist = self.params.get_bool(
            "debug/pose_from_dist", default=False)

    def start_profile(self):
        if self.do_profile:
            self.logger.warn("Running with profiling")
            self.pr = cProfile.Profile()
            self.pr.enable()

    def end_profile(self):
        if self.do_profile:
            self.pr.disable()
            self.pr.dump_stats(os.path.expanduser("~/mushr_rhc_stats.prof"))

    def start(self):
        self.logger.info("Starting RHController")
        self.start_profile()
        self.setup_pub_sub()
        self.rhctrl = self.load_controller()
        self.T = self.params.get_int("T")
        controller = self.params.get_str("controller", default = "umpc")
        self._particles = None
        if controller == "umpc":
            self.umpc = True
        elif controller == "mpc":
            self.umpc = False
        else:
            self.logger.fatal("controller '{}' is not valid".format(cname))

        self.ready_event.set()

        rate = rospy.Rate(50)
        self.logger.info("Initialized")
        

        while not rospy.is_shutdown() and self.run:
            ip = self.inferred_pose()      
            state = ip
            
            next_traj, rollout = self.run_loop(state, ip)
            #print(next_traj[0])

            with self.traj_pub_lock:
                if rollout is not None:
                    self.cur_rollout = rollout.clone()
                    self.cur_rollout_ip = ip

            if next_traj is not None:
                self.publish_traj(next_traj, rollout)
                # For experiments. If the task is complete, notify the
                # experiment tool
                if self.rhctrl.task_complete(self.inferred_pose()):
                    self.expr_complete.publish(Empty())
                    self.task_event.clear()
            rate.sleep()

        self.end_profile()

    def run_loop(self, state, ip):
        self.task_event.wait()

        if rospy.is_shutdown() or state is None:
            return None, None
        with self.reset_lock:
            # If a reset is initialed after the task_event was set, the task
            # will be cleared. So we have to have another task check here.
            if not self.task_event.is_set():
                return None, None
            if state is not None:
                return self.rhctrl.step(state, ip)
            self.logger.err("Shouldn't get here: run_loop")

    def shutdown(self, signum, frame):
        rospy.signal_shutdown("SIGINT recieved")
        self.run = False
        for ev in self.events:
            ev.set()

    def setup_pub_sub(self):
        rospy.Service("~reset/soft", SrvEmpty, self.srv_reset_soft)
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)

        rospy.Service("~task/path", FollowPath, self.srv_path)
        car_name = self.params.get_str("car_name", default="car")

        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1
        )

        rospy.Subscriber(
            "/" + car_name + "/" + rospy.get_param("~inferred_pose_t"),
            PoseStamped,
            self.cb_pose,
            queue_size=10,
        )

    
        self.rp_ctrls = rospy.Publisher(
            "/"
            + car_name
            + "/"
            + self.params.get_str(
                "ctrl_topic", default="mux/ackermann_cmd_mux/input/navigation"
            ),
            AckermannDriveStamped,
            queue_size=2,
        )

        traj_chosen_t = self.params.get_str(
            "traj_chosen_topic", default="~traj_chosen")
        self.traj_chosen_pub = rospy.Publisher(
            traj_chosen_t, Marker, queue_size=10)

        path_viz_t = self.params.get_str("viz_path_topic", default="~path")
        self.path_viz_pub = rospy.Publisher(
            path_viz_t, PoseArray, queue_size=10)

        # For the experiment framework, need indicators to listen on
        self.expr_complete = rospy.Publisher(
            "/experiments/finished", Empty, queue_size=1
        )

    def srv_reset_hard(self, msg):
        """
        Hard reset does a complete reload of the controller
        """
        rospy.loginfo("Start hard reset")
        self.reset_lock.acquire()
        self.load_controller()
        self.task_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End hard reset")
        return []

    def srv_reset_soft(self, msg):
        """
        Soft reset only resets soft state (like tensors). No dependencies or maps
        are reloaded
        """
        rospy.loginfo("Start soft reset")
        self.reset_lock.acquire()
        self.rhctrl.reset()
        self.task_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End soft reset")
        return []

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        self.ready_event.wait()
        if "GOAL" is not self.rhctrl.cost.get_task_type():
            self.logger.err("This controller does not accept goals as a task type. \
                    Try instead " + self.rhctrl.cost.get_task_type())
            return
        if not self.rhctrl.set_task(goal):
            self.logger.err("That goal is unreachable, please choose another.")
            return
        else:
            self.logger.info("Goal set")
        self.task_event.set()

    def srv_path(self, msg):
        path = msg.path.waypoints

        self.ready_event.wait()
        if "PATH" is not self.rhctrl.cost.get_task_type():
            self.logger.err("This controller does not accept paths as a task type. \
                    Try instead " + self.rhctrl.cost.get_task_type())
            return False
        if not self.rhctrl.set_task(path):
            self.logger.err("That path is unreachable, please choose another.")
            return False
        else:
            self.logger.info("Path set")
            self.task_event.set()
            if self.do_path_viz:
                from librhc.rosviz import viz_path
                self.path_viz_pub.publish(viz_path(path))
            return True

    def cb_particles(self, msg):
        self.set_pose_dist(self.dtype(
            map(utils.rospose_to_posetup, msg.poses)))

    def cb_pose(self, msg):
        self.set_inferred_pose(self.dtype(utils.rospose_to_posetup(msg.pose)))

        if self.cur_rollout is not None and self.cur_rollout_ip is not None:
            m = Marker()
            m.header.frame_id = "map"
            m.type = m.LINE_STRIP
            m.action = m.ADD
            with self.traj_pub_lock:
                pts = (
                    self.cur_rollout[:, :2] - self.cur_rollout_ip[:2]
                ) + self.inferred_pose()[:2]

            m.points = map(lambda xy: Point(x=xy[0], y=xy[1]), pts)

            r, g, b = 0x36, 0xCD, 0xC4
            m.colors = [ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=0.7)] * len(
                m.points
            )
            m.scale.x = 0.05
            self.traj_chosen_pub.publish(m)

    def publish_traj(self, traj, rollout):
        assert traj.size() == (self.T, 2)
        assert rollout.size() == (self.T, 3)

        ctrl = traj[0]
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
        self.rp_ctrls.publish(ctrlmsg)

    def set_pose_dist(self, poses):
        with self.pose_dist_lock:
            self._particles = poses
            self.pose_dist_event.set()

    def pose_dist(self):
        self.pose_dist_event.wait()
        with self.pose_dist_lock:
            return self._particles

    def set_inferred_pose(self, ip):
        with self.inferred_pose_lock:
            if self.pose_from_dist:
                self._inferred_pose = self._expected_pose()
            else:
                self._inferred_pose = ip

    def inferred_pose(self):
        with self.inferred_pose_lock:
            return self._inferred_pose

    def _expected_pose(self):
        particles = self.pose_dist()
        if particles is None:
            return self.dtype([0, 0, 0])
        cosines = particles[:, 2].cos().mean(0)
        sines = particles[:, 2].sin().mean(0)
        theta = sines.atan2(cosines)
        position = particles[:, 0:2].mean(0)
        #position[0] += (car_length/2)*np.cos(theta)
        #position[1] += (car_length/2)*np.sin(theta)
        res = self.dtype(3)
        res[0] = position[0]
        res[1] = position[1]
        res[2] = theta
        return res


if __name__ == "__main__":
    params = parameters.RosParams()
    logger = logger.RosLog()
    node = RHCNode(rhctensor.float_tensor(), params, logger, "rhcontroller")

    signal.signal(signal.SIGINT, node.shutdown)
    rhc = threading.Thread(target=node.start)
    rhc.start()

    # wait for a signal to shutdown
    while node.run:
        signal.pause()

    rhc.join()
