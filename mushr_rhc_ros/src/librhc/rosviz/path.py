# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header, Float32
import rospy
import numpy as np
from utils import angle_to_rosquaternion

_waypoint_pub = rospy.Publisher(
    rospy.get_param("~debug/viz_waypoint/topic", "~debug/viz_path_waypoint"),
    PoseStamped,
    queue_size=10,
)

_cte_pub = rospy.Publisher(
    rospy.get_param("~debug/log_cte/topic", "~debug/log_cte"),
    Float32,
    queue_size=10,
)

def viz_path(path):
    poses = []
    for i in range(0, len(path)):
        p = Pose()
        p.position.x = path[i].x
        p.position.y = path[i].y
        p.orientation = angle_to_rosquaternion(path[i].h)
        poses.append(p)
    pa = PoseArray()
    pa.header = Header()
    pa.header.stamp = rospy.Time.now()
    pa.header.frame_id = "map"
    pa.poses = poses
    return pa

def viz_selected_waypoint(pose):
    p = PoseStamped()
    p.header = Header() 
    p.header.stamp = rospy.Time.now()
    p.header.frame_id = "map"
    p.pose.position.x = pose[0]
    p.pose.position.y = pose[1]
    heading = pose[2] % np.pi
    heading = (heading + np.pi) % np.pi
    p.pose.orientation = angle_to_rosquaternion(pose[2])
    _waypoint_pub.publish(p)

def log_cte(cte):
    _cte_pub.publish(Float32(cte))
    