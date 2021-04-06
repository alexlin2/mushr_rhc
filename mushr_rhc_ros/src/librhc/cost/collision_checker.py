import torch 
import rospy
from geometry_msgs.msg import (
    Pose,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    PoseStamped,
    Quaternion,
)

class CollisionChecker:
    def __init__(self, params, circle_radii = 0.5, car_names = ["car30","car38"]):
        self._circle_radii = circle_radii # 1 meter
        self._params = params
        self._car_names = car_names
        self._this_car = ""
        self._obstacles = {}


    def generate_obstacles(self, msg, arg):
        # TODO: you fetch poses for all the cars and you create 2d boxes by using the pose and known geometry of the car
        # push these 2d boxes to the obstacle array (exclude the ego agent bbox in this case)
        # bbox: pose + orientation + geometry -> 4 coordinates [[x1, y1], [x2, y2]...]
        # obstacles -> [[[x1,y1], [x2, y2]...] <- single car, [[]]]
        (x, y) = (msg.pose.position.x, msg.pose.position.y)
        self._obstacles[arg] = [x,y]
        
    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    # Input: paths [(nR, T, 3) tensor] -- Rollout of T positions
    #        obstacles - a list of obstacles, each obstacle represented by a list of occupied points of the form [[x1, y1], [x2, y2], ...].
    def collision_check(self, paths):
        n = paths.shape[0]
        collision_check_array = torch.zeros(n)
        obstacles = list(self._obstacles.values())
        for obs in obstacles:
            collision = torch.zeros(n)
            d = paths[:,:,:2]-torch.tensor(obs)
            collision_dists = torch.norm(d,dim = 2)
            collision[torch.any(collision_dists-self._circle_radii<0, dim=1)] = 1
            collision_check_array += collision
        return collision_check_array
        '''
        for i in range(n):
            collision_free = True
            path = paths[i]
            # Iterate over the points in the path.
            for j in range(len(path)):
                for k in range(len(obstacles)):
                    collision_dists = np.linalg.norm(obstacles[k]-np.asarray(path[j][:2])) 
                    collision_dists = np.subtract(
                        collision_dists, self._circle_radii)
                    collision_free = collision_free and not np.any(
                        collision_dists < 0)
                    if not collision_free:
                        break
                if not collision_free:
                    break
            if collision_free:
                collision_check_array[i] = 0
            else:
                collision_check_array[i] = 1
        '''
        return collision_check_array

    def run(self):
        self._this_car = self._params.get_str("car_name", default="car")
        for car in self._car_names:
            if car != self._this_car:
                rospy.Subscriber(
                    "/" + car + "/" + "car_pose",
                    PoseStamped,
                    self.generate_obstacles,
                    callback_args=car,
                    queue_size=10,
                )
