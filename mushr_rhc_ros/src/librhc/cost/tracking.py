import torch

import threading



class Tracking:
    NPOS = 3  # x, y, theta
    TASK_TYPE = "PATH"

    def __init__(self, params, logger, dtype, map, world_rep, value_fn):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.viz_waypoint = self.params.get_bool("debug/viz_waypoint", True)
        self.do_log_cte = self.params.get_bool("debug/log_cte", True)
        self.viz_rollouts = self.params.get_bool("debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool("debug/viz_rollouts/print_stats", False)

        self.path = None
        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.P = self.params.get_int("P", default=1)
        controller = self.params.get_str("controller", default="umpc")
        if controller == 'umpc':
            self.nR = self.K * self.P # num rollouts
        else:
            self.nR = self.K # num rollouts

        self.finish_threshold = self.params.get_float("cost_fn/finish_threshold", 0.5)
        self.exceed_threshold = self.params.get_float("cost_fn/exceed_threshold", 4.0)
        self.collision_check  = self.params.get_bool("cost_fn/collision_check", True) 

        self.lookahead = self.params.get_float("cost_fn/lookahead", 1.0)

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.error_w = self.params.get_float("cost_fn/error_w", default=1.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.smooth_w = self.params.get_float("cost_fn/smooth_w", default=3.0)
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)

        self._prev_pose = None
        self._prev_index = -1
        self._cache_thresh = 0.01

        self.path_lock = threading.Lock()
        with self.path_lock:
            self.path = None

        if self.collision_check:
            self.world_rep.reset()


    def apply(self, poses, ip):
        """
        Args:
        poses [(nR, T, 3) tensor] -- Rollout of T positions
        ip    [(3,) tensor]: Inferred pose of car in "world" mode

        Returns:
        [(nR,) tensor] costs for each nR paths
        """
        index, waypoint = self._get_reference_index(ip)

        assert poses.size() == (self.nR, self.T, self.NPOS)
        assert self.path.size()[1] == 4

        all_poses = poses.view(self.nR * self.T, self.NPOS)

        # use terminal distance (nR, tensor)
        # TODO: should we look for CTE for terminal distance?
        errorcost = poses[:, self.T - 1, :2].sub(waypoint[:2]).norm(dim=1).mul(self.error_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = (
            ((poses[:, 1:, 2] - poses[:, : self.T - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        ).mul(self.smooth_w)

        result = errorcost.add(smoothness)

        # get all collisions (nR, T, tensor)
        if self.collision_check:
            collisions = self.world_rep.check_collision_in_map(all_poses).view(
                self.nR, self.T
            )
            collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

            obstacle_distances = self.world_rep.distances(all_poses).view(self.nR, self.T)
            obstacle_distances[:].mul_(self.obs_dist_cooloff)

            obs_dist_cost = obstacle_distances[:].sum(dim=1).mul(self.obs_dist_w)

            result = result.add(collision_cost).add(obs_dist_cost)
            # filter out all colliding trajectories
            colliding = collision_cost.nonzero()
            result[colliding] = 1000000000

        if self.viz_waypoint:
            from librhc.rosviz import viz_selected_waypoint
            viz_selected_waypoint(waypoint[:3].clone())

        if self.do_log_cte:
            from librhc.rosviz import log_cte
            error = self._get_error(ip)
            log_cte(error[1])

        if self.viz_rollouts:
            import librhc.rosviz as rosviz

            non_colliding = (collision_cost == 0).nonzero()

            if non_colliding.size()[0] > 0:

                def print_n(c, poses, ns, cmap="coolwarm"):
                    _, all_idx = torch.sort(c)

                    n = min(self.n_viz, len(c))
                    idx = all_idx[:n] if n > -1 else all_idx
                    rosviz.viz_trajs_cmap(poses[idx], c[idx], ns=ns, cmap=cmap)

                p_non_colliding = poses[non_colliding].squeeze()
                print_n(
                    result[non_colliding].squeeze(), p_non_colliding, ns="final_result"
                )
                print_n(errorcost[non_colliding].squeeze(), p_non_colliding, ns="error")
                print_n(
                    collision_cost[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="collision_cost",
                )
                print_n(
                    obs_dist_cost[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="obstacle_dist_cost",
                )
                print_n(
                    smoothness[non_colliding].squeeze(),
                    p_non_colliding,
                    ns="smoothness",
                )

                if self.print_stats:
                    _, all_sorted_idx = torch.sort(result[non_colliding].squeeze())
                    n = min(self.n_viz, len(all_sorted_idx))
                    idx = all_sorted_idx[:n] if n > -1 else all_sorted_idx

                    print("Final Result")
                    print(result[idx])
                    print("Cost 2 Go")
                    print(errorcost[idx])
                    print("Collision Cost")
                    print(collision_cost[idx])
                    print("Obstacle Distance Cost")
                    print(obs_dist_cost[idx])
                    print("Smoothness")
                    print(smoothness[idx])

        return result

    def set_task(self, pathmsg):
        """
        Args:
        path [(x,y,h,v),...] -- list of xyhv named tuple
        """
        self._prev_pose = None
        self._prev_index = None
        path = self.dtype([[pathmsg[i].x, pathmsg[i].y, pathmsg[i].h, pathmsg[i].v] for i in range(len(pathmsg))])
        assert path.size() == (len(pathmsg),4)
        # TODO: any other tests to check validity of path?

        with self.path_lock:
            self.path = path
            self.waypoint_diff = torch.mean(torch.norm(self.path[1:,:2] - self.path[:-1,:2], dim=1))
            # TODO: could add value fn that checks 
            # viability of path
            return True

    def task_complete(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.path_lock:
            if self.path is None:
                return False
        return self._path_complete(state)

    def get_desired_speed(self, state, desired_speed=0):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.path_lock:
            if self.path is None:
                return 0
        index, waypoint = self._get_reference_index(state)
        v =  waypoint[3]
        # TODO: consider ramping down if nearing the last waypoint
        if v < 0:
            return desired_speed
        else:
            return v

    def get_task_type(self):
        return Tracking.TASK_TYPE

    def _path_complete(self, pose):
        '''
        path_complete computes whether the vehicle has completed the path
            based on whether the reference index refers to the final point
            in the path and whether e_x is below the finish_threshold
            or e_y exceeds an 'exceed threshold'.
        input:
            pose - current pose of the vehicle [x, y, heading]
            error - error vector [e_x, e_y]
        output:
            is_path_complete - boolean stating whether the vehicle has
                reached the end of the path
        '''
        index, waypoint = self._get_reference_index(pose)
        if index == (len(self.path) - 1):
            error = self._get_error(pose, index)
            result = (error[0] < self.finish_threshold) or (abs(error[1]) > self.exceed_threshold)
            result =  True if result == 1 else False
            return result
        return False


    def _get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        with self.path_lock:
            if ((self._prev_pose is not None) and \
               (torch.norm(self._prev_pose[:2] - pose[:2]) < self._cache_thresh)):
                return (self._prev_index, self.path[self._prev_index])
            diff = self.path[:,:3] - pose
            dist = diff[:,:2].norm(dim=1)
            index = dist.argmin()
            index += int(self.lookahead / self.waypoint_diff)
            index = min(index, len(self.path)-1)
            self._prev_pose = pose
            self._prev_index = index
            return (index, self.path[index])

    def _get_error(self, pose, index=-1):
        '''
        Computes the error vector for a given pose and reference index.
        input:
            pose - pose of the car [x, y, heading]
            index - integer corresponding to the reference index into the
                reference path
        output:
            e_p - error vector [e_x, e_y]
        '''
        index, waypoint = self._get_reference_index(pose)
        theta = pose[2]
        c, s = torch.cos(theta), torch.sin(theta)
        R = self.dtype([(c, s), (-s, c)])
        return torch.matmul(R, waypoint[:2] - pose[:2])
