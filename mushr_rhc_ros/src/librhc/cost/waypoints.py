# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch

import librhc.utils as utils
import threading


class Waypoints:
    NPOS = 3  # x, y, theta
    TASK_TYPE = "GOAL"

    def __init__(self, params, logger, dtype, map, world_rep, value_fn):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        self.viz_rollouts = self.params.get_bool(
            "debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool(
            "debug/viz_rollouts/print_stats", False)

        self.goal = None
        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.P = self.params.get_int("P", default=1)
        self.nR = self.K * self.P  # num rollouts

        xy_thresh = self.params.get_float("cost_fn/xy_threshold", default=0.5)
        self.goal_threshold = self.dtype([xy_thresh, xy_thresh])

        self.dist_horizon = utils.get_distance_horizon(self.params)

        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float(
            "cost_fn/obs_dist_w", default=5.0)
        self.cost2go_w = self.params.get_float(
            "cost_fn/cost2go_w", default=1.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.smooth_w = self.params.get_float("cost_fn/smooth_w", default=3.0)
        self.bounds_cost = self.params.get_float(
            "cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(
            1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)

        self.goal_lock = threading.Lock()
        with self.goal_lock:
            self.goal = None

        self.world_rep.reset()

    def apply(self, poses, ip):
        """
        Args:
        poses [(nR, T, 3) tensor] -- Rollout of T positions
        ip    [(3,) tensor]: Inferred pose of car in "world" mode

        Returns:
        [(nR,) tensor] costs for each nR paths
        """
        with self.goal_lock:
            goal = self.goal

        assert poses.size() == (self.nR, self.T, self.NPOS)
        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.nR * self.T, self.NPOS)

        # use terminal distance (nR, tensor)
        cost2go = self.value_fn.get_value(
            poses[:, self.T - 1, :]).mul(self.cost2go_w)

        # get all collisions (nR, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(
            self.nR, self.T
        )
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        obstacle_distances = self.world_rep.distances(
            all_poses).view(self.nR, self.T)
        obstacle_distances[:].mul_(self.obs_dist_cooloff)

        obs_dist_cost = obstacle_distances[:].sum(dim=1).mul(self.obs_dist_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = (
            ((poses[:, 1:, 2] - poses[:, : self.T - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        ).mul(self.smooth_w)

        result = cost2go.add(collision_cost).add(obs_dist_cost).add(smoothness)

        # filter out all colliding trajectories
        colliding = collision_cost.nonzero()
        result[colliding] = 1000000000

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
                print_n(cost2go[non_colliding].squeeze(),
                        p_non_colliding, ns="cost2go")
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
                    _, all_sorted_idx = torch.sort(
                        result[non_colliding].squeeze())
                    n = min(self.n_viz, len(all_sorted_idx))
                    idx = all_sorted_idx[:n] if n > -1 else all_sorted_idx

                    print("Final Result")
                    print(result[idx])
                    print("Cost 2 Go")
                    print(cost2go[idx])
                    print("Collision Cost")
                    print(collision_cost[idx])
                    print("Obstacle Distance Cost")
                    print(obs_dist_cost[idx])
                    print("Smoothness")
                    print(smoothness[idx])

        return result

    def set_task(self, goal):
        """
        Args:
        goal [(3,) tensor] -- Goal in "world" coordinates
        """
        assert goal.size() == (3,)

        with self.goal_lock:
            self.goal = goal
            return self.value_fn.set_goal(goal)

    def task_complete(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.goal_lock:
            if self.goal is None:
                return False
        dist = self.goal[:2].sub(state[:2]).abs_()
        return dist.lt(self.goal_threshold).min() == 1

    def get_desired_speed(self, state, desired_speed=0):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.goal_lock:
            if self.goal is None:
                return 0
        return desired_speed * (self._dist_to_goal(state) / (self.dist_horizon))

    def get_task_type(self):
        return self.TASK_TYPE

    def _dist_to_goal(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        return self.goal[:2].dist(state[:2])
