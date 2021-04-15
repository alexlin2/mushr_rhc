import torch





class UMPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, params, logger, dtype, mvmt_model, trajgen, cost):
        self.dtype = dtype
        self.logger = logger
        self.params = params

        self.trajgen = trajgen
        self.kinematics = mvmt_model
        self.cost = cost

        self.reset(init=True)

    def reset(self, init=False):
        """
        Args:
        init [bool] -- whether this is being called by the init function
        """
        self.T = self.params.get_int("T", default=15)  
        self.K = self.params.get_int("K", default=62) 
        self.P = self.params.get_int("P", default=1)
        self.nR = self.K * self.P # num rollouts
        
        # Update number of rollouts required by kinematics model
        self.kinematics.set_k(self.nR)

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.nR * 3, self.T, self.NPOS)

        self.desired_speed = [0.2,0.6,0.8]
        #self.desired_speed = self.params.get_float("trajgen/desired_speed", default=1.0)

        if not init:
            self.trajgen.reset()
            self.kinematics.reset()
            self.cost.reset()

    def step(self, state, ip):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        assert ip.size() == (3,)

        if self.task_complete(ip):
            return None, None

        self.state = state
        self.ip = ip
        # For each K trial, the first position is at the current position
        '''
        v = min(
            self.desired_speed,
            self.cost.get_desired_speed(ip, self.desired_speed)
        )
        '''
        trajs = self.trajgen.get_control_trajectories(self.desired_speed)

        costs = self._rollout2cost(trajs)

        if not self.trajgen.use_trajopt():
            result, idx = self.trajgen.generate_control(trajs, costs)
        else:
            result, idx = self.trajgen.generate_control(trajs, 
                                            costs, self._rollout2cost)
        if idx is None: 
            # TODO: optimize so we only roll out single canonical control
            idx = 0
            k_prev = self.kinematics.K
            self.kinematics.set_k(1)
            self.rollouts[idx, 0] = ip
            for t in range(1, self.T):
                cur_x = self.rollouts[idx, t - 1].resize(1,self.NPOS)
                self.rollouts[idx, t] = self.kinematics.apply(cur_x, 
                        result[t].resize(1,2))
            self.kinematics.set_k(k_prev)

        return result, self.rollouts[idx]

    def set_task(self, task):
        return self.cost.set_task(task)

    def task_complete(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        return self.cost.task_complete(state)

    def _perform_rollout(self, trajs):
        self.rollouts.zero_()
        if self.state.size() == (3,):
            self.rollouts[:, 0] = self.state.expand_as(self.rollouts[:, 0])
        else:
            idx = torch.randint(low=0, high=self.state.size(0), size=(self.P,))
            idx = idx.repeat_interleave(self.K)
            self.rollouts[:, 0] = self.state[idx]
        assert trajs.size() == (self.K * 3, self.T, 2)
        #trajs_e = trajs.repeat(self.P, 1, 1)
        #assert trajs_e.size() == (self.K * self.P, self.T, 2)
        for t in range(1, self.T):
            cur_x = self.rollouts[:, t - 1]
            self.rollouts[:, t] = self.kinematics.apply(cur_x, trajs[:, t - 1])

    def _rollout2cost(self, trajs):
        self._perform_rollout(trajs)
        costs = self.cost.apply(self.rollouts, self.ip)
        return costs #.resize(self.P, self.K).mean(0)
