import torch
from . import TL, Dispersion, MXPI

class Warmstart:
    # Size of control vector
    NCTRL = 2
    init_fns = {"tl": TL, 
                "dispersion": Dispersion}

    def __init__(self, params, logger, dtype, _model, cost=None):
        self.logger = logger
        self.params = params
        self.dtype = dtype
        self.model = _model
        self.cost = cost

        self.reset()

    def use_trajopt(self):
        return True

    def reset(self):
        self.K = self.params.get_int("K", default=62)
        self.T = self.params.get_int("T", default=15)

        init_fn = self.init_fns[self.params.get_str("trajgen/init_fn", 
                                        default="tl")]

        self.init_fn = init_fn(self.params, self.logger, 
                self.dtype, self.model)
        self.trajopt = MXPI(self.params, self.logger, 
                self.dtype, self.model)

        self.v = 0

    def get_control_trajectories(self, velocity):
        """
        Returns:
        [(K, T, NCTRL) tensor] -- of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        """
        self.v = velocity
        return self.init_fn.get_control_trajectories(velocity)

    def generate_control(self, controls, costs, r2c):
        """
        Args:
        controls [(K, T, NCTRL) tensor] -- Returned by get_control_trajectories
        costs [(K, 1) tensor] -- Cost to take a path
        rollouts [(K, T, 1) tensor] -- Initialized rollouts vector

        Returns:
        [(T, NCTRL) tensor] -- The lowest cost trajectory to take
        """
        assert controls.size() == (self.K, self.T, 2)
        assert costs.size() == (self.K,)
        ws_control, _ = self.init_fn.generate_control(controls, costs)

        self.trajopt.warmstart(ws_control)
        trajs = self.trajopt.get_control_trajectories(self.v)
        costs = r2c(trajs)
        return self.trajopt.generate_control(trajs, costs)
