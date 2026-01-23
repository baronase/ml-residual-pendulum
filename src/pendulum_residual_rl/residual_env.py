import gymnasium as gym
import numpy as np

class ResidualWrapper(gym.Wrapper):
    """
    Wraps an env so that the agent outputs a residual action.
    Env receives: u_total = clip(u_base + alpha * u_res, action_low, action_high)
    """
    def __init__(self, env: gym.Env, baseline_fn, alpha: float = 0.3):
        super().__init__(env)
        self.baseline_fn = baseline_fn
        self.alpha = float(alpha)

        # Agent outputs residual in [-1, 1] (normalized)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # original env bounds
        self.low = float(env.action_space.low[0])
        self.high = float(env.action_space.high[0])

        self._last_obs = None

    def step(self, action):
        assert self._last_obs is not None, "Call reset() before step()."
        # action is residual in [-1, 1]
        u_res = float(np.clip(action[0], -1.0, 1.0))

        # baseline action in env units
        u_base = self.baseline_fn(self._last_obs)
        u_base = float(u_base[0])

        u_total = np.clip(u_base + self.alpha * u_res * (self.high - self.low) / 2.0, self.low, self.high)
        self._last_obs, reward, term, trunc, info = self.env.step(np.array([u_total], dtype=np.float32))
        info["u_base"] = u_base
        info["u_res"] = u_res
        info["u_total"] = float(u_total)
        return self._last_obs, reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info
