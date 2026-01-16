from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def obs_to_theta_theta_dot(obs: np.ndarray) -> tuple[float, float]:
    """Pendulum-v1 obs = [cos(theta), sin(theta), theta_dot], with theta in [-pi, pi]."""
    cos_t, sin_t, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(sin_t, cos_t)
    return theta, theta_dot


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class EnergyPDController:
    """
    Baseline controller: energy-shaping swing-up + PD stabilize near upright.

    Conventions:
      - Pendulum-v1 uses theta=0 as upright.
      - We use a normalized energy:
            E = 0.5 * theta_dot^2 + (1 + cos(theta))
        so:
            E* (upright, zero velocity) = 2.0
            E (hanging down at pi, zero velocity) = 0.0

    Action is torque u in [-max_torque, max_torque].
    """
    # PD gains (used near upright)
    kp: float
    kd: float

    # Energy shaping gain (used far from upright)
    ke: float

    # Switch/blend region (radians)
    theta_switch: float  # 0.5 ~ around ~28.6 degrees

    # Residual scale not used here (for later); controller outputs full torque
    max_torque: float

    def __init__(self, kp_in: float = 10.0, kd_in: float = 1.0, ke_in: float = 2.0, theta_switch_in: float = 0.5, max_torque_in: float = 2.0) -> None:
        self.kp = kp_in
        self.kd = kd_in
        self.ke = ke_in
        self.theta_switch = theta_switch_in
        self.max_torque = max_torque_in

    def energy(self, theta: float, theta_dot: float) -> float:
        return 0.5 * (theta_dot ** 2) + (1.0 + math.cos(theta))

    def u_pd(self, theta: float, theta_dot: float) -> float:
        # stabilize around theta=0
        theta = wrap_to_pi(theta)
        return -self.kp * theta - self.kd * theta_dot

    def u_energy(self, theta: float, theta_dot: float) -> float:
        # energy target: upright (theta=0, theta_dot=0) => E*=2
        e = self.energy(theta, theta_dot) - 2.0  # positive => too much energy
        # Pumping direction: "push with the swing" when energy is low, oppose when high
        direction = math.copysign(1.0, theta_dot * math.cos(theta) + 1e-6)
        return -self.ke * e * direction

    def blend_weight(self, theta: float) -> float:
        """
        Weight for PD vs energy:
          w=1 near upright, w=0 when |theta| >= theta_switch.
        """
        a = abs(wrap_to_pi(theta))
        if a >= self.theta_switch:
            return 0.0
        # smooth-ish ramp (quadratic)
        x = 1.0 - (a / self.theta_switch)
        return x * x

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        theta, theta_dot = obs_to_theta_theta_dot(obs)
        u_e = self.u_energy(theta, theta_dot)
        u_p = self.u_pd(theta, theta_dot)

        w = self.blend_weight(theta)
        u = (1.0 - w) * u_e + w * u_p

        # clip to env action bounds
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        return np.array([u], dtype=np.float32)
