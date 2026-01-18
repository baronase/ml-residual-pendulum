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

    def __init__(self, kp_in: float = 10.0, kd_in: float = 1.0,
                 ke_in: float = 5.0, theta_switch_in: float = 0.3,
                 max_torque_in: float = 2.0, log_interval_in=10) -> None:
        self.kp = kp_in
        self.kd = kd_in
        self.ke = ke_in
        self.theta_switch = theta_switch_in
        self.max_torque = max_torque_in
        # DEBUG
        self.step_for_print = 0
        self.log_interval = log_interval_in

        # pendulum
        self.pen_l = 1
        self.pen_m = 1
        self.pen_I = self.pen_m * (self.pen_l ** 2) / 3.0

    def energy(self, theta: float, theta_dot: float) -> float:
        # Ek = 0.5 * I * omega**2
        Ek = 0.5 * self.pen_I * theta_dot ** 2
        # Ep = m * g * l * (1.0 + math.cos(theta))   # Pendulum-v1 convention
        Ep = 1 * 10 * self.pen_l / 2 * (1.0 + math.cos(theta))
        return Ek + Ep

    # def energy(self, theta: float, theta_dot: float) -> float:  # <--- ORIGINAL
    #     return 0.5 * (theta_dot ** 2) + (1.0 + math.cos(theta))

    def u_pd(self, theta: float, theta_dot: float) -> float:
        # stabilize around theta=0
        theta = wrap_to_pi(theta)
        return -self.kp * theta - self.kd * theta_dot

    def u_energy(self, theta: float, theta_dot: float) -> float:
        # energy target: upright (theta=0, theta_dot=0) => E*=2
        e = self.energy(theta, theta_dot) - 10.0  # positive => too much energy
        # Pumping direction: "push with the swing" when energy is low, oppose when high
        # direction = math.copysign(1.0, theta_dot * math.cos(theta) + 1e-6)  # <-- ORIGINAL
        direction = math.copysign(1.0, theta_dot + 1e-6)
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
        # DEBUG
        self.step_for_print += 1

        theta, theta_dot = obs_to_theta_theta_dot(obs)
        u_e = self.u_energy(theta, theta_dot)
        u_p = self.u_pd(theta, theta_dot)

        if self.step_for_print % self.log_interval == 0:
            print(
                f"step: {self.step_for_print} - x:{obs[0]:.2f}, y:{obs[1]:.2f} | theta:{theta:.2f} theta_dot:{theta_dot:.2f}")

        w = self.blend_weight(theta)
        u = (1.0 - w) * u_e + w * u_p
        if self.step_for_print % self.log_interval == 0:
            print(f"step: {self.step_for_print} - u_e:{u_e:.2f}, u_p:{u_p:.2f} | w:{w} u:{u:.2f}")
        # clip to env action bounds
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        return np.array([u], dtype=np.float32)
