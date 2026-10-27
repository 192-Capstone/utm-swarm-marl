"""
cf_velocity_pid.py
------------------
Bitcraze's velocity controller for the Webots Crazyflie (Level 2), standard library only,
so the same code runs in envs/webots_l2.py (single drone, inside the supervisor) and in
webots/controllers/cf_velocity/cf_velocity.py (one controller per drone, multi-drone).

Also the command message the multi-drone supervisor writes into each drone's customData
once per control step:

    "<ep> <mode> <vx> <vy> <vz>"     velocities in m/s, world frame
        ep    episode number; a new number resets the controller
        mode  H = held (supervisor pins the drone during spin-up: motors at hover speed,
                  PID not run)
              F = flying (the PID runs every basic step)
    ""  (empty) = idle, motors off
"""

import math

HOVER_RAD_S = math.sqrt(0.05 * 9.81 / (4 * 4e-05))      # ~55.4 rad/s per motor at hover

MODES = ('H', 'F')


class VelocityPID:
    """Bitcraze's pid_velocity_fixed_height_controller (Webots Crazyflie
    example, MIT licence), same gains and structure, without numpy."""
    G = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
         "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 5, "kd_z": 5}

    def __init__(self):
        self.reset()

    def reset(self, integrator=0.0):
        self.past_vx_error = self.past_vy_error = self.past_alt_error = 0.0
        self.past_pitch_error = self.past_roll_error = 0.0
        self.altitude_integrator = integrator

    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def __call__(self, dt, vx_d, vy_d, yaw_rate_d, alt_d, roll, pitch, yaw_rate, alt, vx, vy):
        g, c = self.G, self._clip
        vx_e, vy_e = vx_d - vx, vy_d - vy
        vx_der, vy_der = (vx_e - self.past_vx_error) / dt, (vy_e - self.past_vy_error) / dt
        pitch_d = g["kp_vel_xy"] * c(vx_e, -1, 1) + g["kd_vel_xy"] * vx_der
        roll_d = -g["kp_vel_xy"] * c(vy_e, -1, 1) - g["kd_vel_xy"] * vy_der
        self.past_vx_error, self.past_vy_error = vx_e, vy_e
        alt_e = alt_d - alt
        alt_der = (alt_e - self.past_alt_error) / dt
        self.altitude_integrator += alt_e * dt
        alt_cmd = g["kp_z"] * alt_e + g["kd_z"] * alt_der + g["ki_z"] * c(self.altitude_integrator, -2, 2) + 48
        self.past_alt_error = alt_e
        pitch_e, roll_e = pitch_d - pitch, roll_d - roll
        pitch_der, roll_der = (pitch_e - self.past_pitch_error) / dt, (roll_e - self.past_roll_error) / dt
        roll_cmd = g["kp_att_rp"] * c(roll_e, -1, 1) + g["kd_att_rp"] * roll_der
        pitch_cmd = -g["kp_att_rp"] * c(pitch_e, -1, 1) - g["kd_att_rp"] * pitch_der
        yaw_cmd = g["kp_att_y"] * c(yaw_rate_d - yaw_rate, -1, 1)
        self.past_pitch_error, self.past_roll_error = pitch_e, roll_e
        m1 = alt_cmd - roll_cmd + pitch_cmd + yaw_cmd
        m2 = alt_cmd - roll_cmd - pitch_cmd - yaw_cmd
        m3 = alt_cmd + roll_cmd - pitch_cmd + yaw_cmd
        m4 = alt_cmd + roll_cmd + pitch_cmd - yaw_cmd
        return [c(m, 0, 600) for m in (m1, m2, m3, m4)]


# integrator pre-load that starts the PID at hover (as WebotsL2Adapter.reset_world)
HOVER_INTEGRATOR = (HOVER_RAD_S - 48.0) / VelocityPID.G["ki_z"]


def format_command(ep, mode, v):
    """Command string for customData. repr() keeps every float exact."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    vx, vy, vz = (float(x) for x in v)
    return f"{int(ep)} {mode} {vx!r} {vy!r} {vz!r}"


def parse_command(s):
    """customData string -> (ep, mode, (vx, vy, vz)), or None when empty (idle).
    Raises ValueError on a malformed string."""
    s = s.strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) != 5 or parts[1] not in MODES:
        raise ValueError(f"bad command {s!r}: expected '<ep> <H|F> <vx> <vy> <vz>'")
    return int(parts[0]), parts[1], (float(parts[2]), float(parts[3]), float(parts[4]))
