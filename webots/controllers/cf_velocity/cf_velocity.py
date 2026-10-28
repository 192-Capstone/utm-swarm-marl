"""
cf_velocity.py
--------------
Webots controller for ONE Crazyflie in the multi-drone Level-2 world
(webots/worlds/swarm_l2_multi.wbt, envs/webots_l2_multi.py). It runs Bitcraze's
velocity PID onboard, every basic step (4 ms), exactly as WebotsL2Adapter does for
the single drone (envs/webots_l2.py), using the drone's own IMU, gyro and GPS.

The supervisor writes one command per control step into this robot's customData
(format: envs/cf_velocity_pid.py):
    ""                      idle: motors off
    "<ep> H vx vy vz"       held at the spawn during spin-up: motors at hover speed, no PID
    "<ep> F vx vy vz"       flying: velocity command (m/s, world frame), yaw rate 0
Every change to F (after H, idle or a new episode number) is a fresh start: the PID is reset
with the hover pre-load and the altitude setpoint and previous GPS are taken from the drone's
position (envs/cf_velocity_pid.py ControllerState), so a parked drone placed on its pad (H, then
F) starts like a drone released at the episode start. A malformed command
is an error: it is printed to stderr, the motors are switched off and the
controller exits with code 1.

E2 radio (webots.radio.enabled, envs/radio_msg.py): started with controllerArgs "--radio" "period_s=.." ...,
the drone also has an Emitter "radio_tx" and a Receiver "radio_rx" (in its extensionSlot). Every
basic step it reads the intents the UTM sent, and while flying (F) it reports its own GPS
position, velocity and intent once per period (RadioSender). Without the argument none of this
runs and no device is looked up.
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from controller import Robot                                            # noqa: E402
from envs.cf_velocity_pid import (VelocityPID, HOVER_RAD_S, HOVER_INTEGRATOR,   # noqa: E402
                                  ControllerState, parse_command)
from envs.radio_msg import RadioSender, parse_controller_args          # noqa: E402


def set_motors(motors, m):
    # L2 119-122: spin directions as in Bitcraze's controller: m1, m3 negative
    motors[0].setVelocity(-m[0]); motors[1].setVelocity(m[1])
    motors[2].setVelocity(-m[2]); motors[3].setVelocity(m[3])


def main():
    robot = Robot()
    basic_ms = int(round(robot.getBasicTimeStep()))                     # L2 62-63
    dt = basic_ms / 1000.0
    motors = []                                                         # L2 72-77
    for k in range(1, 5):
        m = robot.getDevice(f'm{k}_motor')
        m.setPosition(float('inf'))
        m.setVelocity(0.0)
        motors.append(m)
    imu = robot.getDevice('inertial_unit'); imu.enable(basic_ms)       # L2 78-80
    gps = robot.getDevice('gps'); gps.enable(basic_ms)
    gyro = robot.getDevice('gyro'); gyro.enable(basic_ms)
    pid = VelocityPID()                                                 # L2 81

    state = ControllerState()
    z_des = 0.0
    prev_gps = [0.0, 0.0, 0.0]

    radio = parse_controller_args(sys.argv[1:])                         # E2 radio: None when off
    if radio is not None:
        tx, rx = robot.getDevice('radio_tx'), robot.getDevice('radio_rx')
        rx.enable(basic_ms)
        sender = RadioSender(robot.getName(), radio, dt)

    while robot.step(basic_ms) != -1:
        try:
            cmd = parse_command(robot.getCustomData())
        except ValueError as e:
            print(f"cf_velocity [{robot.getName()}]: {e}", file=sys.stderr, flush=True)
            set_motors(motors, [0.0] * 4)
            sys.exit(1)

        act = state.update(cmd)
        if radio is not None:
            while rx.getQueueLength() > 0:                              # intents from the UTM
                try:
                    sender.hear(rx.getString())
                except ValueError as e:                                 # caught by the adapter's log check
                    print(f"cf_velocity [{robot.getName()}]: {e}", file=sys.stderr, flush=True)
                    set_motors(motors, [0.0] * 4)
                    sys.exit(1)
                rx.nextPacket()
            if act in ('idle', 'hold'):
                sender.stop()
        if act == 'idle':                                               # motors off
            set_motors(motors, [0.0] * 4)
            continue
        if act == 'hold':                                               # L2 106: hover speed, no PID
            set_motors(motors, [HOVER_RAD_S] * 4)
            continue
        _, _, (vx, vy, vz) = cmd

        if act == 'arm':                                                # first F after H / idle / new episode
            pid.reset(integrator=HOVER_INTEGRATOR)                      # L2 102
            g = gps.getValues()
            z_des = g[2]                                                # L2 105 (= pad z: the drone was pinned)
            prev_gps = list(g)                                          # L2 116

        # ---- one basic step of WebotsL2Adapter.step_simulation, L2 146-168 ----
        roll, pitch, yaw = imu.getRollPitchYaw()                        # L2 146
        yaw_rate = gyro.getValues()[2]                                  # L2 147
        g = gps.getValues()                                             # L2 148
        v_world = [(g[k] - prev_gps[k]) / dt for k in range(3)]         # L2 149
        prev_gps = list(g)                                              # L2 150
        if radio is not None:
            msg = sender.step(cmd[0], act, robot.getTime(), g)
            if msg is not None:
                tx.send(msg)
        cy, sy = math.cos(yaw), math.sin(yaw)                           # L2 151
        vx_b = v_world[0] * cy + v_world[1] * sy                        # L2 152
        vy_b = -v_world[0] * sy + v_world[1] * cy                       # L2 153
        z_des = max(0.2, z_des + vz * dt)                               # L2 164
        cx_b = vx * cy + vy * sy                                        # L2 165
        cy_b = -vx * sy + vy * cy                                       # L2 166
        m = pid(dt, cx_b, cy_b, 0.0, z_des, roll, pitch, yaw_rate, g[2], vx_b, vy_b)   # L2 167
        set_motors(motors, m)                                           # L2 168


if __name__ == '__main__':
    main()
