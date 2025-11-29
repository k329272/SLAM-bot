# main.py
import time
import threading
import math
import numpy as np

from motor import Motor
from lidar import LidarDriver
from slam import OccupancyGrid, SimpleSLAM, polar_to_cartesian
from planner import ThetaStarPlanner
from web_ui import start_ui_thread

import pigpio

# Hardware pins — adapt to your wiring
# Left motor pins
LEFT_IN1 = 17
LEFT_IN2 = 27
LEFT_PWM = 22
LEFT_ENCODER_PIN = 5

# Right motor pins
RIGHT_IN1 = 23
RIGHT_IN2 = 24
RIGHT_PWM = 25
RIGHT_ENCODER_PIN = 6

# robot geometry and constants
WHEEL_BASE_M = 0.12  # distance between wheels
WHEEL_DIAMETER = 0.06

def enc_counts_to_distance(counts, counts_per_rev, gear_ratio):
    # motor revs = counts / counts_per_rev
    motor_revs = counts / float(counts_per_rev)
    wheel_revs = motor_revs / float(gear_ratio)
    dist = wheel_revs * math.pi * WHEEL_DIAMETER
    return dist

def main():
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio not running. Start pigpiod.")

    # Create motor objects
    left = Motor(pi, LEFT_IN1, LEFT_IN2, LEFT_PWM, LEFT_ENCODER_PIN, reversed=False)
    right = Motor(pi, RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, RIGHT_ENCODER_PIN, reversed=True)

    # tune motor params (counts_per_rev etc)
    left.counts_per_rev = 20
    left.gear_ratio = 50
    left.wheel_diameter_m = WHEEL_DIAMETER

    right.counts_per_rev = 20
    right.gear_ratio = 50
    right.wheel_diameter_m = WHEEL_DIAMETER

    # lidar
    lidar = LidarDriver(port='/dev/ttyUSB0')

    # occupancy grid and SLAM
    grid = OccupancyGrid(size_m=10.0, resolution=0.02)
    slam = SimpleSLAM(grid)
    planner = ThetaStarPlanner(grid)

    # start UI
    start_ui_thread(grid, slam, planner, host='0.0.0.0', port=5000)

    # Odometry loop using motor.rpm estimates -> integrate
    last_time = time.time()
    # Use a simple velocity control interface: for demo we set small forward speed
    # Set motors to zero to begin
    left.set_target_rpm(0.0)
    right.set_target_rpm(0.0)

    # path-following settings
    path = None
    path_idx = 0

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # compute odometry from wheel RPM (approx)
            # rpm -> m/s
            vl = (left.rpm / 60.0) * math.pi * left.wheel_diameter_m
            vr = (right.rpm / 60.0) * math.pi * right.wheel_diameter_m

            # differential drive kinematics
            v = (vl + vr) / 2.0
            omega = (vr - vl) / WHEEL_BASE_M

            # integrate
            dx = v * dt
            dtheta = omega * dt
            slam.odom_update(dx, 0.0, dtheta)  # dx in robot forward direction

            # feed lidar
            scan = lidar.get_scan()
            slam.update_with_scan(scan)

            # If UI requested a path, GLOBAL['path'] is set by web_ui — but we don't import GLOBAL; read planner path directly via web_ui's GLOBAL
            # Instead we simply check planner path stored elsewhere. To keep it simple, we periodically plan to last requested goal if present.
            # For demo: if planner has GLOBAL path it's accessible; but here we'll just follow grid.GLOBAL path if set
            # We'll check the planner's last planned path (we rely on web UI to set planner GLOBAL['path'])
            # Instead of coupling, we can check grid attribute: we'll just request a simple plan back to origin for demo
            # (Real code: integrate socket notifications or shared data structure.) Here, we simply leave motors idle.

            # small sleep
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        lidar.stop()
        left.stop()
        right.stop()
        pi.stop()

if __name__ == '__main__':
    main()
