# lidar.py
from rplidar import RPLidar
import threading
import time
import numpy as np
import math

class LidarDriver:
    def __init__(self, port='/dev/ttyUSB0'):
        self.port = port
        self.lidar = None
        self._scans_lock = threading.Lock()
        self._latest_scan = None  # numpy array of shape (N, 2): angle_rad, distance_m
        self._run = False
        self.thread = threading.Thread(target=self._run_thread, daemon=True)
        self._connect()

    def _connect(self):
        try:
            self.lidar = RPLidar(self.port)
            self.lidar.set_pwm(660)  # ensure motor on; tune if necessary
            self._run = True
            self.thread.start()
        except Exception as e:
            print("LIDAR connect error:", e)
            raise

    def _run_thread(self):
        while self._run:
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=500):
                    # scan is list of (quality, angle, distance_mm)
                    pts = []
                    for q, a, d in scan:
                        if d == 0:
                            continue
                        angle_rad = math.radians(a)
                        dist_m = d / 1000.0
                        pts.append((angle_rad, dist_m))
                    with self._scans_lock:
                        self._latest_scan = np.array(pts, dtype=float)
                    # loop continues
                    if not self._run:
                        break
            except Exception as e:
                print("LIDAR read error:", e)
                time.sleep(1.0)

    def get_scan(self):
        with self._scans_lock:
            if self._latest_scan is None:
                return None
            return np.copy(self._latest_scan)

    def stop(self):
        self._run = False
        time.sleep(0.2)
        if self.lidar:
            try:
                self.lidar.stop()
            except:
                pass
            try:
                self.lidar.disconnect()
            except:
                pass
