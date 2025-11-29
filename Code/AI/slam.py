# slam.py
import numpy as np
import math
from scipy.spatial import cKDTree
import threading
from collections import deque

# Utility functions
def polar_to_cartesian(scan):
    # expects Nx2: angle_rad, dist_m
    a = scan[:, 0]
    r = scan[:, 1]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.vstack((x, y)).T

def transform_points(points, pose):
    # pose = (x, y, theta)
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    R = np.array([[c, -s],[s, c]])
    pts = (R @ points.T).T + np.array([pose[0], pose[1]])
    return pts

def bresenham(x0, y0, x1, y1):
    # integer Bresenham line generator
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    if dy <= dx:
        err = dx // 2
        for _ in range(dx+1):
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for _ in range(dy+1):
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

class OccupancyGrid:
    def __init__(self, size_m=10.0, resolution=0.02):
        self.resolution = resolution
        self.size_m = size_m
        self.width = int(np.ceil(size_m / resolution))
        self.height = int(np.ceil(size_m / resolution))
        self.origin = (-size_m/2.0, -size_m/2.0)  # center origin
        # log-odds grid
        self.grid = np.zeros((self.width, self.height), dtype=float)
        # parameters
        self.lo_occ = math.log(0.9 / 0.1)
        self.lo_free = math.log(0.3 / 0.7)
        self.lo_min = -10
        self.lo_max = 10

    def world_to_cell(self, x, y):
        ix = int((x - self.origin[0]) / self.resolution)
        iy = int((y - self.origin[1]) / self.resolution)
        return ix, iy

    def cell_to_world(self, ix, iy):
        x = ix * self.resolution + self.origin[0] + self.resolution*0.5
        y = iy * self.resolution + self.origin[1] + self.resolution*0.5
        return x, y

    def update_with_scan(self, pose, scan_cartesian):
        # pose world frame
        # scan_cartesian points are in sensor frame (x,y) meters
        # transform scan to world
        pts_world = transform_points(scan_cartesian, pose)
        # robot cell
        rx, ry = pose[0], pose[1]
        rix, riy = self.world_to_cell(rx, ry)
        # for each point, raytrace
        for px, py in pts_world:
            iix, iiy = self.world_to_cell(px, py)
            # avoid out-of-bounds
            if not (0 <= iix < self.width and 0 <= iiy < self.height):
                continue
            # free cells
            for cx, cy in bresenham(rix, riy, iix, iiy):
                # skip endpoint (occupied)
                if cx == iix and cy == iiy:
                    continue
                self.grid[cx, cy] += self.lo_free
                self.grid[cx, cy] = max(self.lo_min, min(self.lo_max, self.grid[cx, cy]))
            # endpoint
            self.grid[iix, iiy] += self.lo_occ
            self.grid[iix, iiy] = max(self.lo_min, min(self.lo_max, self.grid[iix, iiy]))

    def to_image(self):
        # convert log-odds to 8-bit image: occupied dark, free light
        p = 1 - 1/(1+np.exp(self.grid))
        # occupied near 1 => dark; free near 0 => bright
        img = (255 * (1.0 - p)).astype(np.uint8)
        # reshape to HxW for display (opposite axis)
        return np.flipud(img.T.copy())  # returns H x W

class SimpleSLAM:
    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        self.pose = (0.0, 0.0, 0.0)  # x,y,theta in meters and radians
        self._scan_lock = threading.Lock()
        self._last_scan_cart = None
        self._kf_tree = None
        self._initialized = False
        self.history = deque(maxlen=5)

    def odom_update(self, dx, dy, dtheta):
        # naive odom integration (dx,dy in robot frame)
        x, y, th = self.pose
        c = math.cos(th)
        s = math.sin(th)
        # rotate local dx/dy into world
        wx = c * dx - s * dy
        wy = s * dx + c * dy
        self.pose = (x + wx, y + wy, self._wrap_angle(th + dtheta))

    def _wrap_angle(self, a):
        return (a + math.pi) % (2*math.pi) - math.pi

    def update_with_scan(self, scan):
        # scan: Nx2 polar: angle,u distance
        if scan is None or len(scan) < 10:
            return

        cart = polar_to_cartesian(scan)
        # initial pose guess is odom pose (self.pose)
        # do a small ICP-like alignment against previous scan points
        if self._last_scan_cart is not None and len(self._last_scan_cart) >= 10:
            # Transform current with initial pose and align to previous world points
            best_pose = self._icp_align(cart, self._last_scan_cart, self.pose)
            self.pose = best_pose
        # update grid using final pose
        self.grid.update_with_scan(self.pose, cart)
        self._last_scan_cart = transform_points(cart, self.pose)
        self.history.append((self.pose, self._last_scan_cart.copy()))

    def _icp_align(self, src_cart, ref_world, pose_guess, max_iter=10, tol=1e-3):
        # src_cart: Nx2 points in sensor frame (current scan)
        # ref_world: Mx2 points in world frame (previous scan in world)
        # pose_guess: (x,y,theta)
        # iterative closest point with small transforms
        src = src_cart.copy()
        # initial transform: pose_guess
        tx, ty, tth = pose_guess
        pose = np.array([tx, ty, tth], dtype=float)
        ref_kdt = cKDTree(ref_world)
        for i in range(max_iter):
            # transform src to world
            pts_w = transform_points(src, pose)
            # find nearest neighbors
            dists, idxs = ref_kdt.query(pts_w, k=1, n_jobs=-1)
            # filter by distance threshold (e.g., 0.5m)
            mask = dists < 0.4
            if np.count_nonzero(mask) < 20:
                break
            A = []
            b = []
            matched_src = pts_w[mask]
            matched_ref = ref_world[idxs[mask]]
            # compute centroids
            cs = matched_src.mean(axis=0)
            cr = matched_ref.mean(axis=0)
            # compute rotation using SVD on covariance
            S = (matched_ref - cr).T @ (matched_src - cs)
            U, _, Vt = np.linalg.svd(S)
            R = U @ Vt
            # ensure proper rotation
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            angle = math.atan2(R[1,0], R[0,0])
            # compute translation
            t = cr - (R @ cs)
            # update pose (compose)
            # new pose rotation theta' = angle + pose.theta
            new_theta = self._wrap_angle(angle)
            # but we need pose that when applied to sensor frame equals transform R,t
            # approximate by setting pose = [t_x, t_y, angle]
            new_pose = np.array([t[0], t[1], new_theta])
            # check change magnitude
            dp = np.linalg.norm(new_pose[:2] - pose[:2]) + abs(self._wrap_angle(new_pose[2] - pose[2]))
            pose = new_pose
            if dp < tol:
                break
        # finally return pose
        return tuple(pose)
