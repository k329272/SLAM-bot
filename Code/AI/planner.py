# planner.py
import heapq
import math
import numpy as np

class ThetaStarPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        self.occ_thresh = 0.6  # probability threshold to consider occupied

    def is_free(self, ix, iy):
        if not (0 <= ix < self.grid.width and 0 <= iy < self.grid.height):
            return False
        # convert log-odds to prob
        p = 1 - 1/(1+math.exp(self.grid.grid[ix, iy]))
        return p < self.occ_thresh

    def line_of_sight(self, a, b):
        (ax, ay), (bx, by) = a, b
        ix0, iy0 = int(ax), int(ay)
        ix1, iy1 = int(bx), int(by)
        for x, y in self._bresenham(ix0, iy0, ix1, iy1):
            if not self.is_free(x, y):
                return False
        return True

    def _bresenham(self, x0, y0, x1, y1):
        # internal bresenham generator
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

    def plan(self, start_w, goal_w):
        # start_w, goal_w are world coordinates (x,y) in meters
        sx, sy = start_w
        gx, gy = goal_w
        s_ix, s_iy = self.grid.world_to_cell(sx, sy)
        g_ix, g_iy = self.grid.world_to_cell(gx, gy)

        # early checks
        if not self.is_free(g_ix, g_iy):
            return None

        start = (s_ix, s_iy)
        goal = (g_ix, g_iy)
        open_set = []
        g_scores = {start: 0.0}
        f_scores = {start: self._heuristic(start, goal)}
        parent = {start: start}
        heapq.heappush(open_set, (f_scores[start], start))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(parent, current)
            # neighbors: 8-connected
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if dx==0 and dy==0:
                        continue
                    nb = (current[0]+dx, current[1]+dy)
                    if not (0 <= nb[0] < self.grid.width and 0 <= nb[1] < self.grid.height):
                        continue
                    if not self.is_free(nb[0], nb[1]):
                        continue
                    # Theta*: try to connect parent(current) directly to neighbor
                    par = parent[current]
                    if self.line_of_sight(par, nb):
                        # use parent as predecessor
                        tentative_g = g_scores[par] + self._dist(par, nb)
                        if tentative_g < g_scores.get(nb, float('inf')):
                            g_scores[nb] = tentative_g
                            parent[nb] = par
                            f_scores[nb] = tentative_g + self._heuristic(nb, goal)
                            heapq.heappush(open_set, (f_scores[nb], nb))
                    else:
                        tentative_g = g_scores[current] + self._dist(current, nb)
                        if tentative_g < g_scores.get(nb, float('inf')):
                            g_scores[nb] = tentative_g
                            parent[nb] = current
                            f_scores[nb] = tentative_g + self._heuristic(nb, goal)
                            heapq.heappush(open_set, (f_scores[nb], nb))
        return None

    def _heuristic(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _reconstruct_path(self, parent, current):
        path = [current]
        while parent[current] != current:
            current = parent[current]
            path.append(current)
        path.reverse()
        # convert to world coordinates
        world_path = []
        for ix, iy in path:
            world_path.append(self.grid.cell_to_world(ix, iy))
        return world_path
