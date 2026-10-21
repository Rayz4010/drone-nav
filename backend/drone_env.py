"""
Custom Gymnasium Environment: DroneNavEnv-v1
Autonomous Drone Navigation in a 3D obstacle field using Continuous DRL.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Obstacle:
    """Represents a geometric 3D obstacle (cylinder pillar or bounding box)."""
    def __init__(self, kind: str, pos: np.ndarray, size: np.ndarray, color_hex: str = "#4c566a"):
        self.kind = kind  # 'cylinder', 'box', 'prism', 'dodecahedron'
        self.pos = np.array(pos, dtype=np.float32)  # [x, y, z] center
        self.size = np.array(size, dtype=np.float32)  # [radius, height] or [hx, hy, hz] half-extents
        self.color_hex = color_hex

    def check_collision(self, drone_pos: np.ndarray, drone_radius: float = 0.55) -> bool:
        if self.kind in ['cylinder', 'prism']:
            # Cylinder along Y axis
            radius = self.size[0] + drone_radius
            half_h = (self.size[1] / 2.0) + drone_radius
            dx = drone_pos[0] - self.pos[0]
            dz = drone_pos[2] - self.pos[2]
            dy = abs(drone_pos[1] - self.pos[1])
            return (dx * dx + dz * dz <= radius * radius) and (dy <= half_h)
        else:
            # Box / polyhedron approximation with AABB + drone radius
            half_dims = self.size + drone_radius
            diff = np.abs(drone_pos - self.pos)
            return bool(np.all(diff <= half_dims))

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray, max_dist: float) -> float:
        """Calculates distance from origin along direction to surface of obstacle, or max_dist."""
        if self.kind in ['cylinder', 'prism']:
            # Ray - Cylinder intersection (Y aligned)
            r = self.size[0]
            y_min = self.pos[1] - self.size[1] / 2.0
            y_max = self.pos[1] + self.size[1] / 2.0

            ox, oy, oz = origin[0] - self.pos[0], origin[1], origin[2] - self.pos[2]
            dx, dy, dz = direction[0], direction[1], direction[2]

            a = dx * dx + dz * dz
            if a < 1e-8:
                return max_dist

            b = 2.0 * (ox * dx + oz * dz)
            c = ox * ox + oz * oz - r * r
            disc = b * b - 4.0 * a * c
            if disc < 0:
                return max_dist

            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = max_dist
            for t in (t1, t2):
                if 0.0 < t < t_hit:
                    hit_y = oy + t * dy
                    if y_min <= hit_y <= y_max:
                        t_hit = t
            return t_hit
        else:
            # Ray - AABB Box intersection
            b_min = self.pos - self.size
            b_max = self.pos + self.size

            tmin = -float('inf')
            tmax = float('inf')

            for i in range(3):
                if abs(direction[i]) < 1e-8:
                    if origin[i] < b_min[i] or origin[i] > b_max[i]:
                        return max_dist
                else:
                    inv_d = 1.0 / direction[i]
                    t1 = (b_min[i] - origin[i]) * inv_d
                    t2 = (b_max[i] - origin[i]) * inv_d
                    if t1 > t2:
                        t1, t2 = t2, t1
                    tmin = max(tmin, t1)
                    tmax = min(tmax, t2)
                    if tmin > tmax:
                        return max_dist

            if tmin > 0.0:
                return min(tmin, max_dist)
            elif tmax > 0.0:
                return 0.0  # inside
            return max_dist


class DroneNavEnv(gym.Env):
    """
    Continuous 3D Drone Navigation Environment for Deep RL.

    Observation Space (26-dimensional Box):
      - [0:3] Relative vector to goal [dx, dy, dz] normalized by arena scale (e.g. 25m)
      - [3:6] Current linear velocity [vx, vy, vz] normalized by max speed (8 m/s)
      - [6:10] Drone attitude [sin(yaw), cos(yaw), pitch/max_pitch, roll/max_roll]
      - [10:26] 16 synthetic LIDAR / rangefinder normalized distances [0, 1]

    Action Space (4-dimensional continuous Box[-1.0, 1.0]):
      - a[0]: Pitch command (forward / backward thrust)
      - a[1]: Roll command (lateral strafe thrust)
      - a[2]: Yaw rate command (rotation speed around vertical axis)
      - a[3]: Vertical thrust command (altitude control)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, seed: int = None):
        super().__init__()

        # Arena boundaries: X in [-22, 22], Y in [0.4, 18.0], Z in [-22, 22]
        self.arena_size = 22.0
        self.y_min = 0.5
        self.y_max = 16.0

        # Physical constants
        self.dt = 0.05  # 20 Hz simulation step
        self.max_speed = 7.5  # m/s
        self.max_yaw_rate = 2.4  # rad/s
        self.max_tilt = math.radians(25.0)  # max pitch/roll bank angle
        self.drone_radius = 0.55  # collision sphere
        self.sensor_range = 14.0  # meters
        self.num_rays = 16
        self.goal_reach_radius = 1.25  # meters
        self.max_steps = 600

        # Sensor ray angular offsets in drone local coordinate frame [azimuth_deg, elevation_deg]
        # 12 horizontal rays (30 deg intervals) + 2 diagonal pitch rays + 1 straight up + 1 straight down
        self.ray_angles = [
            (0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (90.0, 0.0),
            (120.0, 0.0), (150.0, 0.0), (180.0, 0.0), (210.0, 0.0),
            (240.0, 0.0), (270.0, 0.0), (300.0, 0.0), (330.0, 0.0),
            (0.0, 25.0),   # Forward-up
            (0.0, -25.0),  # Forward-down
            (0.0, 85.0),   # Upward
            (0.0, -85.0),  # Downward
        ]

        # Action and Observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        obs_dim = 3 + 3 + 4 + self.num_rays  # 26 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State attributes
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.prev_distance = 0.0
        self.obstacles: list[Obstacle] = []

        if seed is not None:
            self.reset(seed=seed)

    def _generate_obstacles(self):
        """Generates procedural convex polyhedra and cylindrical pillars."""
        self.obstacles = []
        palette = ["#3b4252", "#434c5e", "#4c566a", "#e06c75", "#4fa6a1", "#d19a66", "#88c0d0"]

        # 1. Tall cylindrical pillars / low-poly prisms
        num_pillars = self.np_random.integers(10, 15)
        for _ in range(num_pillars):
            for _ in range(30):
                x = self.np_random.uniform(-18.0, 18.0)
                z = self.np_random.uniform(-18.0, 18.0)
                if math.hypot(x, z) >= 5.5:
                    break
            h = self.np_random.uniform(7.0, 15.0)
            r = self.np_random.uniform(1.0, 2.0)
            color = self.np_random.choice(palette)
            self.obstacles.append(
                Obstacle("cylinder", np.array([x, h / 2.0, z]), np.array([r, h]), color)
            )

        # 2. Floating / mid-altitude blocks and polyhedral prisms
        num_blocks = self.np_random.integers(12, 16)
        for _ in range(num_blocks):
            for _ in range(30):
                x = self.np_random.uniform(-18.0, 18.0)
                z = self.np_random.uniform(-18.0, 18.0)
                if math.hypot(x, z) >= 5.5:
                    break
            y = self.np_random.uniform(2.5, 10.0)
            hx = self.np_random.uniform(0.9, 1.8)
            hy = self.np_random.uniform(0.9, 2.0)
            hz = self.np_random.uniform(0.9, 1.8)
            color = self.np_random.choice(palette)
            self.obstacles.append(
                Obstacle("box", np.array([x, y, z]), np.array([hx, hy, hz]), color)
            )


    def _sample_valid_position(self, min_clearance: float = 2.5) -> np.ndarray:
        """Finds a 3D coordinate with sufficient clearance from any obstacle."""
        for _ in range(200):
            x = self.np_random.uniform(-18.0, 18.0)
            y = self.np_random.uniform(1.8, 11.0)
            z = self.np_random.uniform(-18.0, 18.0)
            candidate = np.array([x, y, z], dtype=np.float32)

            collision = False
            for obs in self.obstacles:
                if obs.check_collision(candidate, drone_radius=min_clearance):
                    collision = True
                    break
            if not collision:
                return candidate
        return np.array([0.0, 3.0, 0.0], dtype=np.float32)

    def _cast_sensor_rays(self) -> np.ndarray:
        """Computes ray distances for all 16 synthetic LIDAR rangefinders."""
        ranges = np.full(self.num_rays, self.sensor_range, dtype=np.float32)

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        for i, (az_deg, el_deg) in enumerate(self.ray_angles):
            az = math.radians(az_deg)
            el = math.radians(el_deg)

            # Local ray direction in drone body frame
            lx = math.sin(az) * math.cos(el)
            ly = math.sin(el)
            lz = math.cos(az) * math.cos(el)

            # Rotate by yaw around world Y axis
            wx = lx * cos_yaw + lz * sin_yaw
            wy = ly
            wz = -lx * sin_yaw + lz * cos_yaw
            ray_dir = np.array([wx, wy, wz], dtype=np.float32)

            min_hit = self.sensor_range

            # Intersect with ground (Y=0) and ceiling (Y=self.y_max)
            if wy < -1e-5:
                t_ground = -self.pos[1] / wy
                if 0.0 < t_ground < min_hit:
                    min_hit = t_ground
            elif wy > 1e-5:
                t_ceil = (self.y_max - self.pos[1]) / wy
                if 0.0 < t_ceil < min_hit:
                    min_hit = t_ceil

            # Intersect with outer arena boundary walls
            for axis, val in [(0, -self.arena_size), (0, self.arena_size),
                              (2, -self.arena_size), (2, self.arena_size)]:
                denom = ray_dir[axis]
                if abs(denom) > 1e-5:
                    t_wall = (val - self.pos[axis]) / denom
                    if 0.0 < t_wall < min_hit:
                        # Check if within vertical bounds at hit
                        hit_y = self.pos[1] + t_wall * ray_dir[1]
                        if 0.0 <= hit_y <= self.y_max:
                            min_hit = t_wall

            # Intersect with all obstacles
            for obs in self.obstacles:
                t_obs = obs.intersect_ray(self.pos, ray_dir, min_hit)
                if t_obs < min_hit:
                    min_hit = t_obs

            ranges[i] = min_hit

        return ranges

    def _get_obs(self, sensor_ranges: np.ndarray) -> np.ndarray:
        rel_goal = self.goal_pos - self.pos
        norm_rel_goal = rel_goal / self.arena_size
        norm_vel = self.vel / self.max_speed
        attitude = np.array([
            math.sin(self.yaw),
            math.cos(self.yaw),
            self.pitch / self.max_tilt,
            self.roll / self.max_tilt
        ], dtype=np.float32)
        norm_sensors = sensor_ranges / self.sensor_range

        return np.concatenate([
            norm_rel_goal,
            norm_vel,
            attitude,
            norm_sensors
        ]).astype(np.float32)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.step_count = 0

        # Generate obstacles
        self._generate_obstacles()

        # Spawn drone at valid position
        self.pos = self._sample_valid_position(min_clearance=2.2)
        self.vel = np.zeros(3, dtype=np.float32)
        self.yaw = self.np_random.uniform(0, 2.0 * math.pi)
        self.pitch = 0.0
        self.roll = 0.0

        # Sample goal with minimum distance from drone
        for _ in range(50):
            self.goal_pos = self._sample_valid_position(min_clearance=2.0)
            if np.linalg.norm(self.goal_pos - self.pos) >= 8.0:
                break

        self.prev_distance = float(np.linalg.norm(self.goal_pos - self.pos))
        sensors = self._cast_sensor_rays()
        obs = self._get_obs(sensors)

        info = {
            "distance_to_goal": self.prev_distance,
            "drone_pos": self.pos.tolist(),
            "goal_pos": self.goal_pos.tolist(),
            "obstacles": [
                {
                    "kind": o.kind,
                    "pos": o.pos.tolist(),
                    "size": o.size.tolist(),
                    "color": o.color_hex
                }
                for o in self.obstacles
            ]
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        pitch_cmd, roll_cmd, yaw_rate_cmd, thrust_cmd = action

        # 1. Attitude dynamics
        target_yaw_rate = yaw_rate_cmd * self.max_yaw_rate
        self.yaw = (self.yaw + target_yaw_rate * self.dt) % (2.0 * math.pi)

        # Smooth pitch & roll tilt banking
        self.pitch = float(pitch_cmd * self.max_tilt)
        self.roll = float(roll_cmd * self.max_tilt)

        # 2. Translation dynamics in local heading frame
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # Forward/backward along heading
        forward_accel = pitch_cmd * 8.5
        # Strafe lateral along right vector
        strafe_accel = roll_cmd * 6.5
        # Vertical thrust acceleration
        vertical_accel = thrust_cmd * 7.5

        # World accelerations
        ax = (math.sin(self.yaw) * forward_accel + math.cos(self.yaw) * strafe_accel)
        ay = vertical_accel
        az = (math.cos(self.yaw) * forward_accel - math.sin(self.yaw) * strafe_accel)
        accel = np.array([ax, ay, az], dtype=np.float32)

        # Aerodynamic linear damping drag
        drag_coeff = 1.3
        self.vel += (accel - drag_coeff * self.vel) * self.dt

        # Clip velocity to maximum physical speed
        speed = float(np.linalg.norm(self.vel))
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed

        # Integrate position
        self.pos += self.vel * self.dt

        # 3. Collision and Boundary Detection
        terminated = False
        collision = False

        # Boundary check
        if (abs(self.pos[0]) >= self.arena_size or
            abs(self.pos[2]) >= self.arena_size or
            self.pos[1] <= self.y_min or
            self.pos[1] >= self.y_max):
            collision = True
            terminated = True

        # Obstacle collision check
        if not collision:
            for obs in self.obstacles:
                if obs.check_collision(self.pos, self.drone_radius):
                    collision = True
                    terminated = True
                    break

        # 4. Goal Check
        curr_distance = float(np.linalg.norm(self.goal_pos - self.pos))
        goal_reached = curr_distance <= self.goal_reach_radius

        # 5. Reward Calculation
        # Progress reward: positive for moving closer to goal
        dist_progress = (self.prev_distance - curr_distance)
        reward = dist_progress * 3.5

        # Heading alignment bonus (encourages facing the goal and prevents circling)
        rel_h = np.array([self.goal_pos[0] - self.pos[0], self.goal_pos[2] - self.pos[2]], dtype=np.float32)
        rel_h_norm = np.linalg.norm(rel_h)
        if rel_h_norm > 1e-4:
            heading_dir = np.array([math.sin(self.yaw), math.cos(self.yaw)], dtype=np.float32)
            goal_dir = rel_h / rel_h_norm
            alignment = float(np.dot(heading_dir, goal_dir))
            reward += alignment * 0.08

        # Step penalty to encourage shortest path trajectory
        reward -= 0.01

        # Smooth flight regularization (small penalty for extreme actuation)
        action_penalty = float(0.005 * np.sum(action ** 2))
        reward -= action_penalty

        if goal_reached:
            reward += 100.0
            terminated = True

        if collision:
            reward -= 50.0

        self.prev_distance = curr_distance
        truncated = bool(self.step_count >= self.max_steps)

        # Update sensor rays
        sensors = self._cast_sensor_rays()

        # Proximity hazard penalty when getting dangerously close to obstacles (< 2.5m)
        min_sensor = float(np.min(sensors))
        if min_sensor < 2.5:
            reward -= (2.5 - min_sensor) * 0.12

        obs = self._get_obs(sensors)

        info = {
            "distance_to_goal": curr_distance,
            "goal_reached": goal_reached,
            "collision": collision,
            "drone_pos": self.pos.tolist(),
            "linear_speed": float(np.linalg.norm(self.vel)),
            "step": self.step_count
        }

        return obs, float(reward), terminated, truncated, info


# Register environment with Gymnasium
gym.register(
    id="DroneNavEnv-v1",
    entry_point="backend.drone_env:DroneNavEnv",
    max_episode_steps=600,
)
