from scipy.spatial import KDTree
import random
random.seed(42)
import matplotlib.pyplot as plt
from collision_checker import CollisionChecker

class RRT:
    def __init__(self, start, goal, obstacles, bounds, max_iter=1000, step_size=1, goal_tolerance=1.0):
        """
        Initialize the RRT algorithm.
        :param start: Initial configuration (base + joint angles) as a tuple (x, y, joint_angles).
        :param goal: Target configuration (base + joint angles).
        :param obstacles: List of obstacle positions (x, y) from `task.get_obstacles()`.
        :param bounds: Bounds of the configuration space [(x_min, x_max), (y_min, y_max), (joint_min, joint_max)].
        :param max_iter: Maximum iterations for the RRT algorithm.
        :param step_size: Step size for each extension.
        :param goal_tolerance: Distance tolerance to consider reaching the goal.
        """
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.tree = {start: None}  # RRT tree, mapping configurations to parents

        car_bounds = (-0.5, 0.5, -0.5, 0.5, 0.0, 0.1)  # Car bounds
        ground_level = 0.0  # Ground level
        self.collision_checker = CollisionChecker(car_bounds, ground_level, obstacles)

    def sample(self):
        """Randomly sample a point in the configuration space."""
        if random.random() < 0.1:  # 10% bias towards the goal
            print("goal sample")
            return self.goal
        return tuple(
            random.uniform(*bound) for bound in self.bounds
        )

    def nearest(self, point):
        """Find the nearest point in the tree to the given point."""
        kdtree = KDTree(list(self.tree.keys()))
        _, idx = kdtree.query(point)
        return list(self.tree.keys())[idx]

    def steer(self, from_point, to_point):
        """Extend from 'from_point' towards 'to_point' by step size."""
        direction = [t - f for f, t in zip(from_point, to_point)]
        norm = sum(d ** 2 for d in direction) ** 0.5
        direction = [d / norm for d in direction]
        return tuple(
            f + self.step_size * d for f, d in zip(from_point, direction)
        )

    def is_collision_free(self, config):
        """Check if a configuration is collision-free."""

        # First, check if the base hits any obstacles
        x, y = config[:2]
        for obs in self.obstacles:
            obs_pos = obs[0]
            obs_x, obs_y, obs_z = obs_pos[:3]
            obs_radius = obs[1]
            if (x - obs_x) ** 2 + (y - obs_y) ** 2 + (obs_z) ** 2 < (obs_radius+0.5) ** 2:
                return False

        # Next, check if the arm hits the ground, car, or obstacles
        car_pose = config[:3]  # Extract car's pose (x, y, yaw)
        joint_angles = config[3:]  # Extract arm joint angles

        return not self.collision_checker.check_collision(joint_angles, car_pose)    

    def plan(self):
        """Plan the path using RRT."""
        for _ in range(self.max_iter):
            sample = self.sample()
            nearest = self.nearest(sample)
            new_point = self.steer(nearest, sample)

            if not self.is_collision_free(new_point):
                continue
            
            self.tree[new_point] = nearest

            if sum((g - n) ** 2 for g, n in zip(self.goal, new_point)) ** 0.5 < self.goal_tolerance:
                if self.goal not in self.tree:
                    self.tree[self.goal] = new_point
                return self.reconstruct_path()

        return None  # Failed to find a path

    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        path = []
        current = self.goal
        while current:
            path.append(current)
            current = self.tree[current]
        return path[::-1]  # Reverse the path

class RRTStar(RRT):
    def __init__(self, start, goal, obstacles, bounds, max_iter=1000, step_size=1, goal_tolerance=1.0, radius=2.0):
        """
        Initialize the RRT* algorithm.
        :param radius: Neighborhood radius for rewiring.
        """
        super().__init__(start, goal, obstacles, bounds, max_iter, step_size, goal_tolerance)
        self.radius = radius  # Radius for checking neighbors

    def get_nearby_nodes(self, new_point):
        """Find nodes in the neighborhood of the new point within the radius."""
        kdtree = KDTree(list(self.tree.keys()))
        indices = kdtree.query_ball_point(new_point, self.radius)
        return [list(self.tree.keys())[i] for i in indices]

    def cost(self, point):
        """Compute the cost from the start to the given point."""
        visited = set()  # To detect loops
        cost = 0
        while point:
            if point in visited:
                raise ValueError(f"Infinite loop detected at node: {point}")
            visited.add(point)
            
            parent = self.tree.get(point)
            if parent is None:
                break
            cost += sum((p - q) ** 2 for p, q in zip(point, parent)) ** 0.5
            point = parent
        return cost

    def rewire(self, new_point, neighbors):
        """Rewire the tree to improve path cost."""
        for neighbor in neighbors:
            new_cost = self.cost(new_point) + sum((n - p) ** 2 for n, p in zip(neighbor, new_point)) ** 0.5
            if new_cost < self.cost(neighbor) and self.is_collision_free(neighbor):
                if neighbor != new_point:  # Ensure no self-loop
                    self.tree[neighbor] = new_point

    def plan(self):
        """Plan the path using RRT*."""
        for _ in range(self.max_iter):
    
            sample = self.sample()
    
            nearest = self.nearest(sample)

            new_point = self.steer(nearest, sample)

            if not self.is_collision_free(new_point):
                continue

            if new_point in self.tree:
                continue  # Avoid duplicate nodes

            self.tree[new_point] = nearest
    
            neighbors = self.get_nearby_nodes(new_point)
            
            # Reconnect to the best parent
            best_cost = self.cost(nearest) + sum((p - q) ** 2 for p, q in zip(new_point, nearest)) ** 0.5
            best_parent = nearest
            for neighbor in neighbors:
                cost = self.cost(neighbor) + sum((n - p) ** 2 for n, p in zip(new_point, neighbor)) ** 0.5
                if cost < best_cost and self.is_collision_free(new_point):
                    if neighbor != new_point:  # Ensure no self-loop
                        best_parent = neighbor
                        best_cost = cost
            self.tree[new_point] = best_parent

            # Rewire neighbors
            self.rewire(new_point, neighbors)

            # Check if the goal is reached
            if sum((g - n) ** 2 for g, n in zip(self.goal, new_point)) ** 0.5 < self.goal_tolerance:
                if self.goal not in self.tree:
                    self.tree[self.goal] = new_point
                return self.reconstruct_path()

        return None  # Failed to find a path
