import numpy as np
from dm_control import composer
from dm_control.utils import transformations
from car import Car, Button
from dm_control.locomotion.arenas import floors
import time
np.random.seed(42)

DEFAULT_CONTROL_TIMESTEP = 0.05
DEFAULT_PHYSICS_TIMESTEP = 0.005
GOAL_POS_HEIGHT = 0.1
GOAL_REWARD = 10000000000
COLLISION_REWARD = -100000000
GOAL_DISTANCE_THRESHOLD = 1.0
OBSTACLE_DIMS = (0.2, 0.2, 0.4)
OBSTACLE_COLOR = (0.3, 0.3, 0.8, 1)
GRAVITY = (0, 0, -9.81)

class CarTask(composer.Task):

    def __init__(self, num_obstacles=1, physics_timestep=DEFAULT_PHYSICS_TIMESTEP, control_timestep=DEFAULT_CONTROL_TIMESTEP, include_camera=True, goal_position=None, scenario="no-goal"):
        self._arena = floors.Floor()

        self._arena.mjcf_model.option.gravity = GRAVITY
        self._car = Car()
        self.num_obstacles = num_obstacles
        self._obstacle_geoms = []
        # self._add_obstacles()
        self._add_spherical_obstacles()
        
        self._arena.add_free_entity(self._car)
        self.goal_position = goal_position
        self.goal_distance_threshold = GOAL_DISTANCE_THRESHOLD
        self.scenario = scenario
        if self.goal_position is not None:
            self._add_goal_and_light()

        self._car.observables.enable_all()
        if not include_camera:
            self._car.observables.get_observable('realsense_camera').enabled = False

        self.set_timesteps(control_timestep, physics_timestep)
        self._last_positions = np.empty((500, 2), dtype=np.float32)
        self._last_positions.fill(np.inf)
        self._last_vel = np.zeros(2)

    def _add_obstacles(self):
        self.obstacle_pos = self.generate_random_obstacles(self.num_obstacles)
        for pos in self.obstacle_pos:
            geom = self._arena.mjcf_model.worldbody.add(
                'geom', type="box", mass="1.2", contype="1", friction="0.4 0.005 0.00001",
                conaffinity="1", size=OBSTACLE_DIMS, rgba=OBSTACLE_COLOR, pos=[pos[0], pos[1], OBSTACLE_DIMS[2] / 2]
            )
            self._obstacle_geoms.append(geom)

    def _add_spherical_obstacles(self):
        """
        Adds random spherical obstacles to the environment.
        """
        self.obstacles = self.generate_random_spherical_obstacles(self.num_obstacles)
        
        # self.obstacles = self.generate_racetrack(
        #         outer_length=8.0, outer_width=8.0, 
        #         inner_length=4.0, inner_width=4.0, 
        #         sphere_radius=0.3, spacing=0.1
        #     )

        # self.obstacles = self.generate_maze(
        #         maze_size=(10, 10),  # Maze dimensions
        #         cell_size=1.0,       # Size of each cell in the grid
        #         sphere_radius=0.2,   # Radius of obstacles
        #         spacing=0.5,         # Spacing between obstacles
        #         complexity=0.8,      # Complexity of maze
        #         density=0.6          # Density of obstacles
        #     )

        for pos, radius in self.obstacles:
            geom = self._arena.mjcf_model.worldbody.add(
                'geom', type="sphere", size=[radius], pos=pos, mass="1.0",
                contype="1", conaffinity="1", friction="0.5 0.005 0.00001", rgba=OBSTACLE_COLOR
            )
            self._obstacle_geoms.append(geom)

    def _add_goal_and_light(self):
        spawn_goal = self._arena.mjcf_model.worldbody.add('site', pos=(self.goal_position[0], self.goal_position[1], GOAL_POS_HEIGHT))
        self._button = Button()
        spawn_goal.attach(self._button.mjcf_model)
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return {}

    def get_obstacles(self):
        # return self.obstacle_pos
        return self.obstacles
    
    def get_obstacle_geoms(self):
        return self._obstacle_geoms

    def generate_random_obstacles(self, n):
        """Generates n random obstacle positions within the arena."""
        space_range = [-self._arena.size[0], self._arena.size[0]]
        obstacles = []
        for _ in range(n):
            x = np.random.uniform(space_range[0], space_range[1])
            y = np.random.uniform(space_range[0], space_range[1])
            obstacles.append(np.array([x, y]))

        # In addition to random obstacles, add obstacles along the borders of the arena
        for x in space_range:
            for y in range(3 * space_range[0], 3 * space_range[1] + 1):
                y_val = y / 3
                obstacles.append(np.array([x, y_val]))

        for y in space_range:
            for x in range(3 * space_range[0], 3 * space_range[1] + 1):
                x_val = x / 3
                obstacles.append(np.array([x_val, y]))
        
        return obstacles
    
    def generate_random_spherical_obstacles(self, n):
        """
        Generates n random spherical obstacles with varying radii and positions.
        Some obstacles will "float" above the ground.

        :param n: Number of obstacles to generate.
        :return: List of tuples (position, radius) for each obstacle.
        """
        obstacles = []
        for _ in range(n):
            x = np.random.uniform(-self._arena.size[0], self._arena.size[0])
            y = np.random.uniform(-self._arena.size[1], self._arena.size[1])
            z = np.random.uniform(0.6, 1.2)  # Random height above ground
            radius = np.random.uniform(0.3, 0.5)  # Random radius
            
            # If obstacle is too close to the origin, regenerate
            while np.linalg.norm([x, y]) < 2.0:
                x = np.random.uniform(-self._arena.size[0], self._arena.size[0])
                y = np.random.uniform(-self._arena.size[1], self._arena.size[1])

            obstacles.append((np.array([x, y, z]), radius))

        return obstacles

    def generate_racetrack(self, outer_length, outer_width, inner_length, inner_width, sphere_radius=0.2, spacing=0.5):
        """
        Generates a racetrack with an inner and outer rectangle of spherical obstacles.
        The car will always start between the inner and outer rectangles.

        :param outer_length: Length of the outer racetrack boundary.
        :param outer_width: Width of the outer racetrack boundary.
        :param inner_length: Length of the inner racetrack boundary.
        :param inner_width: Width of the inner racetrack boundary.
        :param sphere_radius: Radius of each spherical obstacle.
        :param spacing: Spacing between the centers of adjacent obstacles.
        :return: List of tuples (position, radius) for each obstacle.
        """
        obstacles = []

        # Outer rectangle
        num_spheres_outer_length = int(outer_length // spacing)
        num_spheres_outer_width = int(outer_width // spacing)
        half_avg_width = (outer_width + inner_width) / 4
        half_avg_height = (outer_length + inner_length) / 4

        # Add obstacles for the outer rectangle
        for i in range(num_spheres_outer_length + 1):
            x = i * spacing - outer_length / 2
            y1 = -outer_width / 2  # Bottom side
            y2 = outer_width / 2   # Top side
            obstacles.append((np.array([x + half_avg_height, y1 + half_avg_width, sphere_radius]), sphere_radius))
            obstacles.append((np.array([x + half_avg_height, y2 + half_avg_width, sphere_radius]), sphere_radius))

        for j in range(num_spheres_outer_width + 1):
            y = j * spacing - outer_width / 2
            x1 = -outer_length / 2  # Left side
            x2 = outer_length / 2   # Right side
            obstacles.append((np.array([x1 + half_avg_height, y + half_avg_width, sphere_radius]), sphere_radius))
            obstacles.append((np.array([x2 + half_avg_height, y + half_avg_width, sphere_radius]), sphere_radius))

        # Inner rectangle
        num_spheres_inner_length = int(inner_length // spacing)
        num_spheres_inner_width = int(inner_width // spacing)

        # Add obstacles for the inner rectangle
        for i in range(num_spheres_inner_length + 1):
            x = i * spacing - inner_length / 2
            y1 = -inner_width / 2  # Bottom side
            y2 = inner_width / 2   # Top side
            obstacles.append((np.array([x + half_avg_height, y1 + half_avg_width, sphere_radius]), sphere_radius))
            obstacles.append((np.array([x + half_avg_height, y2 + half_avg_width, sphere_radius]), sphere_radius))

        for j in range(num_spheres_inner_width + 1):
            y = j * spacing - inner_width / 2
            x1 = -inner_length / 2  # Left side
            x2 = inner_length / 2   # Right side
            obstacles.append((np.array([x1 + half_avg_height, y + half_avg_width, sphere_radius]), sphere_radius))
            obstacles.append((np.array([x2 + half_avg_height, y + half_avg_width, sphere_radius]), sphere_radius))

       
        obstacles.append((np.array([half_avg_width, 0, 1]), 0.5))
        obstacles.append((np.array([0, half_avg_height, 1]), 0.5))
        obstacles.append((np.array([-half_avg_width, 0, 1]), 0.5))

        return obstacles

    def generate_maze(self,maze_size, cell_size, sphere_radius=0.2, spacing=0.5, complexity=0.75, density=0.75):
        """
        Generate a maze-like structure for the car and manipulator.
        
        :param maze_size: Tuple of (length, width) for the maze grid.
        :param cell_size: Size of each cell in the maze.
        :param sphere_radius: Radius of spherical obstacles.
        :param spacing: Spacing between adjacent obstacles.
        :param complexity: Complexity of the maze (0 to 1).
        :param density: Density of obstacles (0 to 1).
        :return: List of tuples (position, radius) for each obstacle.
        """
        length, width = maze_size
        obstacles = []

        # Generate a grid of cells
        num_rows = int(length // cell_size)
        num_cols = int(width // cell_size)

        # Add maze walls and gaps based on complexity and density
        for row in range(num_rows):
            for col in range(num_cols):
                if np.random.rand() < density:
                    # Place vertical walls with gaps
                    if row < num_rows - 1 and np.random.rand() > complexity:
                        x = row * cell_size + cell_size / 2
                        y = col * cell_size
                        obstacles.append((np.array([x, y, sphere_radius]), sphere_radius))
                    # Place horizontal walls with gaps
                    if col < num_cols - 1 and np.random.rand() > complexity:
                        x = row * cell_size
                        y = col * cell_size + cell_size / 2
                        obstacles.append((np.array([x, y, sphere_radius]), sphere_radius))

        # Add floating obstacles above certain cells
        for _ in range(int(num_rows * num_cols * density / 5)):
            x = np.random.uniform(-length / 2, length / 2)
            y = np.random.uniform(-width / 2, width / 2)
            z = np.random.uniform(0.5, 1.5)  # Floating height
            radius = np.random.uniform(0.2, 0.4)
            obstacles.append((np.array([x, y, z]), radius))

        return obstacles


    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._arena.initialize_episode(physics, random_state)
        self._car.set_pose(physics, np.array([0, 0, 0]), transformations.euler_to_quat([0, 0, 0]))
        self._car_geomids = set(physics.bind(self._car.mjcf_model.find_all('geom')).element_id)
        
        wheel_geom_names = {'buddy_wheel_fl', 'buddy_wheel_fr', 'buddy_wheel_bl', 'buddy_wheel_br'}
        self._car_geomids_except_wheels = set(
            physics.bind(
                geom for geom in self._car.mjcf_model.find_all('geom')
                if geom.name not in wheel_geom_names
            ).element_id
        )

        self._last_positions.fill(np.inf)

        # Reposition obstacles (so that obstacles are randomized at the start of each episode)
        # self.obstacle_pos = self.generate_random_obstacles(self.num_obstacles)
        # for i, pos in enumerate(self.obstacle_pos):
        #     self._obstacle_geoms[i].pos = [pos[0], pos[1], OBSTACLE_DIMS[2] / 2]

        # Reposition obstacles (randomize at the start of each episode)
        self.obstacles = self.generate_random_spherical_obstacles(self.num_obstacles)

        for i, (pos, radius) in enumerate(self.obstacles):
            self._obstacle_geoms[i].pos = pos
            self._obstacle_geoms[i].size = [radius]

        # self.obstacles = self.generate_racetrack(
        #         outer_length=8.0, outer_width=8.0, 
        #         inner_length=4.0, inner_width=4.0, 
        #         sphere_radius=0.3, spacing=0.1
        #     )

    def should_terminate_episode(self, physics):
        return False

    def after_step(self, physics, random_state):
        car_pos, car_quat = self._car.get_pose(physics)
        self._last_positions = np.roll(self._last_positions, -1, axis=0)
        self._last_positions[-1] = car_pos[:2]

    def get_reward(self, physics):
        reward = 0
        reward += self._compute_collision_reward(physics)
        # reward += self._compute_velocity_reward(physics)
        reward += self._compute_dist_from_origin_reward(physics)

        if self.scenario == "goal":
            car_body = self._car.mjcf_model.find('body', 'buddy')
            if car_body is None:
                raise ValueError('Cannot find the main body of the car (body name: buddy).')

            car_pos = physics.bind(car_body).xpos
            dist_to_goal = np.linalg.norm(car_pos[:2] - self.goal_position[:2])

            reward += self._compute_distance_reward(dist_to_goal)
            reward += self._compute_goal_reward(dist_to_goal)

        return reward

    def _compute_distance_reward(self, dist_to_goal):
        return -dist_to_goal

    def _compute_goal_reward(self, dist_to_goal):
        if dist_to_goal < self.goal_distance_threshold:
            print("Goal reached!")
            return GOAL_REWARD
        return 0

    def _compute_collision_reward(self, physics): # Negative reward for collision
        car_geom_ids = set(physics.bind(self._car.mjcf_model.find_all('geom')).element_id)
        obstacle_geom_ids = set(physics.bind(self._obstacle_geoms).element_id)

        wheel_geom_ids = set()

        for geom in self._car.mjcf_model.find_all('geom'):
            if "buddy_wheel" in str(geom.dclass) and geom.type == "mesh":
                element_id = physics.bind(geom).element_id
                wheel_geom_ids.add(element_id)
                wheel_geom_ids.add(element_id - 1)

        car_geomids_except_wheels = car_geom_ids - wheel_geom_ids
    
        floor_geom_ids = set()
        for geom in self._arena.mjcf_model.find_all('geom'):
            if geom.name == 'groundplane':
                floor_geom_ids.add(physics.bind(geom).element_id)

        for contact in physics.data.contact:
            if (contact.geom1 in car_geom_ids and contact.geom2 in obstacle_geom_ids) or \
               (contact.geom2 in car_geom_ids and contact.geom1 in obstacle_geom_ids):
                return COLLISION_REWARD
            
            if (contact.geom1 in car_geomids_except_wheels and contact.geom2 in floor_geom_ids) or \
            (contact.geom2 in car_geomids_except_wheels and contact.geom1 in floor_geom_ids):
                return COLLISION_REWARD
            
        return 0

    def _compute_dist_from_origin_reward(self, physics):
        car_pos, _ = self._car.get_pose(physics)
        dist = np.linalg.norm(car_pos[:2])
        return np.exp(dist) # reward the car for moving away from the origin
    
    def _compute_velocity_reward(self, physics):
        car_vel = self._car.observables.body_vel_2d(physics)
        reward = np.linalg.norm(car_vel)
        reward = np.exp(10*reward)
        # print("End Effector Position: ", self._get_end_effector_position(physics))
        # print("Velocity Reward: ", reward)
        return reward # reward the car for moving faster
    
    def _get_end_effector_position(self, physics):
        """Get the position of the end effector in the world frame."""
        # Find the end effector body
        end_effector_body = self._car.mjcf_model.find('body', 'tool0_link')
        if end_effector_body is None:
            raise ValueError("End effector (tool0_link) body not found in the MJCF model.")
        
        # Get the world position of the end effector
        end_effector_pos = physics.bind(end_effector_body).xpos  # Shape: (3,)
        return end_effector_pos