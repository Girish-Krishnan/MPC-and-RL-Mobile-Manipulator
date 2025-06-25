import os
import argparse
import yaml
import torch
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import CarEnv
from dm_control import viewer, composer
import numpy as np
import casadi as ca
from scaled_fk import casadi_forward_all_links

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from feature_extractors.simple_vit import ViTFeaturesExtractor
import torch as th
import torch.nn as nn
import cv2
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from rrt import RRT, RRTStar
from matplotlib import pyplot as plt
import time

def load_yaml(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        observations = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        
        print(f"Step: {self.num_timesteps}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print("------")
        
        return True

current_timestep_num = 0
current_time = 0

def policy(timestep, model, model_inputs):
    global current_timestep_num
    vec_obs = []
    depth_obs = None
    point_cloud_obs = None

    # Collect vector observations
    if "pose" in model_inputs:
        pose_obs = timestep.observation['car/body_pose_2d']
        vec_obs += list(pose_obs[:3])
    if "velocity" in model_inputs:
        velocity = timestep.observation['car/body_vel_2d']
        vec_obs += [np.linalg.norm(velocity)]
    if "steering" in model_inputs:
        vec_obs += [timestep.observation['car/steering_pos'][0]]
    if "joint_positions" in model_inputs:
        vec_obs += [timestep.observation['car/shoulder_pan_joint_pos'][0],
                    timestep.observation['car/shoulder_lift_joint_pos'][0],
                    timestep.observation['car/elbow_joint_pos'][0],
                    timestep.observation['car/wrist_1_joint_pos'][0],
                    timestep.observation['car/wrist_2_joint_pos'][0],
                    timestep.observation['car/wrist_3_joint_pos'][0]]

    # Collect depth observation
    if "depth" in model_inputs:
        depth_obs = timestep.observation['car/realsense_camera'].astype(np.float32)
        cv2.imshow("Depth Map", cv2.convertScaleAbs(depth_obs, alpha=0.15))
        cv2.waitKey(1)

    # Collect point cloud observation
    if "point_cloud" in model_inputs:
        point_cloud_obs = timestep.observation['car/compute_point_cloud']

    # Construct observation dictionary
    observation = {}
    if len(vec_obs) > 0:
        observation["vec"] = np.array(vec_obs)
    if depth_obs is not None:
        observation["depth"] = depth_obs
    if point_cloud_obs is not None:
        observation["point_cloud"] = point_cloud_obs

    # Predict action
    action, _ = model.predict(observation, deterministic=True) 

    # Plot current position
    current_base_pos = timestep.observation['car/body_pose_2d'][:2]
    live_current_position_plot(current_base_pos)

    # Print reward
    current_timestep_num += 1

    return action

def compute_rrt_path(task, start, goal):
    """
    Compute the RRT path from start to goal.
    :param task: Task object to access obstacles.
    :param start: Start configuration (base position + joint angles).
    :param goal: Goal configuration (base position + joint angles).
    :return: List of waypoints representing the path, or None if planning failed.
    """
    obstacles = task.get_obstacles()
    obstacle_positions = [obs[0] for obs in obstacles]
    obstacle_radii = [obs[1] for obs in obstacles]
    bounds = [
        (-10, 10), (-10, 10),  # Base x, y bounds
        (-np.pi, np.pi), # Base orientation bounds
        (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # Joint angle bounds
        (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
    ]

    rrt = RRTStar(start, goal, obstacles, bounds, max_iter=40000, step_size=0.5)
    path = rrt.plan()

    if path is None:
        print("RRT failed to find a path.")
        return None
    else:
        print("RRT path computed successfully.")

    # Plot the path
    marker_sizes = [1000 * r for r in obstacle_radii]  # Scale radii for marker sizes
    plt.figure(figsize=(8, 8))
    plt.scatter(
        [p[0] for p in obstacle_positions], 
        [p[1] for p in obstacle_positions], 
        color='red', 
        s=marker_sizes,  # Use scaled radii for marker sizes
        label='Obstacles'
    )
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
    plt.plot([p[0] for p in path], [p[1] for p in path], color='blue', linewidth=2, label='RRT Path')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Path Planning: Obstacles and Goal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()
    
    return path

def live_path_plot(path_points, current_position):
    x, y = zip(*path_points)  # Extract all path points
    plt.scatter(x, y, color="green", label="Planned Path")
    plt.scatter(current_position[0], current_position[1], color="red", label="Current Position")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.001)  # Update the plot

def live_current_position_plot(current_position):
    plt.scatter(current_position[0], current_position[1], color="red", label="Current Position")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.001)  # Update the plot

current_index = 0  # Initialize waypoint index

def rrt_policy(timestep, precomputed_path):
    global current_index
    
    # Extract the current base position and orientation
    current_base_pos = timestep.observation['car/body_pose_2d'][:2]
    path_points = [config[:2] for config in precomputed_path]
    live_path_plot(path_points, current_base_pos)

    current_theta = timestep.observation['car/body_pose_2d'][2]

    # Extract current joint angles
    current_joint_angles = [
        timestep.observation['car/shoulder_pan_joint_pos'][0],
        timestep.observation['car/shoulder_lift_joint_pos'][0],
        timestep.observation['car/elbow_joint_pos'][0],
        timestep.observation['car/wrist_1_joint_pos'][0],
        timestep.observation['car/wrist_2_joint_pos'][0],
        timestep.observation['car/wrist_3_joint_pos'][0],
    ]

    # Check if we have reached the current waypoint
    next_config = precomputed_path[current_index]
    target_base_pos = next_config[:2]
    target_theta = next_config[2]
    target_joint_angles = next_config[3:]

    # Compute base direction and control
    direction_to_target = np.array(target_base_pos) - np.array(current_base_pos)
    distance_to_target = np.linalg.norm(direction_to_target)

    # Check if we are close enough to the current waypoint
    if distance_to_target < 0.25:
        
        if current_index >= len(precomputed_path) - 1:
            print("Goal reached!")
            return np.concatenate(([0.0, 0.0], target_joint_angles))  # Stop at the goal
        
        current_index += 1 # Move to the next waypoint

    # Compute heading error
    target_heading = np.arctan2(direction_to_target[1], direction_to_target[0])
    heading_error = normalize_angle(target_heading - current_theta)

    # Base controls
    throttle = np.clip(1.0 * distance_to_target, 0, 10)  # Scale throttle
    steering = np.clip(3.0 * heading_error, -0.38, 0.38)  # Scale steering

    # Joint controls (PID or simple proportional control)
    # print("Current Joint Angles:", current_joint_angles)
    # print("Target Joint Angles:", target_joint_angles)
    # joint_actions = 1.0 * (np.array(target_joint_angles) - np.array(current_joint_angles))
    # print("Joint Actions:", joint_actions)
    # joint_actions = np.clip(joint_actions, -np.pi / 4, np.pi / 4)  # Clip to joint limits

    # # Concatenate base and joint controls
    # joint_actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No joint movement
    joint_actions = target_joint_angles
    action = np.concatenate(([steering, throttle], joint_actions))

    return action

def normalize_angle(angle):
    """Normalize an angle to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


### MPC
def precompute_obstacle_constraints(obstacles):
    """
    Precompute symbolic obstacle constraints for static obstacles.
    :param obstacles: List of tuples [(pos, radius), ...].
    :return: Precomputed symbolic expressions for obstacle constraints.
    """
    obstacle_constraints = []
    for obs_pos, obs_radius in obstacles:
        obs_pos_sym = ca.MX(obs_pos[:2])  # Static obstacle position
        obs_constraint = lambda joint_pos: ca.sumsqr(joint_pos - obs_pos_sym)
        obstacle_constraints.append((obs_constraint, obs_radius + 0.3))  # Add buffer
    return obstacle_constraints

def mpc_controller(current_state, target_path, dynamics, horizon, bounds, obstacles, dt=1.0, prev_solution=None):

    state_dim = len(current_state)
    control_dim = 8  # 2 for car + 6 for arm

    # Precompute invariant constraints
    obstacle_constraints = precompute_obstacle_constraints(obstacles)

    # Define CasADi variables
    x = ca.MX.sym("x", horizon + 1, state_dim)  # States
    u = ca.MX.sym("u", horizon, control_dim)   # Controls
    x_init = ca.MX.sym("x_init", state_dim)    # Initial state
    cost = 0  # Cost function
    epsilon = 1e-3
    g = []    # Constraints
    lbg = []  # Lower bounds
    ubg = []  # Upper bounds

    # Initial state constraint (equality)
    g.append(ca.reshape(x[0, :], -1, 1) - x_init)
    lbg.extend([-epsilon] * state_dim)
    ubg.extend([epsilon] * state_dim)

    for t in range(horizon):
        # Dynamics constraint (equality)
        next_state = dynamics(x[t, :], u[t, :], dt)
        g.append(ca.reshape(x[t + 1, :], -1, 1) - next_state)
        lbg.extend([-epsilon] * state_dim)
        ubg.extend([epsilon] * state_dim)

        # State bounds (inequality)
        g.append(ca.reshape(x[t + 1, :], -1, 1))
        lbg.extend(bounds["state_min"])
        ubg.extend(bounds["state_max"])

        # Control bounds (inequality)
        g.append(ca.reshape(u[t, :], -1, 1))
        lbg.extend(bounds["control_min"])
        ubg.extend(bounds["control_max"])

        # Obstacle avoidance (inequality)
        for obs_constraint, buffer in obstacle_constraints:
            g.append(obs_constraint(ca.reshape(x[t + 1, :2], -1, 1)))
            lbg.append(buffer) 
            ubg.append(ca.inf)

        next_state_joint_positions = casadi_forward_all_links(x[t + 1, 2:])
        next_state_joint_positions = [T[:3, 3] for T in next_state_joint_positions]

        for i, joint_pos in enumerate(next_state_joint_positions):
            next_state_joint_positions[i] = ca.vertcat(
                ca.cos(x[t + 1, 2]) * joint_pos[0] - ca.sin(x[t + 1, 2]) * joint_pos[1] + x[t + 1, 0],
                ca.sin(x[t + 1, 2]) * joint_pos[0] + ca.cos(x[t + 1, 2]) * joint_pos[1] + x[t + 1, 1],
                joint_pos[2]
            )

        for joint_pos in next_state_joint_positions:
            # Obstacle avoidance for each joint
            for obs_constraint, buffer in obstacle_constraints:
                g.append(obs_constraint(ca.reshape(joint_pos[:2], -1, 1)))
                lbg.append(buffer)
                ubg.append(ca.inf)

            # Ensure the arm doesn't collide with the ground
            g.append(joint_pos[2])
            lbg.append(0.2)
            ubg.append(ca.inf)

        # Path tracking cost
        if t < len(target_path):
            cost += ca.sumsqr(ca.reshape(x[t + 1, :2], -1, 1) - target_path[t][:2])
            cost += ca.sumsqr(ca.reshape(x[t + 1, 3:], -1, 1) - target_path[t][3:])

        # Control smoothness cost
        cost += ca.sumsqr(u[t, :])

    # Flatten constraints
    g = ca.vertcat(*g)

    # Define the optimization problem
    decision_vars = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))
    nlp = {"x": decision_vars, "f": cost, "g": g, "p": x_init}

    # Create solver
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.linear_solver": "mumps", "expand": True}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Initial guess and bounds
    if prev_solution is not None:
        x0 = ca.DM(prev_solution)
    else:
        x0 = ca.DM.zeros((horizon + 1) * state_dim + horizon * control_dim)

    lbg = ca.DM(lbg)
    ubg = ca.DM(ubg)

    # Solve the optimization problem
    try:
        solution = solver(
            x0=x0,
            p=current_state,
            lbg=lbg,
            ubg=ubg
        )
    except RuntimeError as e:
        print("Solver failed:", e)
        return np.zeros(control_dim), None

    # Extract the first control action
    solution_x = solution["x"].full().flatten()
    solution_u = solution_x[-horizon * control_dim:]

    control_action = [solution_u[i * horizon] for i in range(control_dim)]

    return np.array(control_action), solution_x  # Return solution_x for warm-starting



def car_arm_dynamics(state, control, dt=1.0):
    """
    Simulates the car and arm dynamics for one timestep using CasADi symbolic operations.
    :param state: Current state [x, y, theta, joint_angles...].
    :param control: Control inputs [steering, throttle, joint movements...].
    :param dt: Time step duration.
    :return: Next state after applying control.
    """
    # Extract state variables
    x, y, theta = state[0], state[1], state[2]
    joint_angles = state[3:]

    # Extract control inputs
    steering = control[0]
    throttle = control[1]
    joint_controls = control[2:]

    # Car dynamics
    dx = throttle * ca.cos(theta) * dt
    dy = throttle * ca.sin(theta) * dt
    dtheta =  dt * throttle * ca.tan(steering) / 0.2965

    # Manipulator dynamics (joint angle updates)
    next_joint_angles = joint_controls # this is joint control

    # Transpose next joint angles
    next_joint_angles = ca.reshape(next_joint_angles, -1, 1)

    # Combine updated car state and joint angles
    next_state = ca.vertcat(x + dx, y + dy, theta + dtheta, next_joint_angles)
    return next_state

visited_points = set()
prev_solution = None  # Store the previous solution globally

def mpc_policy(timestep, target_path, obstacles, bounds, horizon=2):
    global prev_solution
    global visited_points

    current_joint_angles = np.array([
        timestep.observation['car/shoulder_pan_joint_pos'][0],
        timestep.observation['car/shoulder_lift_joint_pos'][0],
        timestep.observation['car/elbow_joint_pos'][0],
        timestep.observation['car/wrist_1_joint_pos'][0],
        timestep.observation['car/wrist_2_joint_pos'][0],
        timestep.observation['car/wrist_3_joint_pos'][0],
    ])

    current_state = np.concatenate((
        timestep.observation['car/body_pose_2d'],
        current_joint_angles    
    ))

    for point in target_path[:-1]:
        if np.linalg.norm(point[:2] - current_state[:2]) < 2.0:
            print("Visited point:", point)
            visited_points.add(point)

    updated_path = [point for point in target_path if point not in visited_points]

    control_action, solution_x = mpc_controller(
        current_state, updated_path, car_arm_dynamics, horizon, bounds, obstacles, prev_solution=prev_solution
    )
    
    # Update previous solution for warm start
    prev_solution = solution_x

    return control_action


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)

        self.outputs = {}  # Dictionary to store layer outputs for debugging

        # Process vector inputs
        self.vec_network = None
        if 'vec' in observation_space.keys():
            self.vec_network = nn.Sequential(
                nn.Linear(observation_space['vec'].shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.vec_network[-1].register_forward_hook(self._hook("vec_features"))

        # Process depth inputs
        self.cnn = None
        depth_dim = 0
        if 'depth' in observation_space.keys():
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                depth_sample = observation_space['depth'].sample()[None, :, :, 0]
                depth_sample = torch.tensor(depth_sample).float()
                depth_sample = self.normalize_depth_map(depth_sample).unsqueeze(1)
                depth_dim = self.cnn(depth_sample).shape[1]

        # Process point cloud inputs
        self.point_cloud_extractor = None
        point_cloud_dim = 0
        if 'point_cloud' in observation_space.keys():
            self.point_cloud_extractor = SimplifiedPointNetFeatureExtractor(
                        input_dim=3, # Point cloud features (x, y, z)
                        hidden_dims=[64, 128],
                        output_dim=256,
                        aggregation="max"
                    )

            point_cloud_dim = 256

        # Determine the combined input size
        vec_dim = 64 if 'vec' in observation_space.keys() else 0
        combined_input_dim = vec_dim + depth_dim + point_cloud_dim

        # Combined network
        self.combined_network = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim),
            nn.ReLU()
        )
        self.combined_network[-1].register_forward_hook(self._hook("combined_features"))

    def _hook(self, layer_name):
        """
        Hook to store outputs for debugging purposes.
        """
        def hook(module, input, output):
            self.outputs[layer_name] = output.detach().cpu().numpy()
        return hook

    def normalize_depth_map(self, depth_map):
        """
        Normalize depth map values to the range [0, 1].
        """
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)

    def forward(self, observations):
        """
        Forward pass to process observations and extract features.
        """
        # Process vector input
        vec_features = None
        if 'vec' in observations.keys():
            vec_features = self.vec_network(observations['vec'])

        # Process depth input
        depth_features = None
        if 'depth' in observations.keys():
            depth_map = observations['depth'][:, :, :, 0]
            depth_map = self.normalize_depth_map(depth_map).unsqueeze(1)
            depth_features = self.cnn(depth_map)

        # Process point cloud input
        point_cloud_features = None
        if 'point_cloud' in observations.keys():
            point_cloud = observations['point_cloud']  # Shape: [num_envs, N, 3]
            point_cloud_features = self.point_cloud_extractor(point_cloud)

        # Concatenate features
        features = [f for f in [vec_features, depth_features, point_cloud_features] if f is not None]
        combined_features = torch.cat(features, dim=1)

        # Pass through the combined network
        final_output = self.combined_network(combined_features)
        self.outputs["final_output"] = final_output.detach().cpu().numpy()
        return final_output

class PointNetGNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, k=8, aggregation="mean"):
        """
        PointNet-based GNN Feature Extractor.
        :param input_dim: Dimension of the input point features (e.g., 3 for (x, y, z)).
        :param hidden_dims: List of hidden dimensions for the edge MLP and node updates.
        :param output_dim: Final output dimension of the feature representation.
        :param k: Number of nearest neighbors to use for graph construction.
        :param aggregation: Aggregation method ('mean', 'max', 'sum', or 'attention').
        """
        super(PointNetGNNFeatureExtractor, self).__init__()
        self.k = k
        self.aggregation = aggregation

        # MLP for edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dims[0]),  # 3 for relative coordinates
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # MLP for node updates
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dims[1] + input_dim, hidden_dims[1]),  # Combine aggregated edge features with node features
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
        )

        # Optional attention mechanism for aggregation
        if self.aggregation == "attention":
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dims[1], 64),
                nn.ReLU(),
                nn.Linear(64, 1),  # Attention weights
                nn.Sigmoid(),
            )

        # Final MLP for output features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dims[1], output_dim),
            nn.ReLU(),
        )

    def forward(self, point_cloud):
        """
        Forward pass with parallel batch processing using KNN.
        :param point_cloud: Input point cloud tensor of shape (B, N, F).
        :return: Extracted features of shape (B, N, output_dim).
        """
        batch_size, num_points, feature_dim = point_cloud.shape
        device = point_cloud.device

        # Flatten the batch for KNN
        point_cloud_flat = point_cloud.view(-1, feature_dim)  # Shape: (B * N, F)
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_points)  # Shape: (B * N,)

        # Compute KNN graph with batch awareness
        edge_index = knn_graph(point_cloud_flat, k=self.k, batch=batch_indices, loop=False)  # Shape: (2, E)

        src, dest = edge_index
        relative_coords = point_cloud_flat[dest] - point_cloud_flat[src]
        edge_features = torch.cat([point_cloud_flat[src], relative_coords], dim=-1)

        # Apply edge MLP
        edge_features = self.edge_mlp(edge_features)

        # Aggregate edge features to node features
        if self.aggregation == "mean":
            aggregated = scatter_mean(edge_features, dest, dim=0, dim_size=batch_size * num_points)
        elif self.aggregation == "max":
            aggregated, _ = scatter_max(edge_features, dest, dim=0, dim_size=batch_size * num_points)
        elif self.aggregation == "sum":
            aggregated = scatter_sum(edge_features, dest, dim=0, dim_size=batch_size * num_points)
        elif self.aggregation == "attention":
            attention_weights = self.attention_mlp(edge_features)
            edge_features_weighted = edge_features * attention_weights
            aggregated = scatter_sum(edge_features_weighted, dest, dim=0, dim_size=batch_size * num_points)
        else:
            raise ValueError("Invalid aggregation method")

        # Combine aggregated features with original node features
        node_features = torch.cat([point_cloud_flat, aggregated], dim=-1)
        node_features = self.node_mlp(node_features)

        # Apply final MLP
        final_features = self.final_mlp(node_features)

        # Reshape back to batch
        return final_features.view(batch_size, num_points, -1)  # Shape: (B, N, output_dim)

class SimplifiedPointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, aggregation="max"):
        """
        A simplified PointNet-based feature extractor for point clouds.
        :param input_dim: Dimension of input point features (e.g., 3 for x, y, z).
        :param hidden_dims: List of hidden dimensions for intermediate layers.
        :param output_dim: Dimension of the final output features.
        :param aggregation: Aggregation method: "max", "mean", or "sum".
        """
        super(SimplifiedPointNetFeatureExtractor, self).__init__()
        self.aggregation = aggregation

        # Point-wise feature transformation layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.point_mlp = nn.Sequential(*layers)

        # Global feature aggregation
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU()
        )

    def forward(self, point_cloud):
        """
        Forward pass for batched point clouds.
        :param point_cloud: Input tensor of shape (B, N, F) where B is batch size, N is the number of points,
                            and F is the feature dimension (e.g., 3 for x, y, z).
        :return: Extracted features of shape (B, output_dim).
        """
        batch_size, num_points, feature_dim = point_cloud.shape

        # Apply point-wise feature transformation
        point_features = self.point_mlp(point_cloud)  # Shape: (B, N, hidden_dims[-1])

        # Aggregate features across points
        if self.aggregation == "max":
            global_features, _ = torch.max(point_features, dim=1)  # Shape: (B, hidden_dims[-1])
        elif self.aggregation == "mean":
            global_features = torch.mean(point_features, dim=1)  # Shape: (B, hidden_dims[-1])
        elif self.aggregation == "sum":
            global_features = torch.sum(point_features, dim=1)  # Shape: (B, hidden_dims[-1])
        else:
            raise ValueError("Invalid aggregation method. Choose from 'max', 'mean', or 'sum'.")

        # Apply global MLP for final output
        final_features = self.global_mlp(global_features)  # Shape: (B, output_dim)

        return final_features

#######################################

def evaluate_model(model, vec_input, depth_input):
    observation = {"vec": vec_input, "depth": depth_input}
    model.predict(observation, deterministic=True)
    return model.policy.actor.features_extractor.outputs

def make_env(num_obstacles, rank, log_dir=None, goal_position=None, scenario="no-goal", model_inputs=CarEnv.ALL_MODEL_INPUTS):
    """
    Utility function to create a single instance of the CarEnv environment.
    :param num_obstacles: (int) Number of obstacles in the environment
    :param rank: (int) Rank of the environment (used for seeding)
    :param log_dir: (str) Directory to save logs
    :param goal_position: (list) Goal position for the environment
    :param scenario: (str) Scenario to use for the environment
    :param model_inputs: (list) List of model inputs to use
    """
    def _init():
        env = CarEnv(num_obstacles=num_obstacles, goal_position=goal_position, scenario=scenario, model_inputs=model_inputs)
        env.seed(rank)  # Seed the environment for reproducibility
        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init

def ensure_dir_exists(dir_path):
    """
    Ensure that the directory exists, create if it doesn't.
    :param dir_path: (str) Path of the directory to check/create
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model.")
    parser.add_argument('--model_type', type=str, choices=["PPO", "SAC", "RecurrentPPO"], help='Type of model to use', default="SAC")
    parser.add_argument('--scenario', type=str, choices=["goal", "no-goal"], help='Choose navigation scenario', default="no-goal")
    parser.add_argument('--goal_position', type=str, help='Comma-separated goal coordinates for goal-based navigation (e.g., "5,2")', default=None)
    parser.add_argument('--config_path', type=str, help='Path to the YAML config file', default="config/config_sac.yaml")

    parser.add_argument('--model_path', type=str, help='Path to the saved model (.zip) to continue training or for evaluation', default=None)
    parser.add_argument('--log_dir', type=str, help='Directory to save logs and models', default='./my_experiment/')
    parser.add_argument('--file_name', type=str, help='Base name for saved model and logs', default='model')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')

    args = parser.parse_args()

    # Set up directories
    tensorboard_log_dir = os.path.join(args.log_dir, "tensorboard", args.file_name)
    models_dir = os.path.join(args.log_dir, "models")
    logs_dir = os.path.join(args.log_dir, "logs")

    ensure_dir_exists(tensorboard_log_dir)
    ensure_dir_exists(models_dir)
    ensure_dir_exists(logs_dir)

    # Load YAML config
    config = load_yaml(args.config_path)
    training_params = config.get("training", {})
    model_params = config.get("model", {})

    goal_position = None
    if args.goal_position:
        goal_position = np.array([float(x) for x in args.goal_position.split(",")])

    model_class = {"PPO": PPO, "SAC": SAC, "RecurrentPPO": RecurrentPPO}[args.model_type]

    if args.eval:
        # Evaluation Mode
        model = model_class.load(args.model_path)
        
        env = CarEnv(num_obstacles=training_params["num_obstacles"], goal_position=goal_position, scenario=args.scenario, model_inputs=training_params["model_inputs"])
        task = env.task
        original_env = composer.Environment(task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)

        # print("Obstacles:", task.get_obstacle_geoms())
        # RL Method
        viewer.launch(original_env, policy=lambda timestep: policy(timestep, model, model_inputs=training_params["model_inputs"]))
        exit(0)

        # RRT planning phase
        start = (0, 0, 0, 0, 0, 0, 0, 0, 0)  # Replace with desired start
        goal = (6, 6, 0, -np.pi/2, 0, 0, 0, 0, 0)  # Replace with desired goal

        # Check that the goal is not inside an obstacle
        obstacles = task.get_obstacles()
        obstacle_poses = [obstacle[0] for obstacle in obstacles]
        obstacle_radii = [obstacle[1] for obstacle in obstacles] 

        # Before running RRT, plot obstacles and goal
        marker_sizes = [radius * 1000 for radius in obstacle_radii]  # Scale factor to match plot size
        plt.figure(figsize=(8, 8))
        plt.scatter(
            [p[0] for p in obstacle_poses], 
            [p[1] for p in obstacle_poses], 
            color='red', 
            s=marker_sizes,  # Use scaled radii for marker sizes
            label='Obstacles'
        )
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('RRT Path Planning: Obstacles and Goal')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()

        for obstacle in obstacles:
            obstacle_pos = obstacle[0]
            if np.sqrt((goal[0] - obstacle_pos[0]) ** 2 + (goal[1] - obstacle_pos[1]) ** 2 + (obstacle_pos[2] - 0) ** 2) < obstacle[1]:
                print("Goal is inside an obstacle. Exiting.")
                return

        precomputed_path = compute_rrt_path(task, start, goal)
        print("Precomputed path:", precomputed_path)    

        if precomputed_path is None:
            print("No valid path found. Exiting.")
            return

        # Launch viewer with path-following policy
        viewer.launch(
            original_env,
            policy=lambda timestep: rrt_policy(
                timestep, precomputed_path
            ),
        )

        ## MPC Method
        bounds = {
            'state_min': np.array([-10, -10, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
            'state_max': np.array([10, 10, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]),
            'control_min': np.array([-0.38, 0.0, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
            'control_max': np.array([0.38, 10.0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        }

        # Launch viewer with MPC policy
        viewer.launch(
            original_env,
            policy=lambda timestep: mpc_policy(
                timestep, precomputed_path, obstacles, bounds
            ),
        )



    else:
        # Training Mode
        env = DummyVecEnv([make_env(
                    num_obstacles=training_params["num_obstacles"], 
                    rank=i, 
                    log_dir=logs_dir, 
                    goal_position=goal_position, 
                    scenario=args.scenario, 
                    model_inputs=training_params["model_inputs"]) 
                        for i in range(training_params["num_envs"])])
        
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        device = torch.device(training_params["device"])

        if args.model_path:
            print(f"Loading model from {args.model_path}")
            model = model_class.load(args.model_path, env=env, device=device)
        else:
            print("Training a new model")

            model = model_class(
                    env=env,
                    policy_kwargs=dict(
                        features_extractor_class=ViTFeaturesExtractor,
                        features_extractor_kwargs=dict(features_dim=128),
                        net_arch=[256, 256], # 2 layers of 256 units for the latent policy network
                    ),
                    tensorboard_log=tensorboard_log_dir,
                    device=device,
                    **model_params
                )

        callback = ActionLoggerCallback(verbose=1)

        model.learn(total_timesteps=training_params["total_timesteps"], callback=callback, progress_bar=True)

        model_save_path = os.path.join(models_dir, f"{args.file_name}.zip")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        env.save(os.path.join(logs_dir, f"{args.file_name}_vecnormalize.pkl"))

if __name__ == "__main__":
    main()
