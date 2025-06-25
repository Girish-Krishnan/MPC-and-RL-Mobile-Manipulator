import numpy as np
from gym import Env, spaces
from task import CarTask
from dm_control import composer

PLANNING_HORIZON = 250

# Car states
MIN_X = -20
MAX_X = 20
MIN_Y = -20
MAX_Y = 20
MIN_THETA = -np.pi
MAX_THETA = np.pi

# Joint states
MIN_JOINT = -np.pi
MAX_JOINT = np.pi
NUM_JOINTS = 6

# Car actions
MIN_THROTTLE = 0
MAX_THROTTLE = 10
MIN_STEERING = -0.38
MAX_STEERING = 0.38

class CarEnv(Env):
    ALL_MODEL_INPUTS = [
        "pose", "velocity", "steering", "depth", "point_cloud", "joint_positions"
    ]

    def __init__(self, num_obstacles=1, goal_position=None, scenario="no-goal", model_inputs=ALL_MODEL_INPUTS):
        self.goal_position = goal_position
        self.scenario = scenario
        self.model_inputs = model_inputs
        self.task = CarTask(num_obstacles=num_obstacles, goal_position=goal_position, scenario=scenario)
        self.original_env = composer.Environment(self.task, raise_exception_on_physics_error=False, strip_singleton_obs_buffer_dim=True)
        self.mj_state = self.original_env.reset()
        self.timeElapsed = 0

        # Define action space (throttle and steering, and optionally joint positions)
        if "joint_positions" in self.model_inputs:
            self.action_space = spaces.Box(
                low=np.array([MIN_STEERING, MIN_THROTTLE] + [MIN_JOINT] * NUM_JOINTS),
                high=np.array([MAX_STEERING, MAX_THROTTLE] + [MAX_JOINT] * NUM_JOINTS),
                shape=(2 + NUM_JOINTS,),
                dtype=np.float32
            )

        else:
            self.action_space = spaces.Box(
                low=np.array([MIN_STEERING, MIN_THROTTLE]), 
                high=np.array([MAX_STEERING, MAX_THROTTLE]), 
                shape=(2,), 
                dtype=np.float32
            )

        # Define observation space
        obs_space_low = []
        obs_space_high = []
        if "pose" in self.model_inputs:
            obs_space_low += [MIN_X, MIN_Y, MIN_THETA]
            obs_space_high += [MAX_X, MAX_Y, MAX_THETA]
        if "velocity" in self.model_inputs:
            obs_space_low += [MIN_THROTTLE]
            obs_space_high += [MAX_THROTTLE]
        if "steering" in self.model_inputs:
            obs_space_low += [MIN_STEERING]
            obs_space_high += [MAX_STEERING]
        if "joint_positions" in self.model_inputs:
            obs_space_low += [MIN_JOINT] * NUM_JOINTS  # NUM_JOINTS UR5e joints
            obs_space_high += [MAX_JOINT] * NUM_JOINTS

        vec_space = (
            spaces.Box(
                low=np.array(obs_space_low), 
                high=np.array(obs_space_high), 
                dtype=np.float32
            ) if len(obs_space_low) > 0 else None
        )

        depth_space = None
        if "depth" in self.model_inputs:
            depth_shape = self.mj_state.observation['car/realsense_camera'].shape
            depth_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=depth_shape, 
                dtype=np.float32
            )

        point_cloud_space = None
        if "point_cloud" in self.model_inputs:
            depth_shape = self.mj_state.observation['car/realsense_camera'].shape
            point_cloud_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(depth_shape[0] * depth_shape[1], 3), 
                dtype=np.float32
            )

        combined_space = {}
        if vec_space is not None:
            combined_space["vec"] = vec_space
        if depth_space is not None:
            combined_space["depth"] = depth_space
        if point_cloud_space is not None:
            combined_space["point_cloud"] = point_cloud_space

        self.observation_space = spaces.Dict(combined_space)

    def step(self, action):
        self.task._car.apply_action(self.original_env.physics, action, None)
        self.mj_state = self.original_env.step(action)
        self.timeElapsed += 1
        done = self.check_complete()
        reward = self.task.get_reward(self.original_env.physics)

        state_obs = self.get_observation()
        info = {}

        return state_obs, reward, done, info

    def reset(self):
        self.done = False
        self.mj_state = self.original_env.reset()
        self.reward_accumulated = 0
        self.timeElapsed = 0
        obs = self.get_observation()

        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        self.original_env.close()

    def get_observation(self):
        vec_obs = []
        depth_obs = None
        point_cloud_obs = None

        if "pose" in self.model_inputs:
            pos = self.mj_state.observation['car/body_pose_2d']
            vec_obs += list(pos[:3])
        if "velocity" in self.model_inputs:
            velocity = self.mj_state.observation['car/body_vel_2d']
            velocity_mag = np.linalg.norm(velocity)
            vec_obs += [velocity_mag]
        if "steering" in self.model_inputs:
            steering_pos = self.mj_state.observation['car/steering_pos']
            vec_obs += [steering_pos[0]]
        if "joint_positions" in self.model_inputs:
            joint_positions = [
                self.mj_state.observation[f'car/{joint}_joint_pos'][0]
                for joint in ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
            ]
            vec_obs += joint_positions
        if "depth" in self.model_inputs:
            depth_obs = self.mj_state.observation['car/realsense_camera'].astype(np.float32)
        if "point_cloud" in self.model_inputs:
            point_cloud_obs = self.mj_state.observation['car/compute_point_cloud']

        combined_obs = {}
        if len(vec_obs) > 0:
            combined_obs["vec"] = np.array(vec_obs)
        if depth_obs is not None:
            combined_obs["depth"] = depth_obs
        if point_cloud_obs is not None:
            combined_obs["point_cloud"] = point_cloud_obs

        return combined_obs

    def check_complete(self):
        if self.timeElapsed >= PLANNING_HORIZON:
            return True
        if self.task._compute_collision_reward(self.original_env.physics) < 0:
            return True
        # if self.scenario == "goal":
        #     joint_1 = self.mj_state.observation['car/shoulder_pan_joint_pos']
        #     joint_2 = self.mj_state.observation['car/shoulder_lift_joint_pos']
        #     joint_3 = self.mj_state.observation['car/elbow_joint_pos']
        #     joint_4 = self.mj_state.observation['car/wrist_1_joint_pos']
        #     joint_5 = self.mj_state.observation['car/wrist_2_joint_pos']
        #     joint_6 = self.mj_state.observation['car/wrist_3_joint_pos']
        #     ur5e_joint_positions = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]

        #     if np.linalg.norm(self.mj_state.observation['car/body_pose_2d'][:2] - self.task.goal_position[:2]) < self.task.goal_distance_threshold and \
        #        np.linalg.norm(np.array(ur5e_joint_positions) - self.task.goal_position[3:]) < self.task.goal_distance_threshold:
        #         print("Goal reached for task!")
        #         return True
            
        return False
