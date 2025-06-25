from scaled_fk import forward_all_links, forward
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CollisionChecker:
    def __init__(self, car_bounds, ground_level, obstacles):
        """
        Initialize the collision checker.
        :param car_bounds: Bounds of the car as (x_min, x_max, y_min, y_max, z_min, z_max).
        :param ground_level: Height of the ground (z = 0 or other value).
        :param obstacles: List of spherical obstacles as (x, y, z, radius).
        """
        self.car_bounds = car_bounds
        self.ground_level = ground_level
        self.obstacles = obstacles

    def apply_car_transform(self, positions, car_pose):
        """
        Apply the car's pose (x, y, yaw) transformation to the arm link positions.
        :param positions: List of link positions from FK (Nx3 array).
        :param car_pose: (x, y, yaw) tuple representing the car's global pose.
        :return: Transformed link positions.
        """
        car_x, car_y, car_yaw = car_pose
        rotation_matrix = np.array([
            [np.cos(car_yaw), -np.sin(car_yaw), 0],
            [np.sin(car_yaw),  np.cos(car_yaw), 0],
            [0, 0, 1]
        ])
        transformed_positions = []
        for pos in positions:
            global_pos = np.dot(rotation_matrix, pos) + np.array([car_x, car_y, 0])
            transformed_positions.append(global_pos)
        return np.array(transformed_positions)

    def check_collision(self, joint_angles, car_pose):
        """
        Check if the arm collides with the ground, car, or obstacles.
        :param joint_angles: List of joint angles for the arm.
        :param car_pose: (x, y, yaw) tuple representing the car's pose.
        :return: True if a collision is detected, False otherwise.
        """
        # Get positions of all arm links in local frame
        local_positions = [T[:3, 3] for T in forward_all_links(joint_angles)]
        
        # Transform arm link positions to global frame using car's pose
        global_positions = self.apply_car_transform(local_positions, car_pose)

        for i, position in enumerate(global_positions):
            # Check ground collision
            if position[2] <= self.ground_level:
                print("Collision with ground detected!")
                return True
            
            current_local_position = local_positions[i]

            # Check car collision
            if (self.car_bounds[0] <= current_local_position[0] <= self.car_bounds[1] and
                self.car_bounds[2] <= current_local_position[1] <= self.car_bounds[3] and
                self.car_bounds[4] <= current_local_position[2] <= self.car_bounds[5]):
                print("Collision with car detected!, Joint position: ", current_local_position)
                return True
            else:
                print("No collision with car detected!, Joint position: ", current_local_position)

            # Check obstacle collision
            for obs in self.obstacles:
                obs_position = np.array(obs[0])
                obs_radius = obs[1]
                if np.linalg.norm(position - obs_position) <= (obs_radius+0.5):
                    print("Collision with obstacle detected!")
                    return True

        return False

    def visualize(self, joint_angles, car_pose):
        """
        Visualize the arm joints, car base, and obstacles in 3D.
        :param joint_angles: List of joint angles for the arm.
        :param car_pose: (x, y, yaw) tuple representing the car's global pose.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot ground
        ground = np.array([[self.car_bounds[0], self.car_bounds[2], self.ground_level]])
        ax.plot_surface(
            *np.meshgrid(
                np.linspace(self.car_bounds[0] - 1, self.car_bounds[1] + 1, 10),
                np.linspace(self.car_bounds[2] - 1, self.car_bounds[3] + 1, 10)
            ),
            np.zeros((10, 10)),
            alpha=0.1, color='gray'
        )

        # Plot obstacles
        for obs in self.obstacles:
            obs_position, obs_radius = obs
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = obs_radius * np.cos(u) * np.sin(v) + obs_position[0]
            y = obs_radius * np.sin(u) * np.sin(v) + obs_position[1]
            z = obs_radius * np.cos(v) + obs_position[2]
            ax.plot_surface(x, y, z, color='red', alpha=0.6)

        # Plot car base
        car_x, car_y, _ = car_pose
        car_rect = [
            [self.car_bounds[0], self.car_bounds[1], self.car_bounds[1], self.car_bounds[0], self.car_bounds[0]],
            [self.car_bounds[2], self.car_bounds[2], self.car_bounds[3], self.car_bounds[3], self.car_bounds[2]],
            [self.car_bounds[4], self.car_bounds[4], self.car_bounds[4], self.car_bounds[4], self.car_bounds[4]]
        ]
        car_rect[0] = [p + car_x for p in car_rect[0]]
        car_rect[1] = [p + car_y for p in car_rect[1]]
        ax.plot(car_rect[0], car_rect[1], car_rect[2], color='blue')

        # Plot arm links
        local_positions = [T[:3, 3] for T in forward_all_links(joint_angles)]
        print("End effector position: ", local_positions[-1])
        ee_position = forward(joint_angles)[:3, 3]
        print("End effector position: ", ee_position)
        global_positions = self.apply_car_transform(local_positions, car_pose)
        print("Global end effector position: ", global_positions[-1])
   
        global_positions = np.array(global_positions)
        for i in range(len(global_positions)):
            ax.plot(global_positions[i, 0], global_positions[i, 1], global_positions[i, 2], marker='o', color=f'C{i}')

        # Set plot limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Arm Collision Visualization")
        plt.show()


if __name__ == "__main__":
    # Define car bounds, ground level, and obstacles
    car_bounds = (-0.5, 0.5, -0.5, 0.5, 0, 0.2)
    ground_level = 0
    obstacles = [([0.0, 0.0, 1], 0.5)]

    # Initialize collision checker
    collision_checker = CollisionChecker(car_bounds, ground_level, obstacles)

    # Define joint angles and car pose
    joint_angles = [0, -np.pi/2, 0, 0, 0, 0]
    car_pose = (0, 0, 0)

    # Check for collision
    collision = collision_checker.check_collision(joint_angles, car_pose)
    if not collision:
        print("No collision detected!")
    else:
        print("Collision detected!")

    # Visualize the arm, car, and obstacles
    collision_checker.visualize(joint_angles, car_pose)
