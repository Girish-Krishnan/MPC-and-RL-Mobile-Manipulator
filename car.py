from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import dm_control.utils.transformations as tr
import numpy as np

class CarObservables(composer.Observables):

    @property
    def body(self):
        return self._entity._mjcf_root.find('body', 'buddy')

    def get_sensor_mjcf(self, sensor_name):
        return self._entity._mjcf_root.find('sensor', sensor_name)

    @composer.observable
    def realsense_camera(self):
        return observable.MJCFCamera(self._entity._mjcf_root.find('camera', 'buddy_realsense_d435i'), height=64, width=128, buffer_size=1, depth=True)
    
    # @composer.observable
    # def compute_point_cloud(self):
    #     def get_point_cloud(physics):
    #         # Get the depth image from the camera
    #         depth_image = self.realsense_camera(physics)
    #         if depth_image is None:
    #             raise ValueError("Depth image is None. Check camera configuration.")
            
    #         # Process depth image into point cloud
    #         height, width, _ = depth_image.shape
            
    #         # Camera intrinsics for D455
    #         fx = 382.57 # these focal lengths were found from online source describing the D455 camera
    #         fy = 382.57
    #         cx = width / 2
    #         cy = height / 2

    #         # Generate the 2D grid of (x, y) pixel coordinates
    #         xs, ys = np.meshgrid(np.arange(width), np.arange(height))  # Shape: (H, W)

    #         # Normalize pixel coordinates and map to 3D coordinates
    #         xs = (xs - cx) / fx
    #         ys = (ys - cy) / fy
    #         zs = depth_image[:, :, 0]  # Extract depth values (H, W)

    #         # Mask out invalid depth points (e.g., zero or negative depth)
    #         valid_mask = zs > 0
    #         xs, ys, zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]

    #         # Stack valid points into an N x 3 array
    #         point_cloud = np.stack((xs * zs, ys * zs, zs), axis=-1)  # Shape: (N, 3)

    #         return point_cloud

    #     return observable.Generic(get_point_cloud)
    
    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity._mjcf_root.find('body', 'buddy'))

    @composer.observable
    def wheel_speeds(self):
        def get_wheel_speeds(physics):
            return np.concatenate([
                physics.bind(self.get_sensor_mjcf(f'buddy_wheel_{wheel}_vel')).sensordata
                for wheel in ['fl', 'fr', 'bl', 'br']
            ])

        return observable.Generic(get_wheel_speeds)

    @composer.observable
    def body_pose_2d(self):
        def get_pose_2d(physics):
            pos = physics.bind(self.body).xpos[:2]
            yaw = tr.quat_to_euler(physics.bind(self.body).xquat)[2]
            return np.append(pos, yaw)

        return observable.Generic(get_pose_2d)

    @composer.observable
    def body_vel_2d(self):
        def get_vel_2d(physics):
            quat = physics.bind(self.body).xquat
            velocity_local = physics.bind(self.get_sensor_mjcf('velocimeter')).sensordata
            return tr.quat_rotate(quat, velocity_local)[:2]

        return observable.Generic(get_vel_2d)

    # @composer.observable
    # def body_rotation(self):
    #     return observable.MJCFFeature('xquat', self.body)

    # @composer.observable
    # def body_rotation_matrix(self):
    #     def get_rotation_matrix(physics):
    #         quat = physics.bind(self.body).xquat
    #         return tr.quat_to_mat(quat).flatten()
    #     return observable.Generic(get_rotation_matrix)


    # @composer.observable
    # def sensors_vel(self):
    #     return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('velocimeter'))

    @composer.observable
    def steering_pos(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_steering_pos'))

    # @composer.observable
    # def steering_vel(self):
    #     return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_steering_vel'))

    # @composer.observable
    # def sensors_acc(self):
    #     return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_accelerometer'))

    # @composer.observable
    # def sensors_gyro(self):
    #     return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_gyro'))

    # def _collect_from_attachments(self, attribute_name):
    #     out = []
    #     for entity in self._entity.iter_entities(exclude_self=True):
    #         out.extend(getattr(entity.observables, attribute_name, []))
    #     return out

    # @property
    # def kinematic_sensors(self):
    #     return ([self.sensors_vel] + [self.sensors_gyro] +
    #             self._collect_from_attachments('kinematic_sensors'))
    
    ### For robot with arm:

    @composer.observable
    def shoulder_pan_joint_pos(self):
        """Observable for the position of the shoulder_pan_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('shoulder_pan'))

    @composer.observable
    def shoulder_lift_joint_pos(self):
        """Observable for the position of the shoulder_lift_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('shoulder_lift'))

    @composer.observable
    def elbow_joint_pos(self):
        """Observable for the position of the elbow_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('elbow'))

    @composer.observable
    def wrist_1_joint_pos(self):
        """Observable for the position of the wrist_1_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('wrist_1'))

    @composer.observable
    def wrist_2_joint_pos(self):
        """Observable for the position of the wrist_2_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('wrist_2'))

    @composer.observable
    def wrist_3_joint_pos(self):
        """Observable for the position of the wrist_3_joint."""
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('wrist_3'))

    @property
    def ur5e_joint_positions(self):
        """Collect all UR5e joint position observables into a list."""
        return [
            self.shoulder_pan_joint_pos,
            self.shoulder_lift_joint_pos,
            self.elbow_joint_pos,
            self.wrist_1_joint_pos,
            self.wrist_2_joint_pos,
            self.wrist_3_joint_pos
        ]

    @property
    def all_observables(self):
        return [
            # self.body_position,
            # self.body_rotation,
            # self.body_rotation_matrix,
            self.body_pose_2d,
            # self.body_vel_2d,
            # self.wheel_speeds,
            self.realsense_camera,
            # self.compute_point_cloud,
            # self.steering_pos,
            # self.steering_vel,
        ] + self.ur5e_joint_positions # + self.kinematic_sensors
class Car(composer.Robot):
    def _build(self, name='car'):
        model_path = "cars/car_with_arm/buddy.xml"
        self._mjcf_root = mjcf.from_path(f'{model_path}')
        if name:
            self._mjcf_root.model = name

        self._actuators = self.mjcf_model.find_all('actuator')

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return self._actuators

    def apply_action(self, physics, action, random_state=None):
        """Apply action to car's actuators. 
        `action` is expected to be a numpy array with [steering, throttle] + joint positions if the car has an arm.
        """
        steering, throttle = action[0], action[1]
        physics.bind(self.mjcf_model.find('actuator', 'buddy_steering_pos')).ctrl = steering
        physics.bind(self.mjcf_model.find('actuator', 'buddy_throttle_velocity')).ctrl = throttle
        if len(action) > 2:
            shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3 = action[2:]
            physics.bind(self.mjcf_model.find('actuator', 'shoulder_pan')).ctrl = shoulder_pan
            physics.bind(self.mjcf_model.find('actuator', 'shoulder_lift')).ctrl = shoulder_lift
            physics.bind(self.mjcf_model.find('actuator', 'elbow')).ctrl = elbow
            physics.bind(self.mjcf_model.find('actuator', 'wrist_1')).ctrl = wrist_1
            physics.bind(self.mjcf_model.find('actuator', 'wrist_2')).ctrl = wrist_2
            physics.bind(self.mjcf_model.find('actuator', 'wrist_3')).ctrl = wrist_3

    def _build_observables(self):
        return CarObservables(self)
      
class Button(composer.Entity):
    """A button Entity which changes colour when pressed with certain force."""
    def _build(self):
        self._mjcf_model = mjcf.RootElement()
        self._geom = self._mjcf_model.worldbody.add(
            'geom', type='sphere', size=[0.1], rgba=[1, 1, 0, 1])
        self._site = self._mjcf_model.worldbody.add(
        'site', type='cylinder', size=self._geom.size*1.01, rgba=[1, 1, 1, 0])

    @property
    def mjcf_model(self):
        return self._mjcf_model
