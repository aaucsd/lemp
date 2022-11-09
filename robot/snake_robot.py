from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot
import pybullet as p

class SnakeRobot(AbstractRobot):
    
    # Initialize env
    def __init__(self, urdf_file="../data/robot/snake/snake.urdf", collision_eps=0.1):
        super().__init__(urdf_file, collision_eps)
    
    def _get_joints_and_limits(self, urdf_file):
        limits_low = [-9]*2 + [-np.pi]*5
        limits_high = [9]*2 + [np.pi]*5
        return list(range(len(limits_low))), limits_low, limits_high
    
    def load2pybullet(self, phantom=False, base_position=(0, 0, 0.5), base_orientation=(0, 0, 0, 1)):
        
        sphereRadius = 0.25
        alpha = 0.5 if phantom else 1.
        robot_id = p.loadURDF(self.urdf_file, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.resetBasePositionAndOrientation(robot_id, base_position, base_orientation)

        red = [0.95, 0.1, 0.1, 1]
        green = [0.1, 0.8, 0.1, 1]
        yellow = [1.0, 0.8, 0., 1]
        blue = [0.3, 0.3, 0.8, 1]
        colors = [red, blue, green, yellow]
        for i, data in enumerate(p.getVisualShapeData(robot_id, -1)):
            color = colors[i % 4]
            p.changeVisualShape(robot_id, i - 1, rgbaColor=[color[0], color[1], color[2], alpha])

        if phantom:
            for joint_id in list(range(p.getNumJoints(robot_id))) + [-1]:
                p.setCollisionFilterGroupMask(robot_id, joint_id, 0, 0)

        return robot_id        
    
    def set_config(self, config, robot_id=None, sphereRadius=0.25):
        if robot_id is None:
            robot_id = self.robot_id
        p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])

        quat = transforms3d.euler.euler2quat(0, 0, config[3])
        p.resetBasePositionAndOrientation(robot_id, list(config[:2]) + [0.5], np.concatenate((quat[1:], quat[0].reshape(1))))

        for i in range(len(config[3:])):
            p.resetJointState(robot_id, i * 2 + 1, config[i + 2])
        p.performCollisionDetection()

        if len(p.getContactPoints(robot_id)) == 0:
            return True
        else:
            self.collision_point = config
            return False    