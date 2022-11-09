from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot
import pybullet as p

class KukaRobot(AbstractRobot):
    
    # Initialize env
    def __init__(self, urdf_file="../data/robot/kuka_iiwa/model_0.urdf", collision_eps=0.5):
        super().__init__(urdf_file, collision_eps)
        
    def _get_joints_and_limits(self, urdf_file):
        pid = p.connect(p.DIRECT)
        robot_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=pid)
        num_joints = p.getNumJoints(robot_id, physicsClientId=pid)
        limits_low = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[8] for jointId in range(num_joints)]
        limits_high = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[9] for jointId in range(num_joints)]
        p.disconnect(pid)
        return list(range(num_joints)), limits_low, limits_high
    
    def load2pybullet(self, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1)):
        robot_id = p.loadURDF(self.urdf_file, base_position, base_orientation, useFixedBase=True)
        return robot_id        