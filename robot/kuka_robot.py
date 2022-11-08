from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot
import pybullet as p

class KukaRobot(AbstractRobot):
    
    # Initialize env
    def __init__(self, kuka_file="../data/robot/kuka_iiwa/model_0.urdf", collision_eps=0.5):

        self.collision_eps = collision_eps
        self.kuka_file = kuka_file
        
        pid = p.connect(p.DIRECT)
        robot_id = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=pid)
        num_joints = p.getNumJoints(robot_id, physicsClientId=pid)
        limits_low = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[8] for jointId in range(num_joints)]
        limits_high = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[9] for jointId in range(num_joints)]
        p.disconnect(pid)
        
        super().__init__(limits_low, limits_high, list(range(num_joints)), collision_eps)
    
    def load2pybullet(self):
        robot_id = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        return robot_id        