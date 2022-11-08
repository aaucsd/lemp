from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot
import pybullet as p

class UR5Robot(AbstractRobot):
    
    # Initialize env
    def __init__(self, ur5_file="../data/robot/ur5/ur5.urdf", collision_eps=0.1):

        self.collision_eps = collision_eps
        self.ur5_file = ur5_file
        
        pid = p.connect(p.DIRECT)
        robot_id = p.loadURDF(ur5_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        num_joints = p.getNumJoints(robot_id, physicsClientId=pid)
        joints = [p.getJointInfo(robot_id, i, physicsClientId=pid) for i in range(num_joints)]
        joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]     
        limits_low = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[8] for jointId in range(num_joints)]
        limits_high = [p.getJointInfo(robot_id, jointId, physicsClientId=pid)[9] for jointId in range(num_joints)]
        p.disconnect(pid)
        
        super().__init__(limits_low, limits_high, joints, collision_eps)
    
    def load2pybullet(self):
        robot_id = p.loadURDF(self.ur5_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        return robot_id