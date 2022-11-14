from abc import ABC, abstractmethod
import numpy as np
from robot.individual_robot import IndividualRobot
import pybullet as p

class UR5Robot(IndividualRobot):
    
    def __init__(self, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1), urdf_file="../data/robot/ur5/ur5.urdf", collision_eps=0.1, **kwargs):
        super(UR5Robot, self).__init__(base_position=base_position, 
                                       base_orientation=base_orientation, 
                                       urdf_file=urdf_file, 
                                       collision_eps=collision_eps, **kwargs)
        
    def _get_joints_and_limits(self, urdf_file):
        pid = p.connect(p.DIRECT)
        item_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=pid)
        num_joints = p.getNumJoints(item_id, physicsClientId=pid)
        joints = [p.getJointInfo(item_id, i, physicsClientId=pid) for i in range(num_joints)]
        joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]     
        limits_low = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[8] for jointId in joints]
        limits_high = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[9] for jointId in joints]
        p.disconnect(pid)
        return joints, limits_low, limits_high
    
    def load2pybullet(self, **kwargs):
        item_id = p.loadURDF(self.urdf_file, self.base_position, self.base_orientation, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        return item_id