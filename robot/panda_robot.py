from abc import ABC, abstractmethod
import numpy as np
from robot.individual_robot import IndividualRobot
import pybullet as p

class PandaRobot(IndividualRobot):

    def __init__(self, base_position=(0, 0, 1), base_orientation=(0, 0, 0, 1), urdf_file="../data/robot/panda/panda.urdf", collision_eps=0.5, **kwargs):
        super(PandaRobot, self).__init__(base_position=base_position, 
                                        base_orientation=base_orientation, 
                                        urdf_file=urdf_file, 
                                        collision_eps=collision_eps, **kwargs)
        
    def _get_joints_and_limits(self, urdf_file):
        pid = p.connect(p.DIRECT)
        item_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=pid)
        num_joints = p.getNumJoints(item_id, physicsClientId=pid)
        limits_low = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[8] for jointId in range(num_joints)]
        limits_high = [p.getJointInfo(item_id, jointId, physicsClientId=pid)[9] for jointId in range(num_joints)]
        p.disconnect(pid)
        return list(range(num_joints)), limits_low, limits_high
    
    def load2pybullet(self, **kwargs):
        item_id = p.loadURDF(self.urdf_file, self.base_position, self.base_orientation, useFixedBase=True, **kwargs)
        return item_id