from abc import ABC, abstractmethod
import numpy as np
from robot.abstract_robot import AbstractRobot
import pybullet as p


class IndividualRobot(AbstractRobot, ABC):
    # An individual robot
    def __init__(self, base_position, base_orientation, urdf_file, **kwargs):
        # for loading the pybullet
        self.base_position = base_position
        self.base_orientation = base_orientation
        
        self.urdf_file = urdf_file

        joints, limits_low, limits_high = self._get_joints_and_limits(self.urdf_file)
        self.joints = joints
        
        kwargs['base_position'] = base_position
        kwargs['base_orientation'] = base_orientation
        super(IndividualRobot, self).__init__(limits_low=limits_low, 
                                              limits_high=limits_high, **kwargs)

    @abstractmethod
    def _get_joints_and_limits(self, urdf_file):
        raise NotImplementedError
        
    # =====================pybullet module=======================        
        
    def load(self, **kwargs):
        item_id = self.load2pybullet(**kwargs)
        self.collision_check_count = 0
        self.item_id = item_id
        return item_id  
    
    @abstractmethod
    def load2pybullet(self, **kwargs):
        '''
        load into PyBullet and return the id of robot
        '''        
        raise NotImplementedError
        
    def set_config(self, config, item_id=None):
        '''
        set a configuration
        '''            
        if item_id is None:
            item_id = self.item_id
        for i, c in zip(self.joints, config):
            p.resetJointState(item_id, i, c)
        p.performCollisionDetection()  
        
    # =====================internal collision check module=======================

    def no_collision(self):
        '''
        Perform the collision detection
        '''
        p.performCollisionDetection()
        if len(p.getContactPoints(self.item_id)) == 0:
            self.collision_check_count += 1
            return True
        else:
            self.collision_check_count += 1
            return False

    def get_workspace_observation(self):
        '''
        Get the workspace observation
        '''
        raise NotImplementedError