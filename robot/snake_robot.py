from abc import ABC, abstractmethod
import numpy as np
from robot.individual_robot import IndividualRobot
import pybullet as p
import transforms3d

class SnakeRobot(IndividualRobot):
    
    def __init__(self, base_position=(0, 0, 0.5), base_orientation=(0, 0, 0, 1), 
                       urdf_file="../data/robot/snake/snake.urdf", collision_eps=0.1, 
                       phantom=False, **kwargs):
        self.phantom = phantom
        super(SnakeRobot, self).__init__(base_position=base_position, 
                                         base_orientation=base_orientation, 
                                         urdf_file=urdf_file, 
                                         collision_eps=collision_eps, **kwargs)
    
    def _get_joints_and_limits(self, urdf_file):
        limits_low = [-9]*2 + [-np.pi]*5
        limits_high = [9]*2 + [np.pi]*5
        return list(range(len(limits_low))), limits_low, limits_high
    
    def load2pybullet(self, **kwargs):
        
        alpha = 0.5 if self.phantom else 1.
        item_id = p.loadURDF(self.urdf_file, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.resetBasePositionAndOrientation(item_id, self.base_position, self.base_orientation)

        red = [0.95, 0.1, 0.1, 1]
        green = [0.1, 0.8, 0.1, 1]
        yellow = [1.0, 0.8, 0., 1]
        blue = [0.3, 0.3, 0.8, 1]
        colors = [red, blue, green, yellow]
        for i, data in enumerate(p.getVisualShapeData(item_id, -1)):
            color = colors[i % 4]
            p.changeVisualShape(item_id, i - 1, rgbaColor=[color[0], color[1], color[2], alpha])

        if self.phantom:
            for joint_id in list(range(p.getNumJoints(item_id))) + [-1]:
                p.setCollisionFilterGroupMask(item_id, joint_id, 0, 0)

        return item_id        
    
    def set_config(self, config, item_id=None):
        if item_id is None:
            item_id = self.item_id
        p.resetBaseVelocity(item_id, [0, 0, 0], [0, 0, 0])

        quat = transforms3d.euler.euler2quat(0, 0, config[3])
        p.resetBasePositionAndOrientation(item_id, list(config[:2]) + [0.5], np.concatenate((quat[1:], quat[0].reshape(1))))

        for i in range(len(config[3:])):
            p.resetJointState(item_id, i * 2 + 1, config[i + 2])
        p.performCollisionDetection()

        if len(p.getContactPoints(item_id)) == 0:
            return True
        else:
            self.collision_point = config
            return False    