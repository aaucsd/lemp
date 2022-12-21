import numpy as np
from robot.abstract_robot import AbstractRobot
from robot.individual_robot import IndividualRobot
import pybullet as p


class RobotGroup(AbstractRobot):
    
    '''
    Grouping multiple robots together into a meta-robot
    '''

    def __init__(self, robots, grouping_mask_fn=None, collision_eps=None, **kwargs):
        '''
        grouping mask function aims to assign the collision mask to each robot
        the argument to the function is an instance of the robot group
        it will be called when loading the robots into PyBullet
        '''
        assert np.all([isinstance(robot, IndividualRobot) for robot in robots])
        self.robots = robots
        self.grouping_mask_fn = grouping_mask_fn
        if collision_eps is None:
            collision_eps = min([robot.collision_eps for robot in self.robots])
        limits_low, limits_high = self._get_limits()
        super(RobotGroup, self).__init__(limits_low=limits_low, 
                                         limits_high=limits_high, 
                                         collision_eps=collision_eps, **kwargs)

    def _get_limits(self):
        all_limits_low = []
        all_limits_high = []
        for robot in self.robots:
            all_limits_low.extend(robot.limits_low)
            all_limits_high.extend(robot.limits_high)
        return all_limits_low, all_limits_high

    # =====================pybullet module=======================

    def load2pybullet(self, **kwargs):
        item_ids = [robot.load(**kwargs) for robot in self.robots]
        if self.grouping_mask_fn:
            self.grouping_mask_fn(self)
        self.collision_check_count = 0
        return item_ids

    def set_config(self, config, item_ids=None):
        if item_ids is None:
            item_ids = self.item_ids

        ptr = 0
        for item_id, robot in zip(item_ids, self.robots):
            robot.set_config(config[ptr:(ptr+robot.config_dim)])
            ptr += robot.config_dim
        p.performCollisionDetection()
    
    # =====================internal collision check module=======================
    
    def no_collision(self):
        p.performCollisionDetection()
        if np.all([len(p.getContactPoints(item_id)) == 0 for item_id in self.item_ids]):
            self.collision_check_count += 1
            return True
        else:
            self.collision_check_count += 1
            return False