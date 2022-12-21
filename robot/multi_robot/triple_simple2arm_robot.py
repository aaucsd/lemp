from abc import ABC, abstractmethod
import numpy as np
from robot.simple2arm_robot import Simple2ArmRobot
from robot.grouping import RobotGroup
import pybullet as p

class TripleSimple2ArmRobot(RobotGroup):

    def __init__(self, base_positions=((0, 0, 0), (1, 1, 0), (1, 1, 0)),
                       base_orientations=((0, 0, 0.7071, 0.7071), (0, 0, 0, 1), (0, 0, 0, 1)),
                       urdf_file="../data/robot/simple2arm/2dof.urdf",
                       collision_eps=0.5, **kwargs):

        robots = []
        for base_position, base_orientation in zip(base_positions, base_orientations):
            robots.append(Simple2ArmRobot(base_position=base_position, base_orientation=base_orientation, urdf_file=urdf_file, collision_eps=collision_eps))

        super(TripleSimple2ArmRobot, self).__init__(robots=robots, **kwargs)