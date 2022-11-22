from abc import ABC, abstractmethod
import numpy as np
from robot.kuka4_robot import Kuka4Robot
from robot.grouping import RobotGroup
import pybullet as p

class DualKuka4Robot(RobotGroup):

    def __init__(self, base_positions=((0, 0, 0), (0.5, 0, 0)),
                       base_orientations=((0, 0, np.sin(np.pi), np.cos(np.pi)), (0, 0, 0, 1)),
                       urdf_file="../data/robot/kuka_iiwa/model_4dof.urdf",
                       collision_eps=0.5, **kwargs):

        robots = []
        for base_position, base_orientation in zip(base_positions, base_orientations):
            robots.append(Kuka4Robot(base_position=base_position, base_orientation=base_orientation, urdf_file=urdf_file, collision_eps=collision_eps))

        super(DualKuka4Robot, self).__init__(robots=robots, **kwargs)