from abc import ABC, abstractmethod
import numpy as np
from robot.kuka_robot import KukaRobot
import pybullet as p

class ExtendKukaRobot(KukaRobot): # The kuka with 13 DoF
    
    def __init__(self, urdf_file="../data/robot/kuka_iiwa/model_3.urdf", collision_eps=0.5, **kwargs):
        super(ExtendKukaRobot, self).__init__(urdf_file=urdf_file, collision_eps=collision_eps, **kwargs)   