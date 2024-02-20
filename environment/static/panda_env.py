import pybullet as p
from environment.static_env import StaticEnv
from robot.panda_robot import PandaRobot
import numpy as np


class PandaEnv(StaticEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = PandaRobot()
        else:
            robot = PandaRobot(**robot_config)
        super(PandaEnv, self).__init__(objects, robot)
        self.robot.limits_high[7:] = np.array([0.,0.,0.,0.])
        self.robot.limits_low[7:] = np.array([0.,0.,0.,0.])

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=-176.64,
            cameraPitch=-30.31,
            cameraTargetPosition=[0, 0, 0])