import pybullet as p
from environment.abstract_env import AbstractEnv
from robot.extend_kuka_robot import ExtendKukaRobot


class ExtendKukaEnv(AbstractEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = ExtendKukaRobot()
        else:
            robot = ExtendKukaRobot(**robot_config)
        super(ExtendKukaEnv, self).__init__(objects, robot)
        
    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=-176.64,
            cameraPitch=-30.31,
            cameraTargetPosition=[0, 0, 0])        