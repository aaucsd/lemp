import pybullet as p
from environment.abstract_env import AbstractEnv
from robot.kuka_robot import KukaRobot


class KukaEnv(AbstractEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = KukaRobot()
        else:
            robot = KukaRobot(**robot_config)
        super(KukaEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=-176.64,
            cameraPitch=-30.31,
            cameraTargetPosition=[0, 0, 0])