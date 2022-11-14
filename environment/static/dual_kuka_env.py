import pybullet as p
from environment.static_env import StaticEnv
from robot.multi_robot.dual_kuka_robot import DualKukaRobot

class DualKukaEnv(StaticEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = DualKukaRobot()
        else:
            robot = DualKukaRobot(**robot_config)
        super(DualKukaEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.57699,
            cameraYaw=203.809,
            cameraPitch=-30.335,
            cameraTargetPosition=[0, 0, 0.7])
