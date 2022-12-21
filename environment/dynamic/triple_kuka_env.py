import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.multi_robot.triple_kuka_robot import TripleKukaRobot

class TripleKukaEnv(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = TripleKukaRobot()
        else:
            robot = TripleKukaRobot(**robot_config)
        super(TripleKukaEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])


