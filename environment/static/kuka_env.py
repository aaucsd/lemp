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

    def render(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=-176.64,
            cameraPitch=-30.31,
            cameraTargetPosition=[0, 0, 0])  
        return p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]