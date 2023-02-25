import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.simple2arm_robot import Simple2ArmRobot

class Simple2ArmEnv(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = Simple2ArmRobot()
        else:
            robot = Simple2ArmRobot(**robot_config)
        super(Simple2ArmEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])
