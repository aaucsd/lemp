import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.multi_robot.dual_simple2arm_robot import DualSimple2ArmRobot

class DualSimple2ArmEnv(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = DualSimple2ArmRobot()
        else:
            robot = DualSimple2ArmRobot(**robot_config)
        super(DualSimple2ArmEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])
