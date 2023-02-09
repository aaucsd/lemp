import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.multi_robot.triple_simple2arm_robot import TripleSimple2ArmRobot

class TripleSimple2ArmEnv(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = TripleSimple2ArmRobot()
        else:
            robot = TripleSimple2ArmRobot(**robot_config)
        super(TripleSimple2ArmEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])


