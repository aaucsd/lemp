import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.multi_robot.dual_kuka4_robot import DualKuka4Robot

class DualKuka4Env(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = DualKuka4Robot()
        else:
            robot = DualKuka4Robot(**robot_config)
        super(DualKuka4Env, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])
