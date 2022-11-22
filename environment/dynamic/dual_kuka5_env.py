import pybullet as p
from environment.dynamic_env import DynamicEnv
from robot.multi_robot.dual_kuka5_robot import DualKuka5Robot

class DualKuka5Env(DynamicEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = DualKuka5Robot()
        else:
            robot = DualKuka5Robot(**robot_config)
        super(DualKuka5Env, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.,
            cameraYaw=25,
            cameraPitch=-20,
            cameraTargetPosition=[0.5, 0.5, 0.5])
