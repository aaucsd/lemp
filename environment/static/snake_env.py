import pybullet as p
from environment.abstract_env import AbstractEnv
from robot.snake_robot import SnakeRobot


class SnakeEnv(AbstractEnv):
    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = SnakeRobot()
        else:
            robot = SnakeRobot(**robot_config)
        super(SnakeEnv, self).__init__(objects, robot)

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=0,
            cameraPitch=-89.8,
            cameraTargetPosition=[0, 0, 10])
    
    def post_process(self):
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.plane_id = p.createMultiBody(0, plane)