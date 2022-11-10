import pybullet as p
from environment.abstract_env import AbstractEnv
from robot.snake_robot import SnakeRobot


class SnakeEnv:
    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = SnakeEnv()
        else:
            robot = SnakeEnv(**robot_config)
        super(SnakeEnv, self).__init__(objects, robot)

    def render(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=0,
            cameraPitch=-89.8,
            cameraTargetPosition=[0, 0, 10])  
        return p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]            
    
    def post_process(self):
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.plane_id = p.createMultiBody(0, plane)