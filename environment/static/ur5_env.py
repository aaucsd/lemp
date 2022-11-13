import pybullet as p
from environment.static_env import StaticEnv
from robot.ur5_robot import UR5Robot


class UR5Env(StaticEnv):

    def __init__(self, objects, robot_config=None):
        if robot_config is None:
            robot = UR5Robot()
        else:
            robot = UR5Robot(**robot_config)
        super(UR5Env, self).__init__(objects, robot)        

    def set_camera_angle(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=12.040756225585938,
            cameraPitch=-37.56093978881836,
            cameraTargetPosition=[0, 0, 0.7])         
        
    def post_process(self):
        plane = p.createCollisionShape(p.GEOM_PLANE)
        self.plane_id = p.createMultiBody(0, plane)
        p.setCollisionFilterPair(self.robot_id, self.plane_id, 1, -1, 0)