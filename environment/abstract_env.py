from abc import ABC, abstractmethod
import numpy as np
import pybullet_data
import pybullet as p


class AbstractEnv(ABC):

    def __init__(self, objects, robot):
        '''
        objects is a list of AbstractObject (static objects) and DynamicObject (dynamic object)
        robot is a instance of AbstractRobot
        If there are multiple robots that will be controlled simultaneously, use robot.grouping.RobotGroup to group these robots into one meta-robot first
        '''
        self.objects = objects
        self.robot = robot

    def load(self, **kwargs):
        self.initialize_pybullet(**kwargs)

        self.object_ids = []
        self.robot_id = None
        
        for object_ in self.objects:
            self.object_ids.append(object_.load())
        self.robot_id = self.robot.load()

        self.post_process()        
        p.performCollisionDetection()

    def initialize_pybullet(self, reconnect=True, GUI=False, light_height_z=100):
        
        if reconnect:
            try:
                # close all the previous pybullet connections
                while True:
                    p.resetSimulation()
                    p.disconnect()
            except:
                pass
        
            if GUI:
                p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
            else:
                p.connect(p.DIRECT)
        
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, light_height_z])
        if GUI:        
            self.set_camera_angle()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

    def post_process(self):
        """
        Do nothing for parent class, optionally masking collision among groups of objects
        """           
        pass
    
    def set_camera_angle(self):
        """
        Do nothing for parent class
        """           
        pass    

    def render(self):
        """
        Return a snapshot of the current environment
        """        
        return p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]  