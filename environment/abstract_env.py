from abc import ABC, abstractmethod
import numpy as np
import pybullet_data


class AbstractEnv(ABC):

    def __init__(self, objects, robot):
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

    def initialize_pybullet(self, GUI=False, light_height_z=100):
        try:
            p.resetSimulation()
            p.disconnect()
        except:
            pass
        
        if GUI:
            p.connect(p.GUI, options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition = [0, 0, light_height_z])
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

    def post_process(self):
        """
        Do nothing for parent class
        """           
        pass

    @abstractmethod
    def render(self):
        """
        Return a snapshot of the current environment
        """        
        raise NotImplementedError