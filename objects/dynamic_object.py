from abc import ABC, abstractmethod
from objects.abstract_object import AbstractObject
from objects.trajectory import AbstractTrajectory
import pybullet as p


class MovableObject(AbstractObject, ABC):
    
    @abstractmethod
    def set_config(self, config):
        raise NotImplementedError
        

class MovableBaseObject(MovableObject):

    def __init__(self, move_mode, **kwargs):
        super(MovableBaseObject, self).__init__(**kwargs)
        assert (move_mode=='p') or (move_mode=='o') or (move_mode=='po') or (move_mode=='op')
        self.move_mode = move_mode
    
    def set_config(self, config):
        position, orientation = p.getBasePositionAndOrientation(self.item_id)
        if self.move_mode == 'p':
            assert len(config)==3
            p.resetBasePositionAndOrientation(self.item_id, config, orientation)
        elif self.move_mode == 'o':
            assert len(config)==4
            p.resetBasePositionAndOrientation(self.item_id, position, config)
        else:
            assert len(config)==7
            p.resetBasePositionAndOrientation(self.item_id, config[:3], config[3:])


class DynamicObject(AbstractObject):

    def __init__(self, item: MovableObject, trajectory: AbstractTrajectory, **kwargs):
        self.item = item
        self.trajectory = trajectory
        super(DynamicObject, self).__init__(**kwargs)

    def set_config_at_time(self, t):
        spec = self.trajectory.get_spec(t)
        self.trajectory.set_spec(self.item, spec) 
        

class MovableObjectFactory:      
    @staticmethod
    def create_movable_object_class(ObjectX, MovableXObject):
        '''
        Argument: turning a object class into a movable object class
        '''
        assert not issubclass(ObjectX, MovableObject)
        assert issubclass(MovableXObject, MovableObject)
        class MovableSpecificObject(ObjectX, MovableXObject):
            '''just a pure inheritance'''
        return MovableSpecificObject


class DynamicObjectFactory:

    @staticmethod
    def create_dynamic_object_class(MovableXObject):
        '''
        Argument: turning a movable object class into a dynamic object class
        '''
        assert issubclass(MovableXObject, MovableObject)
        class DynamicSpecificObject(MovableXObject, DynamicObject):
            def __init__(self, trajectory: AbstractTrajectory, **kwargs):
                super(DynamicSpecificObject, self).__init__(item=self, trajectory=trajectory, **kwargs)
        return DynamicSpecificObject