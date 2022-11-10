from abc import ABC, abstractmethod


class AbstractObject(ABC):
    
    def __init__(self, base_position, base_orientation, **kwargs):
        self.base_position = base_position
        self.base_orientation = base_orientation
        super(AbstractObject, self).__init__()

    def load(self, **kwargs):
        item_id = self.load2pybullet(**kwargs)
        self.item_id = item_id
        return item_id

    @abstractmethod
    def load2pybullet(self, **kwargs):
        '''
        load into PyBullet and return the id of robot
        '''        
        raise NotImplementedError        