from objects.abstract_object import AbstractObject
import pybullet as p
import numpy as np

class VoxelObject(AbstractObject):
    
    def __init__(self, base_position, base_orientation, half_extents, color=None, **kwargs):
        super().__init__(**kwargs)
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.half_extents = half_extents
        if color is None:
            color = np.random.uniform(0, 1, size=3).tolist() + [1]
        self.color = color

    def load2pybullet(self):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)

        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=self.color,
                                          # specularColor=[0.4, .4, 0],
                                          halfExtents=self.half_extents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=self.base_position,
                                     baseOrientation=self.base_orientation)
        return groundId