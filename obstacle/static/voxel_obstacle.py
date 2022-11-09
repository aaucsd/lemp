from obstacle.abstract_obstacle import AbstractObstacle
import pybullet as p
import numpy as np

class VoxelObstacle(AbstractObstacle):
    
    def load2pybullet(self, base_position, base_orientation, half_extents, color=None):
        groundColId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        
        if color is None:
            color = np.random.uniform(0, 1, size=3).tolist() + [1]

        groundVisID = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          rgbaColor=color,
                                          # specularColor=[0.4, .4, 0],
                                          halfExtents=half_extents)
        groundId = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=groundColId,
                                     baseVisualShapeIndex=groundVisID,
                                     basePosition=base_position,
                                     baseOrientation=base_orientation)
        return groundId