    def plot(self, path, make_gif=False):
        path = np.array(path)

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        for halfExtents, basePosition in self.obstacles:
            self.create_voxel(halfExtents, basePosition)

        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[0])

        target_kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[-1], target_kukaId)

        prev_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        final_pos = p.getLinkState(target_kukaId, self.kukaEndEffectorIndex)[0]

        p.setGravity(0, 0, -10)
        p.stepSimulation()

        gifs = []
        current_state_idx = 0

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])

            new_kuka = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                                  flags=p.URDF_IGNORE_COLLISION_SHAPES)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)

            K = int(np.ceil(d / 0.2))
            for k in range(0, K):

                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, new_kuka)
                new_pos = p.getLinkState(new_kuka, self.kukaEndEffectorIndex)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 10, 0)
                prev_pos = new_pos
                p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                if make_gif:
                    gifs.append(self.render())

            current_state_idx += 1
            if current_state_idx == len(path) - 1:
                self.set_config(path[-1], new_kuka)
                p.addUserDebugLine(prev_pos, final_pos, [1, 0, 0], 10, 0)
                p.loadURDF("sphere2red.urdf", final_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                break

        return gifs
    
    
    def plot(self, path, make_gif=False):
        path = np.array(path)
        self.reset_env()
        for halfExtents, basePosition in self.obstacles:
            self.create_voxel(halfExtents, basePosition, [0, 0, 0, 1])

        new_ur5 = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                             flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[-1], new_ur5)
        final_pos = p.getLinkState(new_ur5, self.tip_index)[0]

        self.set_config(path[0], self.ur5)
        if make_gif:
            for _ in range(100):
                p.stepSimulation()
                sleep(0.1)

        prev_pos = p.getLinkState(self.ur5, self.tip_index)[0]

        # [p.getClosestPoints(self.ur5, obs, distance=0.09) for obs in self.obs_ids]

        gifs = []
        current_state_idx = 0

        new_ur5 = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                             flags=p.URDF_IGNORE_COLLISION_SHAPES)
        for data in p.getVisualShapeData(new_ur5):
            color = list(data[-1])
            color[-1] = 0.5
            p.changeVisualShape(new_ur5, data[1], rgbaColor=color)

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])

            K = int(np.ceil(d / 0.5))
            for k in range(0, K):

                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, new_ur5)
                # p.performCollisionDetection()
                # p.stepSimulation()
                new_pos = p.getLinkState(new_ur5, self.tip_index)[0]
                p.addUserDebugLine(prev_pos, new_pos, [1, 0, 0], 10, 0)
                prev_pos = new_pos
                b = p.loadURDF("sphere2red.urdf", new_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                # if k==0:
                #     p.changeVisualShape(b, -1, rgbaColor=[0, 0, 0.7, 0.7])
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

            # if np.linalg.norm(
            #         np.array([p.getJointStates(self.ur5, self.joints)[i][0] for i in range(len(self.joints))]) -
            #         np.array(path[current_state_idx + 1])) < 0.5:
            current_state_idx += 1

            # gifs.append(p.getCameraImage(width=1000, height=800, lightDirection=[1, 1, 1], shadow=1,
            #                              renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            if current_state_idx == len(path) - 1:
                self.set_config(path[-1], new_ur5)
                p.addUserDebugLine(prev_pos, final_pos, [1, 0, 0], 10, 0)
                p.loadURDF("sphere2red.urdf", final_pos, globalScaling=0.05, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                break

        return gifs    