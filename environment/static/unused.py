    def set_random_init_goal(self):
        while True:
            points = self.sample_n_points(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal
        
    def init_new_problem(self, index=None):
        '''
        Initialize a new planning problem
        '''
        if index is None:
            self.index = self.episode_i
        else:
            self.index = index

        obstacles, start, goal, path = self.problems[index]

        self.episode_i += 1
        self.episode_i = (self.episode_i) % len(self.order)
        self.collision_check_count = 0
        self.collision_time = 0

        self.collision_point = None

        self.obstacles = obstacles
        self.init_state = start
        self.goal_state = goal
        self.path = path

        for halfExtents, basePosition in obstacles:
            self.create_voxel(halfExtents, basePosition)

        return self.get_problem()
    
    
    def plot(self, map, path, make_gif=False):
        # self.reset(map)
        path = np.array(path)
        self.set_config(path[0])

        # p.setGravity(0, 0, -10)
        # p.stepSimulation()

        # if make_gif:
        #     for _ in range(100):
        #         p.stepSimulation()
        #         sleep(0.1)

        gifs = []
        current_state_idx = 0

        new_snake = self.create_snake(phantom=False)
        self.set_config(path[-1], snakeId=new_snake)

        while True:
            current_state = path[current_state_idx]
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])
            K = int(d / 0.5)
            for k in range(0, K):
                new_snake = self.create_snake(phantom=True)
                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, snakeId=new_snake)
                # p.stepSimulation()
                if make_gif:
                    gifs.append(p.getCameraImage(width=1100, height=900, lightDirection=[1, 1, 1], shadow=1,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
            current_state_idx += 1
            if current_state_idx == (len(path)-1):
                break

        return gifs
    
    
    def create_maze(self, map):
        self.mazeIds = []
        numHeightfieldRows = 30
        numHeightfieldColumns = 30
        for j in range(len(map)):
            for i in range(len(map[0])):
                if map[i, j]:
                    self.mazeIds.append(self.create_voxel([0.7, 0.7, 1], [1.4*i-10.5, 1.4*j-10.5, 0]))
        return self.mazeIds    