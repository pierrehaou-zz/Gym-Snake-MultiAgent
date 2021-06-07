import pygame
import numpy as np

class Arena:
    """
    A class that holds pygame logic related to the game arena
    """

    def __init__(self, dimensions, spacing, num_fruits, agents, agent_colors):

        self.agent_colors = agent_colors
        self.agents = agents

        self.display_surf = None
        self.image_surf = None
        self.fruit_surf = None

        self.fruits = []
        self.num_fruits = num_fruits
        self.num_active_fruits = num_fruits

        self.window_dimension = (dimensions + 2) * spacing
        self.dimensions = dimensions
        self.spacing = spacing

        # initialize global map that keeps track of collision control
        # 0: free, 1: fruit, i>=2: agent i-2
        self.coll_map = np.zeros((dimensions, dimensions)).astype(int)

        self.window_init = False

        # Initialize goals
        for f in range(self.num_active_fruits):
            self.fruits.append(self.generate_goal())


    def pygame_init(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode((self.window_dimension, self.window_dimension),
                                                     pygame.HWSURFACE)
        self.agent_surfs = []
        self.agent_surfs_head = []

        for i, p in enumerate(self.agents):
            image_surf = pygame.Surface([self.spacing - 4, self.spacing - 4])
            image_surf.fill(self.agent_colors[i % len(self.agent_colors)])
            self.agent_surfs.append(image_surf)

            # draw the head darker
            image_surf_head = pygame.Surface([self.spacing - 4, self.spacing - 4])
            image_surf_head.fill(tuple([c // 2 for c in self.agent_colors[i % len(self.agent_colors)]]))
            self.agent_surfs_head.append(image_surf_head)

        self.fruit_surf = pygame.Surface([self.spacing - 4, self.spacing - 4])
        self.fruit_surf.fill((255, 0, 0))

        # wall spacing and color
        self.wall_surf = pygame.Surface([self.spacing, self.spacing])
        self.wall_surf.fill((112, 128, 144))

    def pygame_draw(self, surface, image, pos):
        surface.blit(image, ((pos[0] + 1) * self.spacing, (pos[1] + 1) * self.spacing))

    def draw_env(self):
        """
        Generates background
        """
        self.display_surf.fill((0, 100, 0))

        for i in range(0, self.window_dimension, self.spacing):
            self.display_surf.blit(self.wall_surf, (0, i))
            self.display_surf.blit(self.wall_surf, (self.window_dimension - self.spacing, i))

        for i in range(0, self.window_dimension, self.spacing):
            self.display_surf.blit(self.wall_surf, (i, 0))
            self.display_surf.blit(self.wall_surf, (i, self.window_dimension - self.spacing))

    def generate_goal(self):
        """
        Places fruit at an unoccupied cell in the grid
        """
        # place fruit at unoccupied grid cell
        free_posx, free_posy = np.where(self.coll_map == 0)

        if len(free_posx) == 0:
            print("Warning: No free space for fruit anymore")

            return None

        # pick random index among free grid cells
        rand_ind = np.random.randint(len(free_posx))
        x = free_posx[rand_ind]
        y = free_posy[rand_ind]

        # mark down fruit on collision map
        self.coll_map[x, y] = 1

        return [x, y]

    def remove_agent(self, i):
        """
        Removes agent from collision map. Excludes head
        """

        self.coll_map[self.coll_map == i + 2] = 0

    def sample_agent_position(self, init_length):
        """
        Potential agent positions are sampled, continues until there is free space in the grid
        """

        found_free_pos = False

        while not found_free_pos:

            # first sample a direction
            direction = np.random.randint(0, 4)

            # this determines the relative coordinates of the tail
            # e.g.: direction = 0 (E), tail in W direction, dx = -1, dy = 0
            dx_array = [-1, 1, 0, 0]
            dy_array = [0, 0, 1, -1]

            # sample x,y coordinates such that the tail does not overlap with the wall
            minx = np.maximum(0, -(init_length - 1) * dx_array[direction])
            maxx = self.dimensions + np.minimum(0, -(init_length - 1) * dx_array[direction])
            miny = np.maximum(0, -(init_length - 1) * dy_array[direction])
            maxy = self.dimensions + np.minimum(0, -(init_length - 1) * dy_array[direction])

            # sample agent position
            x = np.random.randint(minx, maxx)
            y = np.random.randint(miny, maxy)

            # now we still have to check whether this agent collides with any other agent
            is_free = True
            for i in range(init_length):
                if self.coll_map[x + i * dx_array[direction], y + i * dy_array[direction]] != 0:
                    is_free = False

            if is_free:
                found_free_pos = True

        return x, y, direction, dx_array[direction], dy_array[direction]

    def close(self):

        pygame.quit()