import numpy as np


class Agent:
    """
    A class to represent a snake agent
    """

    def __init__(self, x, y, spacing=22, length=3, direction=0, max_buffer_length=2000):

        self.init_length = length
        self.length = length
        self.max_buffer_length = max_buffer_length
        self.spacing = spacing
        self.x = x
        self.y = y
        self.direction = direction

    def reset(self, x, y, direction):
        self.length = self.init_length

        self.direction = direction

        self.x = -1 + np.zeros((self.max_buffer_length,)).astype(int)
        self.y = -1 + np.zeros((self.max_buffer_length,)).astype(int)

        # initial positions, no collision.
        if self.direction == 0:
            self.x[:self.length] = x - np.arange(self.length).astype(int)
            self.y[:self.length] = y

        if self.direction == 1:
            self.x[:self.length] = x + np.arange(self.length).astype(int)
            self.y[:self.length] = y

        if self.direction == 2:
            self.y[:self.length] = y + np.arange(self.length).astype(int)
            self.x[:self.length] = x

        if self.direction == 3:
            self.y[:self.length] = y - np.arange(self.length).astype(int)
            self.x[:self.length] = x

    def act(self, action):
        """
        Directional Control for Snake
        """
        if action == 0:
            if self.direction == 1:
                return

            self.direction = 0

        elif action == 1:
            if self.direction == 0:
                return

            self.direction = 1

        elif action == 2:
            if self.direction == 3:
                return

            self.direction = 2

        elif action == 3:
            if self.direction == 2:
                return

            self.direction = 3

        else:
            # continue direction
            pass

    def remove_tail(self):
        # set last array element to -1
        self.x[self.length] = -1
        self.y[self.length] = -1

    def update(self):
        """
        Updates position of snake
        """
        # update previous positions
        for i in range(self.length, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # update position of head of snake
        if self.direction == 0:
            self.x[0] += 1
        if self.direction == 1:
            self.x[0] -= 1
        if self.direction == 2:
            self.y[0] -= 1
        if self.direction == 3:
            self.y[0] += 1

    def draw(self, surface, image, image_head):
        """
        Renders snake in the game
        """

        for i in range(0, self.length):
            if i == 0:
                # draw head
                surface.blit(image_head, ((self.x[i] + 1) * self.spacing, (self.y[i] + 1) * self.spacing))
            else:
                # draw tail
                surface.blit(image, ((self.x[i] + 1) * self.spacing, (self.y[i] + 1) * self.spacing))
