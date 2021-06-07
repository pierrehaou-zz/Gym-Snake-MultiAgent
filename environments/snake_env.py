
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pygame
import time
from copy import deepcopy
from collections import deque
from core.agent_core import Agent
from core.pygame_core import Arena

# CONSTANTS FOR OBSERVATION SPACE
DEAD_OBS = -1
FREE_OBS = 0
FRUIT_OBS = 1
SELF_AGENT_OBS = 2
SELF_AGENT_HEAD_OBS = 3
OTHER_AGENT_OBS = 4
OTHER_AGENT_HEAD_OBS = 5

# CONSTANTS FOR AGENT COLORS
AGENT_1_COLOR = (255, 0, 255)
AGENT_2_COLOR = (0, 0, 255)
AGENT_3_COLOR = (255, 255, 0)
AGENT_4_COLOR = (76, 0, 153)

class SnakeEnvironment(gym.Env):
    AGENT_COLORS = [
        AGENT_1_COLOR,
        AGENT_2_COLOR,
        AGENT_3_COLOR,
        AGENT_4_COLOR,
    ]

    def __init__(self, num_agents=1, num_fruits=3, dimensions=4, spacing=22, init_length=3, reward_fruit=1.0,
                 reward_killed=-1.0, reward_finished=0.0, flatten_states=True,
                 history=1):

        self.agents = []
        self.fruits = []
        self.num_agents = num_agents
        self.active_agents = num_agents
        self.num_fruits = num_fruits
        self.num_active_fruits = num_fruits
        self.init_length = init_length
        self.reward_fruit = reward_fruit
        self.reward_killed = reward_killed
        self.reward_finished = reward_finished
        self.flatten_states = flatten_states
        self.history = history

        self.arena = Arena(dimensions, spacing, num_fruits, self.agents, self.AGENT_COLORS)

        # creates agent(s)
        for i in range(self.num_agents):
            agent = self._create_agent(i, self.init_length, create_object=True)
            self.agents.append(agent)

        self.killed = [False] * self.num_agents

        # return observation and action spaces from the viewpoint of a single agent
        if self.flatten_states:
            self.observation_space = spaces.Box(low=-1, high=5, shape=(self.arena.dimensions ** 2 * self.history,))
        else:
            self.observation_space = spaces.Box(low=-1, high=5,
                                                shape=(self.arena.dimensions, self.arena.dimensions, self.history))

        self.action_space = spaces.Discrete(4)

        min_reward = np.min([reward_fruit, reward_killed, reward_finished])
        max_reward = np.max([reward_fruit, reward_killed, reward_finished])
        self.reward_range = (min_reward, max_reward)

        # initialize buffer for observations for every agent
        self.obs_buffer = [deque(maxlen=self.history)] * self.num_agents

    def step(self, actions):

        # If necessary, converts actions to list, in case of single agent
        if self.num_agents == 1 and not isinstance(actions, list):
            actions = [actions]

        new_obs = []
        killed_on_step = [False] * self.num_agents
        rewards = [0.0] * self.num_agents

        # Check whether snake collides with something
        self._snake_collision(rewards, killed_on_step, actions)

        for i, k in enumerate(killed_on_step):
            if k:
                rewards[i] = self.reward_killed
                self.active_agents -= 1
                self.killed[i] = True

        # Adds penalty for each movement step
        for i, k in enumerate(rewards):
            if k == 0:
                rewards[i] -= 0.03

        done = False

        if self.active_agents <= 0:
            done = True

        for i in range(self.num_agents):
            ob = self._generate_obs(i)
            new_obs.append(ob)

        if self.num_agents == 1:
            new_obs = new_obs[0]
            rewards = rewards[0]

        return deepcopy(new_obs), deepcopy(rewards), self.killed, done

    def render(self, mode='human', wait=.15):

        if not self.arena.window_init:
            self.arena.pygame_init()

        self.arena.draw_env()

        for i, f in enumerate(self.fruits):
            self.arena.pygame_draw(self.arena.display_surf, self.arena.fruit_surf, f)

        for i, p in enumerate(self.agents):

            if self.killed[i]:
                continue

            p.draw(self.arena.display_surf, self.arena.agent_surfs[p.color_i], self.arena.agent_surfs_head[p.color_i])

        pygame.display.flip()

        time.sleep(wait)

    def reset(self):

        # clear observation buffers
        [ob.clear() for ob in self.obs_buffer]

        # reset collision map
        self.arena.coll_map = np.zeros((self.arena.dimensions, self.arena.dimensions)).astype(int)

        for i, p in enumerate(self.agents):
            self.killed[i] = False

            self._create_agent(i, self.init_length, create_object=False)

        self.num_active_fruits = self.num_fruits
        self.fruits = []
        for f in range(self.num_active_fruits):
            self.fruits.append(self.arena.generate_goal())

        self.active_agents = self.num_agents

        # reset returns observation of current state
        new_obs = []
        for i in range(self.num_agents):
            ob = self._generate_obs(i)
            new_obs.append(ob)

        if self.num_agents == 1:
            new_obs = new_obs[0]

        return deepcopy(new_obs)

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _snake_collision(self, rewards, killed_on_step, actions):
        """
        Helper function to check whether snake collides with wall, snake, itself, or fruit
        """

        for i, p in enumerate(self.agents):

            if self.killed[i]:
                continue

            # Agent takes action
            p.act(actions[i])
            p.update()

            found_fruit = False

            # Check head of snake for collision with fruit
            for f_i, f in enumerate(self.fruits):

                if f[0] == p.x[0] and f[1] == p.y[0]:

                    new_fruit_pos = self.arena.generate_goal()

                    if new_fruit_pos is not None:
                        self.fruits[f_i] = new_fruit_pos
                    else:
                        del self.fruits[f_i]

                    p.length += 1
                    rewards[i] = self.reward_fruit
                    found_fruit = True

            if not found_fruit:
                # remove tail from snake environment collision control
                self.arena.coll_map[p.x[p.length], p.y[p.length]] = 0
                p.remove_tail()

            # Checks if snake hits wall
            if p.x[0] < 0 or p.y[0] < 0 or p.x[0] >= self.arena.dimensions or p.y[0] >= self.arena.dimensions:
                killed_on_step[i] = True
                self.arena.remove_agent(i)
                continue

            # Checks if snake hits itself or other snake
            if self.arena.coll_map[p.x[0], p.y[0]] >= 2:
                agent_i = self.arena.coll_map[p.x[0], p.y[0]] - 2

                # snake hits itself
                if agent_i == i:
                    killed_on_step[i] = True
                    self.arena.remove_agent(i)

                # snake hits another snake
                else:
                    killed_on_step[i] = True
                    killed_on_step[agent_i] = True
                    self.arena.remove_agent(i)
                    self.arena.remove_agent(agent_i)

                continue

            # if snake has filled all grid cells, the game is won
            if p.length == self.arena.dimensions ** 2:
                rewards[i] += self.reward_finished
                self.active_agents -= 1
                self.arena.remove_agent(i)


            # mark new snake head position on collision map
            if not killed_on_step[i]:
                self.arena.coll_map[p.x[0], p.y[0]] = i + 2

    def get_active_agents(self):
        """
        returns active agents in arena
        """
        return [not self.killed[i] for i in range(self.num_agents)]

    def get_num_agents(self):
        """
        returns total number of agents
        """
        return self.num_agents

    def _create_agent(self, i, init_length, init_pose=None, create_object=True):
        """
        Creates snake agent using Agent class
        """

        if init_pose is None:
            x, y, direction, dx, dy = self.arena.sample_agent_position(init_length)

        else:
            x = init_pose['x']
            y = init_pose['y']
            direction = init_pose['direction']
            dx_array = [-1, 1, 0, 0]
            dy_array = [0, 0, 1, -1]
            dx = dx_array[direction]
            dy = dy_array[direction]

            is_free = True
            for j in range(init_length):
                if self.arena.coll_map[x + j * dx, y + j * dy] != 0:
                    is_free = False

            assert (is_free == True)

        if create_object:

            agent = Agent(x, y, self.arena.spacing, direction=direction, length=init_length,
                          max_buffer_length=self.arena.dimensions ** 2)

            agent.color_i = i % len(self.AGENT_COLORS)

        else:

            agent = self.agents[i]

            agent.reset(x, y, direction)

        # mark occupied grid cells with index: agent_index+2
        for j in range(init_length):
            self.arena.coll_map[x + j * dx, y + j * dy] = i + 2

        if create_object:
            return deepcopy(agent)

    def _generate_obs(self, agent):
        """
        Generates new observation for agent
        """
        if self.killed[agent]:
            obs = DEAD_OBS + np.zeros((self.arena.dimensions, self.arena.dimensions))
        else:
            # generate current observation
            obs = np.zeros((self.arena.dimensions, self.arena.dimensions)) + FREE_OBS

            # Adds agent's head to observation
            obs[self.agents[agent].x[0]][self.agents[agent].y[0]] = SELF_AGENT_HEAD_OBS

            # Adds rest of agent body to observation
            for i in range(1, self.agents[agent].length):
                obs[self.agents[agent].x[i]][self.agents[agent].y[i]] = SELF_AGENT_OBS

            for i, p in enumerate(self.agents):

                if self.killed[i] or i == agent:
                    continue

                # Adds head
                obs[self.agents[i].x[0]][self.agents[i].y[0]] = OTHER_AGENT_HEAD_OBS

                # Adds body
                for j in range(1, self.agents[i].length):
                    obs[self.agents[i].x[j]][self.agents[i].y[j]] = OTHER_AGENT_OBS

            for i, f in enumerate(self.fruits):
                obs[f[0]][f[1]] = FRUIT_OBS


        if self.flatten_states:
            obs = obs.flatten()

        else:
            obs = obs[:, :, None]

        # check observation buffer for this agent
        # if length of buffer smaller than required history - 1 (e.g. at the beginning of the episode)
        # fill with zeros
        for _ in range(self.history - len(self.obs_buffer[agent]) - 1):
            self.obs_buffer[agent].append(np.zeros_like(obs))

        # add current observation to buffer
        self.obs_buffer[agent].append(obs)

        # convert observation buffer to numpy array
        if self.flatten_states:
            obs_total = np.asarray(self.obs_buffer[agent]).flatten()
        else:
            # concatenate numpy arrays along 3rd axis
            obs_total = np.concatenate(self.obs_buffer[agent], axis=2)

        return deepcopy(obs_total)
