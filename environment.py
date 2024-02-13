import math
import random
import gymnasium as gym
import numpy as np
import pygame

from snake import Snake
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rbg_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=10, prev_moves_count=10):
        self.score = 0
        self.x_max = 600
        self.y_max = 600
        self.board_dim = size
        self.square_size = math.floor(self.x_max / self.board_dim)
        self.grid_width = 1
        self.apple_coord = (random.randint(1, self.board_dim - 1), random.randint(1, self.board_dim - 1))
        self.snake = Snake(self.board_dim)
        self.prev_moves_count = prev_moves_count
        self.prev_moves = [0] * prev_moves_count

        # from direction snake head is heading: left, right or continue
        self.action_space = spaces.Discrete(3)
        """
        want to know position of snake head and apple relative to head (delta).
            and then length of snake + previous moves
        that way the agent knows the exact coordinates of every body part, and can learn to avoid
        prev_moves_count = number of prev moves known to agent.
            experiment w this. but: higher = more complex model for agent to learn
            lower = too little information to work with?
        array of 5 represents:
        first 2: apples x and y
        next 2: snake head x and y
        last: snake body length
        """
        prev_moves_count = 10
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0] + [0]*prev_moves_count),
            high=np.array([1, 1, 1, 1, 1] + [1]*prev_moves_count),
            dtype=np.float32)

        # 0: left, 1: continue, 2: right
        self._action_to_dir = {
            'left': {
                0: 'down',
                1: 'left',
                2: 'up'
            },
            'right': {
                0: 'up',
                1: 'right',
                2: 'down'
            },
            'up': {
                0: 'left',
                1: 'up',
                2: 'right'
            },
            'down': {
                0: 'right',
                1: 'down',
                2: 'left'
            }
        }

        self.window = None
        self.clock = None

    # start by apple x, apple y, snake head x, snake head y, snake length, prev moves
    # later implement delta and see diff
    def _get_observation(self):
        # check with one-liner when everything works:
        # apple_x, apple_y = self.apple_coord
        apple_x = self.apple_coord[0]
        apple_y = self.apple_coord[1]

        # check with one-liner when everything works:
        # snake_head_x, snake_head_y = self.snake.get_head()
        snake_head_x = self.snake.get_head()[0]
        snake_head_y = self.snake.get_head()[1]

        snake_size = self.snake.get_size()
        
        prev_moves = self.get_prev_moves()
        padding = [-1] * (self.prev_moves_count - len(prev_moves))
        prev_moves.extend(padding)
        
        observation = np.array([apple_x, apple_y, snake_head_x,
                                snake_head_y, snake_size] + prev_moves, dtype=np.float32)

        return observation

    def _get_info(self):
        return {
            "snake length": self.snake.get_size(),
            "apple position": self.apple_coord
        }

    # reset snake to mid of screen, apples current coordinates and score
    def reset(self, seed=None, options=None):
        self.snake.reset(self.board_dim)
        self.score = 0
        self.reset_apple()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        curr_dir = self.snake.get_dir()
        new_dir = self._action_to_dir[curr_dir][int(action)]
        # move snake body, then head
        self.snake.move()
        curr_head_pos = self.snake.get_head()
        self.move_head(new_dir, curr_head_pos)
        done = False
        reward = 0
        self.prev_moves.append(action)
        if len(self.prev_moves) > self.prev_moves_count:
            self.prev_moves.pop(0)

        if self.snake.check_wall_collision(self.board_dim) or self.snake.check_self_collision():
            reward -= 100
            done = True
        else:
            if self.snake.check_apple_eat(self.apple_coord):
                self.snake.grow()
                self.score += 1
                reward += 10
                self.reset_apple()
            else:
                reward -= 1

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, info, done

    # tries to place apple randomly, if apple is not in snake body returns
    def reset_apple(self):
        while True:
            self.apple_coord = tuple(self.np_random.integers(0, self.board_dim, size=2, dtype=int))
            if not self.snake.check_apple_coord(self.apple_coord):
                break

    #
    def move_head(self, dir, curr_head_pos):
        if dir == 'left':
            self.snake.set_head((curr_head_pos[0] - 1, curr_head_pos[1]))
            self.snake.set_dir('left')
        if dir == 'right':
            self.snake.set_head((curr_head_pos[0] + 1, curr_head_pos[1]))
            self.snake.set_dir('right')
        if dir == 'up':
            self.snake.set_head((curr_head_pos[0], curr_head_pos[1] - 1))
            self.snake.set_dir('up')
        if dir == 'down':
            self.snake.set_head((curr_head_pos[0], curr_head_pos[1] + 1))
            self.snake.set_dir('down')

    # def one_hot_encode_action(self, action):
    #     mapping = {
    #         'left': {0, 0, 0, 1},
    #         'right': {0, 1, 0, 0},
    #         'up': {1, 0, 0, 0},
    #         'down': {0, 0, 1, 0}
    #     }
    #
    #     return mapping[action]

    def get_prev_moves(self):
        # return [self.one_hot_encode_action(action) for action in self.prev_moves]
        return self.prev_moves

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            size = (self.x_max, self.y_max)
            self.window = pygame.display.set_mode(size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # render
        self.window.fill("black")

        # draw apple
        x = self.square_size * self.apple_coord[0]
        y = self.square_size * self.apple_coord[1]
        pygame.draw.rect(self.window, 'red', [x, y, self.square_size, self.square_size])

        # draw snake
        for i in range(self.snake.get_size()):
            x = self.square_size * (self.snake.body[i][0][0])
            y = self.square_size * (self.snake.body[i][0][1])
            pygame.draw.rect(self.window, 'green', [x, y, self.square_size, self.square_size])

        # draw grid
        for i in range(1, self.board_dim):
            pygame.draw.line(self.window, 'white', [self.square_size * i, 0], [self.square_size * i, self.y_max], self.grid_width)
            pygame.draw.line(self.window, 'white', [0, self.square_size * i], [self.x_max, self.square_size * i], self.grid_width)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick((self.metadata["render_fps"]))
        else:
            pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
