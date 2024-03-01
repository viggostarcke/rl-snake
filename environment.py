import math
import random
import gymnasium as gym
import numpy as np
import pygame

from snake import Snake
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rbg_array"], "render_fps": 20}

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

        # apple coords, snake head coords, 10 prev moves
        # prev_moves_count = 10
        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0, 0] + [0] * prev_moves_count),
        #     high=np.array([1, 1, 1, 1, 1] + [1] * prev_moves_count),
        #     dtype=np.float32)
        # self.observation_space = spaces.Discrete(15)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        # apple delta coords, adjacent fields
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

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

    """
    retired observation getter
    """
    # start by apple x, apple y, snake head x, snake head y, snake length, prev moves
    # later implement delta and see diff
    # def _get_observation(self):
    #     apple_x, apple_y = self.apple_coord
    #     snake_head_x, snake_head_y = self.snake.get_head()
    #     snake_size = self.snake.get_size()
    #     prev_moves = self.get_prev_moves()
    #     padding = [-1] * (self.prev_moves_count - len(prev_moves))
    #     prev_moves.extend(padding)
    #
    #     # normalize apple coords
    #     apple_x_norm = apple_x / (self.board_dim - 1)
    #     apple_y_norm = apple_y / (self.board_dim - 1)
    #
    #     # normalize snake head coords
    #     snake_head_x_norm = snake_head_x / (self.board_dim - 1)
    #     snake_head_y_norm = snake_head_y / (self.board_dim - 1)
    #
    #     # normalize snake size
    #     snake_size_norm = snake_size / self.board_dim
    #
    #     # normalize prev moves
    #     # 0.1: left
    #     # 0.2: continue
    #     # 0.3: right
    #     # 1.0: padding
    #     prev_moves_norm = [(move + 1) / self.board_dim if move != -1 else 1.0 for move in prev_moves]
    #
    #     observation = np.array([
    #                                apple_x_norm,
    #                                apple_y_norm,
    #                                snake_head_x_norm,
    #                                snake_head_y_norm,
    #                                snake_size_norm]
    #                            + prev_moves_norm, dtype=np.float32)
    #
    #     return observation

    def _get_observation(self):
        # get apple delta coords
        apple_coords = np.array(self.apple_coord)
        snake_head_coords = np.array(self.snake.get_head())
        apple_delta = apple_coords - snake_head_coords

        adjacent_fields = np.array(self.get_adjacent_fields())

        # normalize apple delta
        apple_delta_norm = apple_delta / (self.board_dim - 1)

        observation = np.concatenate((apple_delta_norm, adjacent_fields))

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

        if self.render_mode == "human":
            self._render_frame()

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
            reward -= 500
            self.reset_apple()
            done = True
        else:
            if self.snake.check_apple_eat(self.apple_coord):
                self.snake.grow()
                self.score += 1
                reward += 100
                self.reset_apple()
            else:
                reward -= 5

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, done, False, info

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

    def get_prev_moves(self):
        return self.prev_moves

    def render(self):
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
            if i == 0:
                x = self.square_size * (self.snake.body[i][0][0])
                y = self.square_size * (self.snake.body[i][0][1])
                pygame.draw.rect(self.window, 'blue', [x, y, self.square_size, self.square_size])
            else:
                x = self.square_size * (self.snake.body[i][0][0])
                y = self.square_size * (self.snake.body[i][0][1])
                pygame.draw.rect(self.window, 'green', [x, y, self.square_size, self.square_size])

        # draw grid
        for i in range(1, self.board_dim):
            pygame.draw.line(self.window, 'white', [self.square_size * i, 0], [self.square_size * i, self.y_max],
                             self.grid_width)
            pygame.draw.line(self.window, 'white', [0, self.square_size * i], [self.x_max, self.square_size * i],
                             self.grid_width)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick((self.metadata["render_fps"]))
        else:
            pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_adjacent_fields(self):
        snake_head_pos = np.array(self.snake.get_head())
        snake_head_dir = self.snake.get_dir()
        fields = [0, 0, 0]
        adjacent_fields = {
            'left': snake_head_pos + np.array([-1, 0]),
            'right': snake_head_pos + np.array([1, 0]),
            'up': snake_head_pos + np.array([0, -1]),
            'down': snake_head_pos + np.array([0, 1])
        }
        fields_to_look = self._action_to_dir[snake_head_dir].values()
        adjacent_fields = [adjacent_fields[field] for field in fields_to_look]
        assert len(adjacent_fields) == 3

        for i, field in enumerate(adjacent_fields):
            if field[0] < 0 or field[0] == self.board_dim or field[1] < 0 or field[1] == self.board_dim:
                fields[i] = 1
            elif self.snake.check_apple_eat(field):  # reusing code for apple eat check (i.e. if field is in body)
                fields[i] = 1

        return fields
