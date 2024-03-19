import math
import random
import gymnasium as gym
import numpy as np
import pygame

from snake import Snake
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rbg_array"],
        "render_fps": 20
    }

    def __init__(self, render_mode=None, size=10):
        self.score = 0
        self.x_max = 600
        self.y_max = 600
        self.board_dim = size
        self.square_size = math.floor(self.x_max / self.board_dim)
        self.apple_coord = (random.randint(1, self.board_dim - 1), random.randint(1, self.board_dim - 1))
        self.snake = Snake(self.board_dim)

        # from direction snake head is heading: left, right or continue
        self.action_space = spaces.Discrete(3)

        # DICT: apple compass dir & 3 adjacent tiles
        self.observation_space = spaces.Dict(
            {
                'apple_direction': spaces.Discrete(8),
                'adjacent_tiles': spaces.Box(low=0, high=1, shape=(3,), dtype=int)
            }
        )

        # DICT: apple compass dir & 3x3 adjacent tiles grid
        # self.observation_space = spaces.Dict(
        #     {
        #         'apple_direction': spaces.Discrete(8),
        #         'adjacent_tiles': spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.float32)
        #     }
        # )

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

    def _get_observation(self):
        apple_dir = self.get_apple_dir(self.apple_coord)
        corrected_apple_dir = self.rotate_apple_dir(apple_dir)
        adjacent_tiles = np.array(self.get_adjacent_tiles())

        return {
            "apple_direction": corrected_apple_dir,
            "adjacent_tiles": adjacent_tiles
        }

    def _get_info(self):
        apple_dir = self.get_apple_dir(self.apple_coord)
        corrected_apple_dir = self.rotate_apple_dir(apple_dir)
        adjacent_tiles = np.array(self.get_adjacent_tiles())

        return {
            "apple_direction": corrected_apple_dir,
            "adjacent_tiles": adjacent_tiles
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
        # 3 action space
        curr_dir = self.snake.get_dir()
        new_dir = self._action_to_dir[curr_dir][int(action)]
        # 4 action space
        # action_dir_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
        # new_dir = action_dir_map[int(action)]

        # move snake body, then head
        self.snake.move()
        curr_head_pos = self.snake.get_head()
        self.move_head(new_dir, curr_head_pos)
        done = False
        reward = 0

        if self.snake.check_wall_collision(self.board_dim) or self.snake.check_self_collision():
            reward -= 1000
            self.reset_apple()
            done = True
        else:
            if self.snake.check_apple_eat(self.apple_coord):
                self.snake.grow()
                self.score += 1
                # reward += 100
                manhattan_dist = self.get_manhattan_dist(curr_head_pos, self.apple_coord)
                # print("head_pos: {}, ".format(curr_head_pos) + "apple: {}, ".format(self.apple_coord) + "MD: {}".format(manhattan_dist))
                reward += int(manhattan_dist) * 200
                self.reset_apple()
            else:
                reward -= 1

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

        bg_block_image = pygame.image.load('.img/black_square.png')
        bg_block_image = pygame.transform.scale(bg_block_image, (self.square_size, self.square_size))

        for row in range(self.board_dim):
            for col in range(self.board_dim):
                x = col * self.square_size
                y = row * self.square_size
                self.window.blit(bg_block_image, (x, y))

        # draw apple
        apple_image = pygame.image.load('.img/apple.png')
        apple_image = pygame.transform.scale(apple_image, (self.square_size, self.square_size))

        x = self.square_size * self.apple_coord[0]
        y = self.square_size * self.apple_coord[1]
        self.window.blit(apple_image, (x, y))
        # pygame.draw.rect(self.window, 'red', [x, y, self.square_size, self.square_size])

        # draw snake
        arrow_left = pygame.image.load('.img/arrow_left.png')
        arrow_left = pygame.transform.scale(arrow_left, (self.square_size, self.square_size))

        arrow_right = pygame.image.load('.img/arrow_right.png')
        arrow_right = pygame.transform.scale(arrow_right, (self.square_size, self.square_size))

        arrow_up = pygame.image.load('.img/arrow_up.png')
        arrow_up = pygame.transform.scale(arrow_up, (self.square_size, self.square_size))

        arrow_down = pygame.image.load('.img/arrow_down.png')
        arrow_down = pygame.transform.scale(arrow_down, (self.square_size, self.square_size))

        for i in range(self.snake.get_size()):
            x = self.square_size * (self.snake.body[i][0][0])
            y = self.square_size * (self.snake.body[i][0][1])

            if self.snake.get_body_part_dir(i) == 'right':
                self.window.blit(arrow_right, (x, y))
            elif self.snake.get_body_part_dir(i) == 'left':
                self.window.blit(arrow_left, (x, y))
            elif self.snake.get_body_part_dir(i) == 'up':
                self.window.blit(arrow_up, (x, y))
            elif self.snake.get_body_part_dir(i) == 'down':
                self.window.blit(arrow_down, (x, y))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick((self.metadata["render_fps"]))
        else:
            pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # gets compass direction apple is in relative to snake head
    # assumes snake direction is 'up'
    def get_apple_dir(self, apple):
        snake_head_x, snake_head_y = self.snake.get_head()
        apple_x, apple_y = apple

        if ((apple_x - snake_head_x) == 0) and ((apple_y - snake_head_y) < 0):
            # apple directly north of snake head
            return 0
        if ((apple_x - snake_head_x) < 0) and ((apple_y - snake_head_y) < 0):
            # apple northwest of snake head
            return 1
        if ((apple_x - snake_head_x) < 0) and ((apple_y - snake_head_y) == 0):
            # apple directly west of snake head
            return 2
        if ((apple_x - snake_head_x) < 0) and ((apple_y - snake_head_y) > 0):
            # apple southwest of snake head
            return 3
        if ((apple_x - snake_head_x) == 0) and ((apple_y - snake_head_y) > 0):
            # apple directly south of snake head
            return 4
        if ((apple_x - snake_head_x) > 0) and ((apple_y - snake_head_y) > 0):
            # apple southeast of snake head
            return 5
        if ((apple_x - snake_head_x) > 0) and ((apple_y - snake_head_y) == 0):
            # apple directly east of snake head
            return 6
        if ((apple_x - snake_head_x) > 0) and ((apple_y - snake_head_y) < 0):
            # apple northeast of snake head
            return 7
        else:
            # apple and snake head same tile
            return 8

    # rotates get_apple_dir to fit snakes direction
    def rotate_apple_dir(self, compass_dir):
        snake_dir = self.snake.get_dir()
        if snake_dir == 'left':
            return (compass_dir + 6) % 8
        elif snake_dir == 'down':
            return (compass_dir + 4) % 8
        elif snake_dir == 'right':
            return (compass_dir + 2) % 8
        return compass_dir

    def get_adjacent_tiles(self):
        snake_head_pos = np.array(self.snake.get_head())
        snake_head_dir = self.snake.get_dir()
        tiles = [0, 0, 0]
        adjacent_tiles = {
            'left': snake_head_pos + np.array([-1, 0]),
            'right': snake_head_pos + np.array([1, 0]),
            'up': snake_head_pos + np.array([0, -1]),
            'down': snake_head_pos + np.array([0, 1])
        }
        tiles_to_look = self._action_to_dir[snake_head_dir].values()
        adjacent_tiles = [adjacent_tiles[tile] for tile in tiles_to_look]
        assert len(adjacent_tiles) == 3

        for i, tile in enumerate(adjacent_tiles):
            if tile[0] < 0 or tile[0] == self.board_dim or tile[1] < 0 or tile[1] == self.board_dim:
                tiles[i] = 1
            elif self.snake.check_apple_eat(tile):  # reusing code for apple eat check (i.e. if tile is in body)
                tiles[i] = 1

        return tiles

    # def get_adjacent_tiles(self):
    #     grid_size = 3
    #     grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    #     snake_head_x, snake_head_y = self.snake.get_head()
    #
    #     for i in range(-1, 2):
    #         for j in range(-1, 2):
    #             pos_x, pos_y = snake_head_x + i, snake_head_y + j
    #             if pos_x < 0 or pos_x >= self.board_dim or pos_y < 0 or pos_y >= self.board_dim:
    #                 grid[i + 1, j + 1] = 0.9  # wall
    #             elif (pos_x, pos_y) in self.snake.get_body_coords():
    #                 grid[i + 1, j + 1] = 0.6  # body
    #             elif (pos_x, pos_y) == self.apple_coord:
    #                 grid[i + 1, j + 1] = 0.3  # apple
    #
    #     return grid
    #
    # # rotates get_adjacent_tiles to fit snakes direction
    # def rotate_adjacent_tiles(self, grid):
    #     rotations = {
    #         'left': 1,
    #         'down': 2,
    #         'right': 3
    #     }
    #
    #     num_rotations = rotations.get(self.snake.get_dir(), 0)
    #
    #     rotated_grid = np.rot90(grid, k=num_rotations)
    #
    #     return rotated_grid

    def get_manhattan_dist(self, coords1, coords2):
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

