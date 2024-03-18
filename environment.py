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
        self.grid_width = 1
        self.apple_coord = (random.randint(1, self.board_dim - 1), random.randint(1, self.board_dim - 1))
        self.snake = Snake(self.board_dim)

        # from direction snake head is heading: left, right or continue
        self.action_space = spaces.Discrete(3)
        # 0: left, 1: right, 2: up, 3: down
        # self.action_space = spaces.Discrete(4)

        # BOX: apple coords, snake head coords, 10 prev moves
        # prev_moves_count = 10
        # self.observation_space = spaces.Box(
        #     low=np.array([0, 0, 0, 0, 0] + [0] * prev_moves_count),
        #     high=np.array([1, 1, 1, 1, 1] + [1] * prev_moves_count),
        #     dtype=np.float32)
        # self.observation_space = spaces.Discrete(15)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

        # BOX: apple delta coords, adjacent tiles
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        # BOX: apple coords, snake head coords
        # self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # DICT: apple coords, snake head coords, adjacent tiles
        self.observation_space = spaces.Dict(
            {
                'apple_direction': spaces.Discrete(8),
                'adjacent_tiles': spaces.Box(low=0, high=1, shape=(3,), dtype=int)
            }
        )

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
    # apple x, apple y, snake head x, snake head y, snake length, prev moves
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

    """
    retired observation getter
    """
    # apple coords (relative to head coords), info from 3 adjacent tiles
    # def _get_observation(self):
    #     # get apple delta coords
    #     apple_coords = np.array(self.apple_coord)
    #     snake_head_coords = np.array(self.snake.get_head())
    #     apple_delta = apple_coords - snake_head_coords
    #
    #     adjacent_tiles = np.array(self.get_adjacent_tiles())
    #
    #     # normalize apple delta
    #     apple_delta_norm = apple_delta / (self.board_dim - 1)
    #
    #     observation = np.concatenate((apple_delta_norm, adjacent_tiles))
    #
    #     return observation

    """
    retired observation getter
    """
    # apple coords, snake head coords
    # def _get_observation(self):
    #     # get apple coords
    #     apple_coords = np.array(self.apple_coord)
    #     # normalize apple coords
    #     apple_coords_norm = apple_coords / (self.board_dim - 1)
    #     # get snake head coords
    #     snake_head_coords = np.array(self.snake.get_head())
    #     # normalize snake head coords
    #     snake_head_coords_norm = snake_head_coords / (self.board_dim - 1)
    #
    #     observation = np.concatenate((apple_coords_norm, snake_head_coords_norm))
    #
    #     return observation

    def _get_observation(self):
        apple_dir = self.get_apple_dir(self.apple_coord)
        corrected_apple_dir = self.rotate(apple_dir)
        adjacent_tiles = np.array(self.get_adjacent_tiles())

        return {
            "apple_direction": corrected_apple_dir,
            "adjacent_tiles": adjacent_tiles
        }

    def _get_info(self):
        apple_dir = self.get_apple_dir(self.apple_coord)
        corrected_apple_dir = self.rotate(apple_dir)
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
            reward -= 500
            self.reset_apple()
            done = True
        else:
            if self.snake.check_apple_eat(self.apple_coord):
                self.snake.grow()
                self.score += 1
                # reward += 100
                manhattan_dist = self.get_manhattan_dist(curr_head_pos, self.apple_coord)
                reward += int(manhattan_dist) * 100
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

            if self.snake.get_body_part_dir(i-1) == 'right':
                self.window.blit(arrow_right, (x, y))
            elif self.snake.get_body_part_dir(i-1) == 'left':
                self.window.blit(arrow_left, (x, y))
            elif self.snake.get_body_part_dir(i-1) == 'up':
                self.window.blit(arrow_up, (x, y))
            elif self.snake.get_body_part_dir(i-1) == 'down':
                self.window.blit(arrow_down, (x, y))

            # if i == 0:
            #     pygame.draw.rect(self.window, 'blue', [x, y, self.square_size, self.square_size])
            # else:
            #     pygame.draw.rect(self.window, 'green', [x, y, self.square_size, self.square_size])

        # draw grid
        # for i in range(1, self.board_dim):
        #     pygame.draw.line(self.window, 'white', [self.square_size * i, 0], [self.square_size * i, self.y_max],
        #                      self.grid_width)
        #     pygame.draw.line(self.window, 'white', [0, self.square_size * i], [self.x_max, self.square_size * i],
        #                      self.grid_width)

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
    def rotate(self, compass_dir):
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

    def get_manhattan_dist(self, coords1, coords2):
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

