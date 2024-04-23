import math
import random
import gymnasium as gym
import numpy as np
import pygame
import wandb

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
        self.curr_episode_length = 0
        self.last_apple_coord = (0, 0)
        self.hunger = 0
        self.stamina = self.board_dim ** 2

        # from direction snake head is heading: left, right or continue
        self.action_space = spaces.Discrete(3)

        # DICT: apple compass dir & 3 adjacent tiles
        # self.observation_space = spaces.Dict(
        #     {
        #         'apple_direction': spaces.Discrete(8),
        #         'adjacent_tiles': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        #     }
        # )

        # BOX: distance compass of tuples: (dist_to_obstacle, dist_to_apple)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 2), dtype=np.float32)

        # DICT: distance compass of tuples : (dist_to_obstacle, dist_to_apple) & 3 adjacent tiles
        self.observation_space = spaces.Dict(
            {
                'compass_distances': spaces.Box(low=0, high=1, shape=(8, 2), dtype=np.float32),
                'adjacent_tiles': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
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

    # DICT: apple compass dir & 3 adjacent tiles
    # def _get_observation(self):
    #     apple_dir = self.get_apple_dir(self.apple_coord)
    #     corrected_apple_dir = self.rotate_apple_dir(apple_dir)
    #     adjacent_tiles = np.array(self.get_adjacent_tiles())
    #
    #     return {
    #         "apple_direction": corrected_apple_dir,
    #         "adjacent_tiles": adjacent_tiles
    #     }

    # BOX: distance compass of tuples: (dist_to_obstacle, dist_to_apple)
    # def _get_observation(self):
    #     vision_rays = self.rotate_vision_rays(self.get_vision_rays())
    #     norm_vision_rays = np.copy(vision_rays).astype(np.float32)
    #     norm_vision_rays[:, 0] = norm_vision_rays[:, 0] / 10
    #     norm_vision_rays[:, 1] = norm_vision_rays[:, 1] / 10
    #
    #     return np.array(norm_vision_rays)

    def _get_observation(self):
        # DICT: distance compass of tuples : (dist_to_obstacle, dist_to_apple) & 3 adjacent tiles

        vision_rays = self.rotate_vision_rays(self.get_vision_rays())
        norm_vision_rays = np.copy(vision_rays).astype(np.float32)
        norm_vision_rays[:, 0] = norm_vision_rays[:, 0] / 10
        norm_vision_rays[:, 1] = np.array(norm_vision_rays[:, 1] / 10)

        adjacent_tiles = np.array(self.get_adjacent_tiles())

        return {
            "compass_distances": norm_vision_rays,
            "adjacent_tiles": adjacent_tiles
        }

    def _get_info(self):
        return {
            "episode": {"r": self.score, "l": self.curr_episode_length},
        }

    def reset(self, seed=None, options=None):
        # reset snake and apple's position, and score

        self.snake.reset(self.board_dim)
        self.score = 0
        self.reset_apple()
        self.hunger = 0
        observation = self._get_observation()
        # info = self._get_info()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # moves the snake according to given action, then checks for collisions with either obstacles or apple; and rewards accordingly.

        # 3 action space
        curr_dir = self.snake.get_dir()
        new_dir = self._action_to_dir[curr_dir][int(action)]
        # 4 action space: retired
        # action_dir_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
        # new_dir = action_dir_map[int(action)]

        # move snake's body, then head
        self.snake.move()
        curr_head_pos = self.snake.get_head()
        self.move_head(new_dir, curr_head_pos)

        done = False
        reward = 0

        if self.snake.check_wall_collision(self.board_dim) or self.snake.check_self_collision():
            reward -= 200
            self.reset_apple()
            done = True
        elif self.hunger >= self.stamina:
            reward -= 100
            done = True
        else:
            if self.snake.check_apple_eat(self.apple_coord):
                self.snake.grow()
                self.score += 1
                self.last_apple_coord = self.apple_coord
                reward += 300
                self.reset_apple()
                self.hunger = 0
            else:
                self.hunger += 1
                reward -= 1

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_observation()
        if done:
            info = self._get_info()
            # wandb.log({'episode_reward': self.score})
            self.curr_episode_length = 0
        else:
            info = {}
            self.curr_episode_length += 1

        return observation, reward, done, False, info

    def reset_apple(self):
        # resets apple's position

        while True:
            self.apple_coord = tuple(self.np_random.integers(0, self.board_dim, size=2, dtype=int))
            if not self.snake.check_apple_coord(self.apple_coord):
                break

    #
    def move_head(self, dir, curr_head_pos):
        # moves head according to given string dir

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

    def render(self):
        # renders the current game state

        return self._render_frame()

    def _render_frame(self):
        """
        updates the display window with current game state.
        """

        if self.window is None and self.render_mode == "human":
            pygame.init()
            size = (self.x_max, self.y_max)
            self.window = pygame.display.set_mode(size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # render
        self.window.fill("white")

        bg_block_image = pygame.image.load('.img/white_square.png')
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

    def get_apple_dir(self, apple):
        """
        gets compass direction apple is in assuming a direction of 'up'.

        :param apple: apple coordinates.
        :return: an integer indicating which compass direction apple is located in.
        """

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

    def rotate_apple_dir(self, compass_dir):
        """
        rotates get_apple_dir to fit snakes actual direction.

        :param compass_dir: return statement from get_apple_dir, an integer indicating what compass direction apple is located in.
        :return: rotated compass direction according to snake's direction.
        """

        snake_dir = self.snake.get_dir()
        if snake_dir == 'left':
            return (compass_dir + 6) % 8
        elif snake_dir == 'down':
            return (compass_dir + 4) % 8
        elif snake_dir == 'right':
            return (compass_dir + 2) % 8
        return compass_dir

    def get_adjacent_tiles(self):
        """
        gets 3 immediate surrounding tiles around snake's head, and assigns a value according to content of that tile.
        wall/body = 1.0, apple = 0.5, nothing = 0.0.

        :return: tuple containing 3 floats describing 3 adjacent tiles relative to snake's head and direction.
        """

        snake_head_pos = np.array(self.snake.get_head())
        snake_head_dir = self.snake.get_dir()
        tiles = [0.0, 0.0, 0.0]
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
                tiles[i] = 1.0
            elif self.snake.check_apple_eat(tile):  # reusing code for apple eat check (i.e. if tile is in body)
                tiles[i] = 1.0
            elif tile[0] == self.apple_coord[0] and tile[1] == self.apple_coord[1]:
                tiles[i] = 0.5

        return tiles

    def get_vision_rays(self):
        """
        gets 8 tuples, 1 for each compass direction, representing distance to nearest obstacle and distance to apple.

        :return: 8x2 element array containing tuples.
        """

        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]  # all compass directions clockwise
        vision = np.zeros((8, 2))

        for i, (dx, dy) in enumerate(directions):
            vision[i] = self.calc_distances(dx, dy)

        return vision

    def rotate_vision_rays(self, vision_rays):
        """
        rotates a given vision ray to fit snake's direction making it a compass of distance tuples.

        :param vision_rays: 8x2 element array containing tuples.
        :return: 8x2 element array containing tuples. order: start at forward and rotate clockwise
        """

        snake_dir = self.snake.get_dir()

        dir_steps = {
            'up': 0,
            'right': 2,
            'down': 4,
            'left': 6
        }

        steps = dir_steps[snake_dir]

        rotated_vision_rays = np.roll(vision_rays, shift=-steps, axis=0)

        return rotated_vision_rays

    def calc_distances(self, dx, dy):
        """
        calculates distance to nearest obstacle and apple for a coordinate relative to snake's head position.

        :param dx: x coordinate.
        :param dy: y coordinate.
        :return: tuple: (dist_to_obstacle, dist_to_apple).
        """

        x, y = self.snake.get_head()
        dist_to_obstacle = None
        dist_to_apple = None
        step_dist = 0

        while True:
            x += dx
            y += dy
            step_dist += 1

            if x < 0 or x >= self.board_dim or y < 0 or y >= self.board_dim:  # wall
                if dist_to_obstacle is None:
                    dist_to_obstacle = step_dist
                break

            if (x, y) in self.snake.get_body_coords() and dist_to_obstacle is None:
                dist_to_obstacle = step_dist

            if (x, y) == self.apple_coord and dist_to_apple is None:
                dist_to_apple = step_dist

            if dist_to_obstacle is not None and dist_to_apple is not None:
                break

        if dist_to_apple is None:
            dist_to_apple = 0

        return np.array([dist_to_obstacle, dist_to_apple])

    def calculate_spaciousness(self, start_position):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        queue = [(start_position, 0)]  # Each element is (position, move_count)
        visited = set()  # To track visited positions
        visited.add(start_position)

        reachable_states = 0

        while queue:
            current_position, move_count = queue.pop(0)
            if move_count < 3:  # Limit to 3 moves
                for dx, dy in moves:
                    next_position = (current_position[0] + dx, current_position[1] + dy)
                    if (0 <= next_position[0] < self.board_dim) and (
                            0 <= next_position[1] < self.board_dim):  # Stay within bounds
                        if next_position not in self.snake.get_body_coords() and next_position not in visited:
                            visited.add(next_position)
                            queue.append((next_position, move_count + 1))
                            reachable_states += 1

        return reachable_states
