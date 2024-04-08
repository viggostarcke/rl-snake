import pygame
import math
import random

from queue import PriorityQueue
from snake import Snake

pygame.init()
clock = pygame.time.Clock()

score = 0
x_max = 600
y_max = 600
board_dim = 10
square_size = math.floor(x_max / board_dim)
grid_width = 1
speed = 10
apple_coord = (random.randint(1, board_dim - 1), random.randint(1, board_dim - 1))
size = (x_max, y_max)
screen = pygame.display.set_mode(size)
snake = Snake(board_dim)
run_game = True


def get_neighbours(current):
    """
    get valid neighbours for the given tile considering game constraints

    :param current: current node getting
    :return: list of tuples consisting of all the valid tiles snakes head can move to
    """

    neighbours = []
    directions = [('up', (0, -1)), ('down', (0, 1)), ('left', (-1, 0)), ('right', (1, 0))]

    for direction, (dx, dy) in directions:
        neighbour = (current[0] + dx, current[1] + dy)

        if 0 <= neighbour[0] < board_dim and 0 <= neighbour[1] < board_dim:
            if neighbour not in [part[0] for part in snake.body[:-1]]:
                neighbours.append(neighbour)

    return neighbours


def heuristic(a, b):
    """
    get heuristic value from given tile to apple as the Manhattan distance between the two

    :param a: tile coords
    :param b: apple coords
    :return: manhattan distance between the two as int
    """

    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def coord_to_dir(coord):
    """

    :param next_step:
    :return:
    """

    dx = coord[0] - snake.get_head()[0]
    dy = coord[1] - snake.get_head()[1]

    if dx == 1:
        return 'right'
    elif dx == -1:
        return 'left'
    elif dy == 1:
        return 'down'
    elif dy == -1:
        return 'up'
    else:
        print("no valid coord_to_dir")
        return None


def a_star(start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]

        if current == apple_coord:
            break

        for next_node in get_neighbours(current):
            new_cost = cost_so_far[current] + 1
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                frontier.put((priority, next_node))
                came_from[next_node] = current

        if goal in came_from:
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            if len(path) > 1:
                return path[1]
            else:
                return path[0]

while run_game:
    clock.tick(speed)

    next_step = a_star(snake.get_head(), apple_coord)
    next_dir = coord_to_dir(next_step)
    snake.set_dir(next_dir)

    snake.move()  # Move body, not head
    curr_head_pos = snake.get_head()

    # use snake head direction to update position of head
    if snake.get_dir() == 'left':
        snake.set_head((curr_head_pos[0] - 1, curr_head_pos[1]))
    if snake.get_dir() == 'right':
        snake.set_head((curr_head_pos[0] + 1, curr_head_pos[1]))
    if snake.get_dir() == 'up':
        snake.set_head((curr_head_pos[0], curr_head_pos[1] - 1))
    if snake.get_dir() == 'down':
        snake.set_head((curr_head_pos[0], curr_head_pos[1] + 1))

    curr_head_pos = snake.get_head()

    # check if collision
    if snake.check_wall_collision(board_dim) or snake.check_self_collision():
        print("Score: {}".format(score))
        snake.reset(board_dim)
        while True:
            apple_coord = (random.randint(0, board_dim - 1), random.randint(0, board_dim - 1))
            if not snake.check_apple_coord(apple_coord):
                break
        score = 0

    # check if snake eats apple
    if curr_head_pos == apple_coord:
        snake.grow()
        score += 1
        while True:
            apple_coord = (random.randint(0, board_dim - 1), random.randint(0, board_dim - 1))
            if not snake.check_apple_coord(apple_coord):
                break

    # render
    screen.fill("black")

    bg_block_image = pygame.image.load('.img/black_square.png')
    bg_block_image = pygame.transform.scale(bg_block_image, (square_size, square_size))

    for row in range(board_dim):
        for col in range(board_dim):
            x = col * square_size
            y = row * square_size
            screen.blit(bg_block_image, (x, y))

    # draw apple
    apple_image = pygame.image.load('.img/apple.png')
    apple_image = pygame.transform.scale(apple_image, (square_size, square_size))

    x = square_size * apple_coord[0]
    y = square_size * apple_coord[1]
    screen.blit(apple_image, (x, y))
    # pygame.draw.rect(screen, 'red', [x, y, square_size, square_size])

    # draw snake
    arrow_left = pygame.image.load('.img/arrow_left.png')
    arrow_left = pygame.transform.scale(arrow_left, (square_size, square_size))

    arrow_right = pygame.image.load('.img/arrow_right.png')
    arrow_right = pygame.transform.scale(arrow_right, (square_size, square_size))

    arrow_up = pygame.image.load('.img/arrow_up.png')
    arrow_up = pygame.transform.scale(arrow_up, (square_size, square_size))

    arrow_down = pygame.image.load('.img/arrow_down.png')
    arrow_down = pygame.transform.scale(arrow_down, (square_size, square_size))

    for i in range(snake.get_size()):
        x = square_size * (snake.body[i][0][0])
        y = square_size * (snake.body[i][0][1])

        if snake.get_body_part_dir(i) == 'right':
            screen.blit(arrow_right, (x, y))
        elif snake.get_body_part_dir(i) == 'left':
            screen.blit(arrow_left, (x, y))
        elif snake.get_body_part_dir(i) == 'up':
            screen.blit(arrow_up, (x, y))
        elif snake.get_body_part_dir(i) == 'down':
            screen.blit(arrow_down, (x, y))

        # if i == 0:
        # #     pygame.draw.rect(screen, 'blue', [x, y, square_size, square_size])
        # # else:
        # #     pygame.draw.rect(screen, '#61DE2A', [x, y, square_size, square_size])

    # draw grid play area
    # for i in range(1, board_dim):
    #     pygame.draw.line(screen, 'black', [square_size * i, 0], [square_size * i, y_max], grid_width)
    #     pygame.draw.line(screen, 'black', [0, square_size * i], [x_max, square_size * i], grid_width)

    pygame.display.flip()

pygame.quit()
