import pygame
import math
import random

from queue import PriorityQueue
from snake import Snake

pygame.init()
clock = pygame.time.Clock()

score = 0
x_max = 800
y_max = 800
board_dim = 10
square_size = math.floor(x_max / board_dim)
speed = 10
apple_coord = (random.randint(1, board_dim - 1), random.randint(1, board_dim - 1))
size = (x_max, y_max)
screen = pygame.display.set_mode(size)
snake = Snake(board_dim)
run_game = True
total_score = 0
num_games = 0


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
    converts coordinates of given tile to into a direction based on snake's head position

    :param coord: coordinates which should be corresponding to a tile adjacent to the position of the snake's head
    :return: direction snake needs to move to get to tile with coords
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
        return None


def is_move_safe(head, move, snake_body, depth=6):
    """
    determines if a proposed move is safe for the snake by searching a depth of 6 moves ahead.

    :param head: the current position of the snake's head.
    :param move: the proposed move as a tuple containing a coordinate.
    :param snake_body: snake's body coordinates in as a list.
    :param depth: the depth of the move lookahead. defaults to 6.
    :return: boolean value indicating if the move is safe.
    """

    new_head = (head[0] + move[0], head[1] + move[1])

    if not (0 <= new_head[0] < board_dim and 0 <= new_head[1] < board_dim):
        return False
    if new_head in snake_body:
        return False

    if depth == 0:
        return True

    for next_move in get_neighbours(new_head):
        if next_move != head:
            if not is_move_safe(new_head, (next_move[0] - new_head[0], next_move[1] - new_head[1]), snake_body[:-1] + [new_head], depth - 1):
                return False

    return True


def a_star(start, goal):
    """
    implements A* to find the shortest path from snake's head to apple.

    :param start: snake's current head position
    :param goal: apples position
    :return: a list representing the path from snake's head to apple as coordinates, or None in case no valid path exists.
    """

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

            return path


def find_next_move(start, goal, snake_body):
    """
    determines the next move towards the apple using A*, or a safe alternative if there is no valid path for A*.

    :param start: coordinates of snake's head
    :param goal: coordinates of the apple
    :param snake_body: coordinates of the snake's body as list
    :return: a list of the shortest path towards the apple, or the safest move in case no valid path can be found.
    """

    path = a_star(start, goal)
    if path is not None:
        return path

    safe_moves = []
    for move in get_neighbours(start):
        if is_move_safe(start, (move[0] - start[0], move[1] - start[1]), snake_body):
            safe_moves.append(move)

    if safe_moves:
        return [max(safe_moves, key=lambda m: heuristic(m, goal))]

    return None


while run_game:
    clock.tick(speed)

    path = find_next_move(snake.get_head(), apple_coord, snake.body)
    if path is None:
        total_score += score
        num_games += 1
        print("Game: {}, ".format(num_games) + "Score: {}, ".format(score) + "Avg. Score: {}".format(total_score / num_games))
        snake.reset(board_dim)
        while True:
            apple_coord = (random.randint(0, board_dim - 1), random.randint(0, board_dim - 1))
            if not snake.check_apple_coord(apple_coord):
                break
        score = 0
        continue

    next_step = path[1] if(len(path) > 1) else path[0]

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
        total_score += score
        num_games += 1
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
    screen.fill("white")

    bg_block_image = pygame.image.load('.img/white_square.png')
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

        if snake.get_body_part_dir(i + 1) == 'right':
            screen.blit(arrow_right, (x, y))
        elif snake.get_body_part_dir(i + 1) == 'left':
            screen.blit(arrow_left, (x, y))
        elif snake.get_body_part_dir(i + 1) == 'up':
            screen.blit(arrow_up, (x, y))
        elif snake.get_body_part_dir(i + 1) == 'down':
            screen.blit(arrow_down, (x, y))

    pygame.display.flip()

pygame.quit()
