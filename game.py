import pygame
import math
import random
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

while run_game:
    clock.tick(speed)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_game = False
        elif event.type == pygame.KEYDOWN:  # update direction of head from instruction
            if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and not (snake.get_dir() == 'right'):
                snake.set_dir('left')
            if (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and not (snake.get_dir() == 'left'):
                snake.set_dir('right')
            if (event.key == pygame.K_UP or event.key == pygame.K_w) and not (snake.get_dir() == 'down'):
                snake.set_dir('up')
            if (event.key == pygame.K_DOWN or event.key == pygame.K_s) and not (snake.get_dir() == 'up'):
                snake.set_dir('down')

    snake.move()  # Move body, not head
    curr_head_pos = snake.get_head()
    instr = pygame.key.get_pressed()

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

    # draw apple
    x = square_size * apple_coord[0]
    y = square_size * apple_coord[1]
    pygame.draw.rect(screen, 'red', [x, y, square_size, square_size])

    # draw snake
    for i in range(snake.get_size()):
        if i == 0:
            x = square_size * (snake.body[i][0][0])
            y = square_size * (snake.body[i][0][1])
            pygame.draw.rect(screen, 'blue', [x, y, square_size, square_size])
        else:
            x = square_size * (snake.body[i][0][0])
            y = square_size * (snake.body[i][0][1])
            pygame.draw.rect(screen, 'green', [x, y, square_size, square_size])

    # draw grid play area
    for i in range(1, board_dim):
        pygame.draw.line(screen, 'white', [square_size * i, 0], [square_size * i, y_max], grid_width)
        pygame.draw.line(screen, 'white', [0, square_size * i], [x_max, square_size * i], grid_width)

    pygame.display.flip()

pygame.quit()
