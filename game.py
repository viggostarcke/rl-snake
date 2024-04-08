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
