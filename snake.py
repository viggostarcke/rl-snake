import numpy as np


class Snake:
    def __init__(self, board_dim):
        head_start_pos = (board_dim - 7, board_dim - 4)
        self.body = [
            [head_start_pos, 'right'],
            [(head_start_pos[0] - 1, head_start_pos[1]), 'right'],
            [(head_start_pos[0] - 2, head_start_pos[1]), 'right']
        ]
        # self.body tracks both the position and direction of each part of the snake body

    # methods to get or set snake attributes
    def get_head(self):
        return self.body[0][0]

    def get_tail(self):
        return self.body[-1]

    def get_dir(self):
        return self.body[0][1]

    def get_size(self):
        return len(self.body)

    def set_head(self, coord):
        self.body[0][0] = coord

    def set_dir(self, instr):
        self.body[0][1] = instr

    def move(self):
        for i in reversed(range(1, self.get_size())):
            self.body[i][0] = self.body[i - 1][0]
            self.body[i][1] = self.body[i - 1][1]

    # each body part tracks the position and the direction the head was moving at a previous space
    # grow appends a new body part at the end of the tail segment using tail direction to position it
    def grow(self):
        curr_tail_pos, curr_tail_dir = self.get_tail()

        new_tail_pos = (0, 0)
        new_tail_dir = curr_tail_dir

        if curr_tail_dir == 'left':
            new_tail_pos = (curr_tail_pos[0] + 1, curr_tail_pos[1])
        if curr_tail_dir == 'right':
            new_tail_pos = (curr_tail_pos[0] - 1, curr_tail_pos[1])
        if curr_tail_dir == 'up':
            new_tail_pos = (curr_tail_pos[0], curr_tail_pos[1] + 1)
        if curr_tail_dir == 'down':
            new_tail_pos = (curr_tail_pos[0], curr_tail_pos[1] - 1)

        self.body.append([new_tail_pos, new_tail_dir])

    def check_self_collision(self):
        for i in range(1, self.get_size()):
            if (np.array(self.get_head()) == np.array(self.body[i][0])).all():
                return True
        return False

    def check_apple_coord(self, coord):
        for i in range(self.get_size()):
            if (np.array(coord) == np.array(self.body[i][0])).all():
                return True
        return False

    def reset(self, board_dim):
        head_start_pos = (board_dim - 7, board_dim - 4)
        self.body = [
            [head_start_pos, 'right'],
            [(head_start_pos[0] - 1, head_start_pos[1]), 'right'],
            [(head_start_pos[0] - 2, head_start_pos[1]), 'right']
        ]

    def check_wall_collision(self, board_dim):
        curr_head_pos = self.get_head()
        return curr_head_pos[0] < 0 or curr_head_pos[0] == board_dim or curr_head_pos[1] < 0 or curr_head_pos[1] == board_dim

    def check_apple_eat(self, apple):
        curr_head_pos = self.get_head()
        return (np.array(curr_head_pos) == np.array(apple)).all()
