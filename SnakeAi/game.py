import pygame
import sys
import numpy as np
from pygame.math import Vector2


class FRUIT:
    def __init__(self, screen, cell_size, cell_number):
        self.screen = screen
        self.cell_size = cell_size
        self.cell_number = cell_number
        self.pos = None
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(self.pos.x * self.cell_size, self.pos.y * self.cell_size, self.cell_size,
                                 self.cell_size)
        pygame.draw.rect(self.screen, (255, 50, 50), fruit_rect)

    def randomize(self):
        x = np.random.randint(0, self.cell_number - 1)
        y = np.random.randint(0, self.cell_number - 1)
        self.pos = Vector2(x, y)


class SNAKE:
    def __init__(self, screen, cell_size, cell_number):
        self.body = [Vector2(4, 5), Vector2(3, 5), Vector2(2, 5)]
        self.direction = Vector2(1, 0)
        self.screen = screen
        self.cell_size = cell_size
        self.cell_number = cell_number

    def draw_snake(self):
        for block in self.body[1:]:
            block_rect = pygame.Rect(block.x * self.cell_size, block.y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (5, 50, 5), block_rect)

        block = self.body[0]
        block_rect = pygame.Rect(block.x * self.cell_size, block.y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (50, 150, 50), block_rect)

    def move_body(self):
        body_copy = self.body[:-1]
        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy


class Game:
    def __init__(self, cell_size, cell_number):
        pygame.init()

        self.cell_size = cell_size
        self.cell_number = cell_number

        self.SCREEN_UPDATE = pygame.USEREVENT
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.cell_number * self.cell_size, self.cell_number * self.cell_size))
        self.set_timer(1)

        self.snake = SNAKE(self.screen, self.cell_size, self.cell_number)
        self.fruit = FRUIT(self.screen, self.cell_size, self.cell_number)
        self.score = 0
        self.n_moves = 0
        self.states = None
        self.is_updated = None
        self.reset()

    def set_timer(self, timer):
        pygame.time.set_timer(self.SCREEN_UPDATE, timer)

    def reset(self):
        self.snake = SNAKE(self.screen, self.cell_size, self.cell_number)
        self.fruit = FRUIT(self.screen, self.cell_size, self.cell_number)
        self.n_moves = 0

    def check_collision(self):
        return self.snake.body[0] == self.fruit.pos

    def check_fail(self):
        fail = False
        for block in self.snake.body[1:]:
            fail = (self.snake.body[0] == block)

        return not 0 <= self.snake.body[0].x < self.cell_number \
               or not 0 <= self.snake.body[0].y < self.cell_number or fail

    def randomize(self):
        self.fruit.randomize()
        for block in self.snake.body[1:]:
            if self.fruit.pos == block:
                self.randomize()

    def get_states(self):
        head_x = self.snake.body[0].x + 1
        head_y = self.snake.body[0].y + 1
        fruit_x = self.fruit.pos.x + 1
        fruit_y = self.fruit.pos.y + 1

        states = [
            fruit_y > head_y,
            fruit_x > head_x,
            fruit_y < head_y,
            fruit_x < head_x,
            self.snake.direction == (1, 0),
            self.snake.direction == (-1, 0),
            self.snake.direction == (0, 1),
            self.snake.direction == (0, -1),
            1 / head_x,
            1 / head_y,
            1 / (self.cell_number - head_x + 1),
            1 / (self.cell_number - head_y + 1),
            head_x == 1,
            head_y == 1,
            head_x == self.cell_number,
            head_y == self.cell_number
        ]
        return states

    def spin_once(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if action == 0:
            self.snake.direction = self.snake.direction
        elif action == 1:
            if self.snake.direction == (0, 1):
                self.snake.direction = (1, 0)
            elif self.snake.direction == (1, 0):
                self.snake.direction = (0, -1)
            elif self.snake.direction == (0, -1):
                self.snake.direction = (-1, 0)
            elif self.snake.direction == (-1, 0):
                self.snake.direction = (0, 1)
        elif action == 2:
            if self.snake.direction == (0, 1):
                self.snake.direction = (-1, 0)
            elif self.snake.direction == (1, 0):
                self.snake.direction = (0, 1)
            elif self.snake.direction == (0, -1):
                self.snake.direction = (1, 0)
            elif self.snake.direction == (-1, 0):
                self.snake.direction = (0, -1)
        #if action == 0:
        #    if self.snake.direction != (0, -1):
        #        self.snake.direction = (0, 1)
        #elif action == 1:
        #    if self.snake.direction != (0, 1):
        #        self.snake.direction = (0, -1)
        #elif action == 2:
        #    if self.snake.direction != (1, 0):
        #        self.snake.direction = (-1, 0)
        #elif action == 3:
        #    if self.snake.direction != (-1, 0):
        #        self.snake.direction = (1, 0)

        self.snake.move_body()
        self.n_moves += 1

        reward = -0.1
        done = False

        if self.check_fail():
            reward = -10
            done = True
            self.reset()

        elif self.check_collision():
            self.snake.body.insert(0, self.fruit.pos)
            self.score += 1
            self.randomize()
            reward = 10

        elif self.n_moves > len(self.snake.body) * 100:
            done = True
            print("reached to the limit")

        return self.get_states(), reward, done

    def draw(self):
        self.screen.fill((175, 215, 120))
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        pygame.display.update()
