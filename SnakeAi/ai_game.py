import pygame
import sys
import numpy as np
from pygame.math import Vector2

cell_size = 60
cell_number = 10


class FRUIT:
    def __init__(self, screen):
        self.randomize()
        self.screen = screen

    def draw_fruit(self):
        fruit_rect = pygame.Rect(self.pos.x * cell_size, self.pos.y * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (255, 50, 50), fruit_rect)

    def randomize(self):
        x = np.random.randint(0, cell_number - 1)
        y = np.random.randint(0, cell_number - 1)
        self.pos = Vector2(x, y)


class SNAKE:
    def __init__(self, screen):
        self.body = [Vector2(4, 5), Vector2(3, 5), Vector2(2, 5)]
        self.direction = Vector2(1, 0)
        self.screen = screen

    def draw_snake(self):
        for block in self.body[1:]:
            block_rect = pygame.Rect(block.x * cell_size, block.y * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (5, 50, 5), block_rect)
        block = self.body[0]
        block_rect = pygame.Rect(block.x * cell_size, block.y * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (50, 150, 50), block_rect)

    def move_body(self):
        body_copy = self.body[:-1]
        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy


class GAME:
    def __init__(self):
        pygame.init()
        self.SCREEN_UPDATE = pygame.USEREVENT
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
        self.set_timer(1)

        self.snake = SNAKE(self.screen)
        self.fruit = FRUIT(self.screen)
        self.score = 0
        self.states = None
        self.is_updated = None
        self.reset()

    def set_timer(self, timer):
        pygame.time.set_timer(self.SCREEN_UPDATE, timer)

    def reset(self):
        self.snake = SNAKE(self.screen)
        self.fruit = FRUIT(self.screen)

    def check_collision(self):
        return self.snake.body[0] == self.fruit.pos

    def check_fail(self):
        for block in self.snake.body[1:]:
            fail = (self.snake.body[0] == block)
            if fail:
                return fail
        return not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number

    def randomize(self):
        self.fruit.randomize()
        for block in self.snake.body[1:]:
            if self.fruit.pos == block:
                self.randomize()

    def get_states(self):
        states = np.zeros((2, cell_number, cell_number))
        for n, block in enumerate(self.snake.body, start=1):
            states[0, int(block.x), int(block.y)] = 1 - 0.9 * (n / len(self.snake.body))
        states[1, int(self.fruit.pos.x), int(self.fruit.pos.y)] = 1
        return states

    def spin_once(self, action, n_moves):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        reward = -1
        done = False

        #        if action == 0:
        #            if self.snake.direction[1] != -1:
        #                self.snake.direction = (0, 1)
        #        elif action == 1:
        #            if self.snake.direction[1] != 1:
        #                self.snake.direction = (0, -1)
        #        elif action == 2:
        #            if self.snake.direction[0] != -1:
        #                self.snake.direction = (1, 0)
        #        elif action == 3:
        #            if self.snake.direction[0] != 1:
        #                self.snake.direction = (-1, 0)
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

        self.snake.move_body()

        if self.check_fail():
            reward = -10
            done = True
            self.reset()

        if self.check_collision():
            self.snake.body.insert(0, self.fruit.pos)
            self.score += 1
            self.randomize()
            reward = 10

        if n_moves > len(self.snake.body) * 100:
            reward = -10

        return self.get_states(), reward, done, self.score

    def draw(self):
        self.screen.fill((175, 215, 120))
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        pygame.display.update()


if __name__ == "__main__":
    game = GAME()
    input_a = int(input("input:"))
    while True:
        if input_a == "q":
            game.reset()
        else:
            game.spin_once(input_a)
            game.draw()
            print(game.snake.direction)
