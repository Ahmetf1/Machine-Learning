import pygame
import random
import sys
from pygame.math import Vector2

cell_size = 30
cell_number = 20


class FRUIT:
    def __init__(self, screen):
        self.randomize()
        self.screen = screen

    def draw_fruit(self):
        fruit_rect = pygame.Rect(self.pos.x * cell_size, self.pos.y * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (255, 50, 50), fruit_rect)

    def randomize(self):
        x = random.randint(0, cell_number - 1)
        y = random.randint(0, cell_number - 1)
        self.pos = Vector2(x, y)


class SNAKE:
    def __init__(self, screen):
        self.body = [Vector2(7, 10), Vector2(6, 10), Vector2(5, 10)]
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
        pygame.time.set_timer(self.SCREEN_UPDATE, 150)
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))

        self.snake = SNAKE(self.screen)
        self.fruit = FRUIT(self.screen)
        self.score = 0

    def reset(self):
        self.snake = SNAKE(self.screen)
        self.fruit = FRUIT(self.screen)
        self.score = 0

    def update(self):
        self.snake.move_body()
        if self.check_fail():
            self.reset()
        if self.check_collision():
            self.snake.body.insert(0, self.fruit.pos)
            self.score += 1
            self.fruit.randomize()
            for block in self.snake.body[1:0]:
                if self.fruit.pos==block:
                    self.fruit.randomize()
            for block in self.snake.body[1:0]:
                if self.fruit.pos==block:
                    self.fruit.randomize()


    def check_collision(self):
        return self.snake.body[0] == self.fruit.pos

    def check_fail(self):
        for block in self.snake.body[1:]:
            fail = (self.snake.body[0] == block)
            if fail:
                return fail
        return not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == self.SCREEN_UPDATE:
                    self.update()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        if self.snake.direction[1] != -1:
                            self.snake.direction = (0, 1)
                    if event.key == pygame.K_UP:
                        if self.snake.direction[1] != 1:
                            self.snake.direction = (0, -1)
                    if event.key == pygame.K_RIGHT:
                        if self.snake.direction[0] != -1:
                            self.snake.direction = (1, 0)
                    if event.key == pygame.K_LEFT:
                        if self.snake.direction[0] != 1:
                            self.snake.direction = (-1, 0)
            self.screen.fill((175, 215, 120))
            self.fruit.draw_fruit()
            self.snake.draw_snake()
            pygame.display.update()
            self.clock.tick(30)


# pygame.init()
game = GAME()
game.main_loop()
# SCREEN_UPDATE = pygame.USEREVENT
# pygame.time.set_timer(SCREEN_UPDATE, 150)

# clock = pygame.time.Clock()
# screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
# surface = pygame.Surface((100, 100))
# screen.fill((100, 100, 100))
# rect = pygame.Rect(100, 100, 50, 50)
comment = """ while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            game.update()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                if game.snake.direction[1] != -1:
                    game.snake.direction = (0, 1)
            if event.key == pygame.K_UP:
                if game.snake.direction[1] != 1:
                    game.snake.direction = (0, -1)
            if event.key == pygame.K_RIGHT:
                if game.snake.direction[0] != -1:
                    game.snake.direction = (1, 0)
            if event.key == pygame.K_LEFT:
                if game.snake.direction[0] != 1:
                    game.snake.direction = (-1, 0)
    screen.fill((175, 215, 70))
    game.fruit.draw_fruit()
    game.snake.draw_snake()
    pygame.display.update()
    clock.tick(30)"""
