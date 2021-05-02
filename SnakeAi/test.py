from ai_game import GAME, SNAKE, FRUIT
from network import Network, Agent
import torch

gamma = 0.99
epsilon = 0.01
batch_size = 512
n_action = 4
input_dims = [2, 20, 20]
mem_size = 100000
epsilon_end = 0.01
epsilon_dec = 1e-4

network = torch.load('model_2000')
agent = Agent(network, gamma, epsilon, batch_size, n_action, input_dims, mem_size, epsilon_end, epsilon_dec)
game = GAME()
game.set_timer(150)

while True:
    observation = game.get_states()
    done = 0
    while not done:
        action = agent.choose_action(observation)
        observation = game.get_states()
        observation_, reward, done, score = game.spin_once(action)
        game.draw()
        observation = observation_
