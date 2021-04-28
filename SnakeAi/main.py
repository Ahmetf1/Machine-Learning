from network import Network, Agent
from ai_game import GAME
import torch


cell_size = 30
cell_number = 20

n_output = 4
lr_rate = 1e-3

gamma = 0.99
epsilon = 1
batch_size = 512
n_action = 4 
input_dims = [2, cell_number, cell_number]
mem_size = 100000
epsilon_end = 0.01
epsilon_dec = 5e-4

n_games = 10000

if __name__ == "__main__":
    game = GAME()
    game.set_timer(1)
    network = Network(n_output, lr_rate)
    agent = Agent(network, gamma, epsilon, batch_size, n_action, input_dims, mem_size, epsilon_end, epsilon_dec)

    scores = []

    score_max = 0
    for n in range(n_games):
        score = 0
        done = False
        observation = game.get_states()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, score = game.spin_once(action)
            agent.store_data(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            if score > score_max:
                score_max = score
                print(f"max score: {score_max}")
        if n % 100 == 0:
            print(f"episode {n} completed with score: {score}")
        if n % 10000 == 0:
            try:
                torch.save(network, "model")
            except:
                print("cant save")


