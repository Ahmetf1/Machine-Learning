from network import Network, Agent
from ai_game import GAME
import torch
import matplotlib.pyplot as plt

import msvcrt

cell_size = 60
cell_number = 10

n_output = 4
lr_rate = 1e-3

gamma = 0.90
batch_size = 0
n_action = 3
input_dims = [2, cell_number, cell_number]
mem_size = 1000000

n_games = 10000


if __name__ == "__main__":
    game = GAME()
    game.set_timer(100)
    network = Network(n_output, lr_rate)
    agent = Agent(network, gamma, batch_size, n_action, input_dims, mem_size)

    scores = []
    avg_rewards = []

    score_max = 0
    n_total = 0

    for n in range(n_games):
        if not n == 0:
            scores.append(score)
            avg_rewards.append(total_reward / n_moves)
            agent.learn(n_total, n_total + n_moves)
        score = 0
        game.score = 0
        total_reward = 0
        n_moves = 0
        n_total = n_moves+n_total
        done = False
        observation = game.get_states()
        while not done:
            action = agent.choose_action(observation)
            if n < -1:
                game.draw()
                action = int(msvcrt.getch().decode("utf-8"))

            observation_, reward, done, score = game.spin_once(action, n_moves)
            agent.store_data(observation, action, reward, observation_, done)
            observation = observation_
            total_reward += reward
            n_moves += 1
            #game.draw()
            if score > score_max:
                print(score)
                score_max = score
                print(f"max score: {score_max}")
        if n % 100 == 0:
            print(f"episode {n} completed with score: {score}"
                  f"avg_reward: {total_reward / n_moves}")
        if n % 2000 == 0 and n != 0:
            try:
                torch.save(network, f"model_{n}")
            except:
                print("cant save")
        plt.plot(scores)
        plt.plot(avg_rewards)
        plt.pause(0.0000001)
