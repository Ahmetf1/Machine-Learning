from network import Network, Agent
from ai_game import GAME
import torch
import matplotlib.pyplot as plt

cell_size = 30
cell_number = 20

n_output = 4
lr_rate = 1e-3

gamma = 0.99
epsilon = 1
batch_size = 256
n_action = 4
input_dims = [2, cell_number, cell_number]
mem_size = 200000
epsilon_end = 0.01
epsilon_dec = 1e-4

n_games = 10000

if __name__ == "__main__":
    game = GAME()
    game.set_timer(10)
    network = Network(n_output, lr_rate)
    agent = Agent(network, gamma, epsilon, batch_size, n_action, input_dims, mem_size, epsilon_end, epsilon_dec)

    scores = []
    avg_rewards = []
    score_max = 0

    for n in range(n_games):
        if not n == 0:
            scores.append(score)
            avg_rewards.append(total_reward / n_moves)
        score = 0
        total_reward = 0
        n_moves = 0
        done = False
        observation = game.get_states()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, score = game.spin_once(action)
            game.draw()
            agent.store_data(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            total_reward += reward
            n_moves += 1
            if score > score_max:
                score_max = score
                print(f"max score: {score_max}")
        if n % 100 == 0:
            print(f"episode {n} completed with score: {score} epsilon: {agent.epsilon} "
                  f"avg_reward: {total_reward / n_moves}")
        if n % 2000 == 0 and n != 0:
            try:
                torch.save(network, f"model_{n}")
            except:
                print("cant save")
        plt.plot(scores)
        plt.plot(avg_rewards)
        plt.pause(0.001)
        print(score)
