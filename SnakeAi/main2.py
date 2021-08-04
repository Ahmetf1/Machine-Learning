from model import Network2, Network3
from game import Game
from agent import Agent
import time
import statistics
import matplotlib.pyplot as plt

max_episodes = 1000

if __name__ == "__main__":
    game = Game(cell_size=30, cell_number=10)
    model = Network3(cell_number=10).to("cuda:0")
    agent = Agent(network_model=model, gamma=0.9, lr_rate=0.00013629, batch_size=1000, mem_size=2500, epsilon=0.99
                  , epsilon_dec=0.01, epsilon_min=0.05)

    n_episodes = 0
    max_score = 0

    avg_rewards = []
    scores = []
    total_score = 0

    while n_episodes < max_episodes:
        if n_episodes > 1:
            avg_rewards.append(statistics.mean(episode_rewards))
            total_score += game.score
            scores.append(total_score / n_episodes)
        episode_rewards = []
        if game.score > max_score:
            max_score = game.score
            print("max score: ", max_score)
        n_episodes += 1
        game.score = 0
        if n_episodes > 15:
            loss = agent.train_long_memory()
        done = False
        while not done:
            old_state = game.get_states()
            action = agent.choose_action(old_state)
            state, reward, done = game.spin_once(action)
            agent.store_data(old_state, action, reward, state, done)
            agent.train_short_memory(old_state, action, reward, state, done)
            episode_rewards.append(reward)
            game.draw()
            #time.sleep(0.05)
            #breakpoint()

        plt.plot(avg_rewards)
        plt.plot(scores)
        plt.pause(0.1)
