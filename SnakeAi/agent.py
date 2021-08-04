from collections import deque
import torch as T
import torch.nn as nn
import random


class Agent:
    def __init__(self, network_model, gamma, lr_rate, batch_size, mem_size, epsilon, epsilon_dec, epsilon_min):
        self.model = network_model
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon

        self.memory = deque(maxlen=mem_size)
        self.optimizer = T.optim.Adam(self.model.parameters(), lr_rate)
        self.loss = nn.MSELoss()
        self.device = "cuda:0"

    def learn(self, state, action, reward, new_state, done):
        T.set_grad_enabled(True)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.device)
        #done = T.tensor(done).to(self.device, dtype=T.int)

        if len(state.shape) == 1:
            state = T.unsqueeze(state, 0)
            action = T.unsqueeze(action, 0)
            reward = T.unsqueeze(reward, 0)
            new_state = T.unsqueeze(new_state, 0)
            done = (int(1),) if done else (int(0),)

        pred = self.model(state)
        target = pred.clone()

        for i in range(len(done)):
            if done[i]:
                q = reward[i]
            else:
                q = reward[i] + self.gamma * T.max(self.model(new_state[i]))
            target[i][T.argmax(action[i]).item()] = q

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss

    def store_data(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            sample = self.memory
        else:
            sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, new_states, dones = zip(*sample)
        loss = self.learn(states, actions, rewards, new_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, new_state, done):
        loss = self.learn(state, action, reward, new_state, done)
        return loss

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32).to(self.device)
        #if self.epsilon > self.epsilon_min:
        #   self.epsilon -= self.epsilon_dec
        action = T.multinomial(
            T.softmax(self.model(state).to(self.device), dim=1, dtype=T.float)[0], 1)[0]
        #if self.epsilon < T.rand(1):
        #    action = T.argmax(self.model(state).to(self.device))
        return action.item()
        #else:
        #    return T.randint(0,4,(1,1)).item()

