import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):
    def __init__(self, n_output=4, lr_rate=1e-3):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.linear1 = nn.Linear(16 * 16 * 16, 20 * 20)
        self.linear2 = nn.Linear(20 * 20, 20)
        self.linear3 = nn.Linear(20, n_output)
        self.optimizer = T.optim.Adam(self.parameters(), lr_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss()
        print(f"device: {self.device}")

    def forward(self, x):
        x = x.view(-1, 2, 20, 20)
        x = T.relu(self.conv1(x))
        x = T.relu(self.conv2(x))
        x = x.view(-1, 16 * 16 * 16)
        x = T.relu(self.linear1(x))
        x = F.dropout(x, 0.2)
        x = T.relu(self.linear2(x))
        x = F.dropout(x, 0.2)
        x = self.linear3(x)
        return x


class Agent:
    def __init__(self, network_model, gamma, batch_size, n_action, input_dims, mem_size=100000):
        self.network_model = network_model
        self.gamma = gamma
        self.batch_size = batch_size
        self.actions = [i for i in range(n_action)]
        self.mem_size = int(mem_size)
        self.mem_ctr = 0
        self.state_memory = np.zeros((int(mem_size), *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((int(mem_size), *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(mem_size, dtype=np.bool)
        self.network_model.to(self.network_model.device)

    def store_data(self, state, action, reward, new_state, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float32).to(self.network_model.device)
        actions = T.multinomial(
            T.softmax(self.network_model.__call__(state).to(self.network_model.device), dim=1, dtype=T.float32), 4)
        action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return
        else:
            self.network_model.optimizer.zero_grad()
            max_mem = min(self.mem_size, self.mem_ctr)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.network_model.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.network_model.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.network_model.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.network_model.device)
            action_batch = T.tensor(self.action_memory[batch], dtype=T.long)
            q_eval = self.network_model.__call__(state_batch)[batch_index, action_batch]
            q_next = self.network_model.__call__(new_state_batch)
            q_next[terminal_batch] = 0.0

            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
            loss = self.network_model.loss(q_target, q_eval).to(self.network_model.device)
            loss.backward()
            self.network_model.optimizer.step()
