import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, cell_number):
        super(Network, self).__init__()
        self.cell_number = cell_number
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), padding_mode="reflect")
        self.pool = nn.MaxPool2d(3)
        self.linear1 = nn.Linear(4 * 4 * 16, 10)
        self.linear2 = nn.Linear(10, 3)

    def forward(self, x):
        x = x.view(-1, 2, self.cell_number, self.cell_number)
        x = T.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 16)
        x = T.relu(self.linear1(x))
        x = F.dropout(x, 0.2)
        x = self.linear2(x)

        return x


class Network2(nn.Module):
    def __init__(self, cell_number):
        super(Network2, self).__init__()
        self.cell_number = cell_number
        self.linear1 = nn.Linear(300, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, 3)

    def forward(self, x):
        x = x.view(-1, 300)
        x = F.leaky_relu(self.linear1(x))
        x = F.dropout(x, 0.1)
        x = F.leaky_relu(self.linear2(x))
        x = F.dropout(x, 0.1)
        x = F.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Network3(nn.Module):
    def __init__(self, cell_number):
        super(Network3, self).__init__()
        self.cell_number = cell_number
        self.linear1 = nn.Linear(16, 200)
        self.linear2 = nn.Linear(200, 20)
        self.linear3 = nn.Linear(20, 50)
        self.linear4 = nn.Linear(50, 3)

    def forward(self, x):
        x = x.view(-1, 16)
        x = F.leaky_relu(self.linear1(x))
        #x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        return x
