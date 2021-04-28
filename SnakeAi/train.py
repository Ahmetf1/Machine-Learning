from ai_game import GAME, SNAKE, FRUIT
from network import Network
import torch

Episodes = 2000

model = Network().to("cuda")
game = GAME()
action = None
print("training has strated !!")
for i in Episodes:
    game.spin_once(action)
    states = game.get_states()
    states = torch.from_numpy(states).to("cuda", dtype=torch.float)
    output = model(states)
    action = int(torch.argmax(output))
    print(action)