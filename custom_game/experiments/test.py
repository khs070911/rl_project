import pygame, time
import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, OneHotCategorical

from environment_naive import CustomEnvironment
from game.code.settings import screen_height, screen_width

def test():
    
    ## environment setting
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)
    env = CustomEnvironment(screen, size_rate=0.2, num_stack=4)
    
    actor_net = torch.load("./models/actor_checkpoint.pt")
    actor_net = [tmp for tmp in actor_net.module[0].modules()][1]
    
    done = False
    obs = env.reset()
    
    sum_reward = 0
    
    while not done:
        obs = obs / 255
        obs = torch.tensor(obs, dtype=torch.float64)
        with torch.no_grad():
            action_dist = actor_net(obs)
        action = action_dist.argmax()
        obs, reward, done = env.step(action)
        
        sum_reward += reward
        
    
    print(sum_reward)
        
        

if __name__ == "__main__":
    test()