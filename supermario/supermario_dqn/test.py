import os
import time
import gym_super_mario_bros
import torch

from normalized_env import wrapping_env

def test(env, model):
    
    obs = env.reset()
    done = False
    rewards = 0

    model.eval()
    while not done:
        
        obs = torch.tensor(obs.__array__()).unsqueeze(0)
        q_value = model(obs)[0]
        best_action = q_value.argmax()

        next_obs, reward, done, _ = env.step(best_action.item())
        rewards += reward

        obs = next_obs

        time.sleep(0.1)

        env.render()

    print("total reward : ", rewards)

if __name__ == "__main__":
    
    env = wrapping_env(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"))
    env.seed(110)

    model_path = os.path.join("./models", 'old2', "model_mario.pt")
    # model_path = os.path.join("./models", "model_mario.pt")
    model = torch.load(model_path)

    test(env, model)
