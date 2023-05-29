import os
import time
import gym_super_mario_bros
import torch

from Environment import wrapping_env
from Agent import ActorCritic

def test(env, model):
    
    obs = env.reset()
    env.render()
    time.sleep(1)

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

        time.sleep(0.08)

        env.render()

    print("total reward : ", rewards)

if __name__ == "__main__":
    
    env = wrapping_env(gym_super_mario_bros.make("SuperMarioBros-1-2-v0"))

    model_path = os.path.join("./model", "stage1_2", "supermario_model_221209.pts")
    model_param = torch.load(model_path)
    model = ActorCritic(env.reset().__array__().shape[0], env.action_space.n)
    model.load_state_dict(model_param)

    test(env, model)
