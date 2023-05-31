import gymnasium as gym
import RL_

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    terminated=False
    i=0
    while not terminated:
        if i==0:
            action=0.1567
        else:
            action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        i+=1
    return sum_reward

env = gym.make("CustomEnv")
sum_reward = run_one_episode(env)
print(sum_reward)

