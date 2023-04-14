import gym
import RL_

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    done=False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
    return sum_reward

env = gym.make("CustomEnv")
sum_reward = run_one_episode(env)
print(sum_reward)