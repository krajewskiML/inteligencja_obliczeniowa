import gym
from stable_baselines3 import PPO
import json


env = gym.make("LunarLander-v2")

REPS = 10

# load ppo model
model = PPO.load("ppo_9_lunar_lander")

rewards = []

for rep in range(REPS):
    episode_rewards = []
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        env.render()
        if done:
            obs = env.reset()
            break
    rewards.append(episode_rewards)

# dump rewards to json file
with open("ppo_lunar_lander_rewards_best.json", "w") as f:
    json.dump(rewards, f)