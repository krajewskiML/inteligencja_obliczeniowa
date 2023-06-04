import gymnasium
from stable_baselines3 import PPO
import json


env = gymnasium.make("LunarLander-v2", render_mode="human")

REPS = 10

# load ppo model
model = PPO.load("models\gamma_0.999_rep_9_arch_64")

rewards = []

for rep in range(REPS):
    episode_rewards = []
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        episode_rewards.append(reward)
        env.render()
        if done:
            obs = env.reset()
            break
    rewards.append(episode_rewards)

# dump rewards to json file
with open("ppo_lunar_lander_rewards_best.json", "w") as f:
    json.dump(rewards, f)