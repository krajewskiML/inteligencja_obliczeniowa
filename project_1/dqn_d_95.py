import gym
from stable_baselines3 import DQN

env = gym.make("LunarLander-v2")

# create the DQN model with tensorboard
model = DQN('MlpPolicy', env, verbose=1, gamma=0.95, tensorboard_log="./dqn_95_lunar_lander_tensorboard/")
# train the agent for 1,000,000 steps
model.learn(total_timesteps=500_000)
model.save("dqn_95_lunar_lander")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_95_lunar_lander")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()