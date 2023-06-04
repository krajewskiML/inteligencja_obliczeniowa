from custom_env import ArrowAvoider
from stable_baselines3 import PPO

env = ArrowAvoider(render_mode='human', cones=16)

# load model from checkpoint
model = PPO.load("models/model_enhanced_rewarding_continue_gamma_999_2_270000_steps.zip")

# evaluate model
obs = env.reset()
for i in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
env.close()
