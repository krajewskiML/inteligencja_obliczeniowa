from custom_env import ArrowAvoider
from stable_baselines3 import PPO

env = ArrowAvoider(render_mode='human', cones=4)

# load model from checkpoint
model = PPO.load("models\model_just_alive_rewarding_gamma_99_cones_4_continue_240000_steps.zip")

# evaluate model
obs, _ = env.reset()
for i in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _,  info = env.step(action)
    env.render()
    if dones:
        obs, _ = env.reset()
env.close()
