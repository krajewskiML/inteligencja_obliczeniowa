from custom_env import ArrowAvoider
from stable_baselines3 import PPO

env = ArrowAvoider(cones=16)

# learn PPo model for 500k steps and log info to tensorboard
model = PPO.load("models/model_enhanced_rewarding_continue_gamma_999_390000_steps.zip", env=env, tensorboard_log="./logs/", gamma=0.999)


# learn model for 500k steps and save checkpoint every 10k steps using callback
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',name_prefix='model_enhanced_rewarding_continue_gamma_999_2')

model.learn(total_timesteps=500000, callback=checkpoint_callback)