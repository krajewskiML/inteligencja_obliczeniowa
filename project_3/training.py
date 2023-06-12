from custom_env import ArrowAvoider
from stable_baselines3 import PPO

env = ArrowAvoider(cones=8)

# learn model from scratch for 500k steps and log info to tensorboard and try it with smaller gamma
model = PPO(policy= 'MlpPolicy', env=env, tensorboard_log="./logs/", gamma=0.9, verbose=1)


# learn model for 500k steps and save checkpoint every 10k steps using callback
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',name_prefix='model_just_alive_cones_8_gamma_9')

model.learn(total_timesteps=500000, callback=checkpoint_callback)