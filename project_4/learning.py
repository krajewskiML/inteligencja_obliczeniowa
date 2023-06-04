import gymnasium
from stable_baselines3 import PPO
import torch

REPS = 10
gammas = [0.97, 0.98, 0.99, 0.995, 0.999]
STEPS = 150_000
archs = [{"pi": [32, 16], "vf": [32, 16]}, {'pi': [64, 64], "vf": [64, 64]}]

for arch in archs:
    for gamma in gammas:
        for rep in range(REPS):
            # Create environment lunar lander
            env = gymnasium.make('LunarLander-v2')
            model_name = f"gamma_{gamma}_rep_{rep}_arch_{arch['pi'][0]}"
            policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=arch)
            model = PPO('MlpPolicy', env, verbose=1, gamma=gamma, tensorboard_log=f"./tensorboards/{model_name}/", policy_kwargs=policy_kwargs)
            model.learn(total_timesteps=STEPS)
            model.save(f"models/{model_name}")
            del model





