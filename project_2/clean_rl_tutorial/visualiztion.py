import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from pettingzoo.butterfly import pistonball_v6



def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x

# def agent
class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load agent
agent = torch.load("models/ppo_120.pt")

""" RENDER THE POLICY """
env = pistonball_v6.parallel_env(render_mode="rgb_array", continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)

agent.eval()
episodes = 5
episodes_frames = [[] for _ in range(episodes)]

print("Rendering 5 episodes of the trained policy.")

with torch.no_grad():
    # render 5 episodes out
    for episode in range(episodes):
        steps = 0
        obs = batchify_obs(env.reset(seed=None), device)
        terms = [False]
        truncs = [False]
        while not any(terms) and not any(truncs):
            print("step: ", steps)
            steps += 1
            episodes_frames[episode].append(env.render())
            actions, logprobs, _, values = agent.get_action_and_value(obs)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            obs = batchify_obs(obs, device)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]

# for each list in episodes_frames, save the frames as a gif
import imageio
for i, episode in enumerate(episodes_frames):
    imageio.mimsave(f"ppo_120_episode_{i}_slower.gif", episode, duration=50)
