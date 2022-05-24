# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import a3_gym_env

# define policy network
class Feedforward(nn.Module):
    """
    """
    def __init__(self, inp, out):
        super().__init__()
        self.layer1 = nn.Linear(inp, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out)

    def forward(self, obs):
        """
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        out_layer1 = F.relu(self.layer1(obs))
        out_layer2 = F.relu(self.layer2(out_layer1))
        return self.layer3(out_layer2)



# create environment
env = gym.make("Pendulum-v1-custom")
# instantiate the policy
policy = Feedforward( env.observation_space.shape[0], env.action_space.shape[0])
# create an optimizer
optimizer = torch.optim.Adam(policy.parameters())
# initialize gamma and stats
gamma=0.99
n_episode = 1
returns = deque(maxlen=100)
render_rate = 100 # render every render_rate episodes
while True:
    rewards = []
    actions = []
    states  = []
    # reset environment
    state = env.reset()
    while True:
        # render episode every render_rate epsiodes
        if n_episode%render_rate==0:
            env.render()

        # calculate probabilities of taking each action
        probs = policy(torch.tensor(state).unsqueeze(0).float())
        # sample an action from that set of probs
        sampler = Categorical(probs)
        action = sampler.sample()

        action = action.detach().numpy()

        # use that action in the environment
        # new_state, reward, done, info = env.step(action.item())
        new_state, reward, done, info = env.step(action)
        
        # store state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break

    # preprocess rewards
    rewards = np.array(rewards)
    # calculate rewards to go for less variance
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
    # or uncomment following line for normal rewards
    #R = torch.sum(torch.tensor(rewards))

    # preprocess states and actions
    states = torch.tensor(states).float()
    actions = torch.tensor(actions)

    # calculate gradient
    probs = policy(states)
    sampler = Categorical(probs)
    log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
    pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_episode, np.mean(returns)))
    n_episode += 1

# close environment
env.close()
