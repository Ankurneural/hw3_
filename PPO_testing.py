import torch
import gym
from PPO import Feedforward

env = gym.make('Pendulum-v1')

# checkpoint = torch.load("C:\Apps\Masters_SJSU\Semester 2\CMPE 260\Assignment\Assignment 3\hw3_\ppo_actor.pth")
# pol = Feedforward(env.observation_space.shape[0], env.action_space.shape[0])
# pol.load_state_dict(checkpoint)

checkpoint = torch.load("C:\Apps\Masters_SJSU\Semester 2\CMPE 260\Assignment\Assignment 3\hw3_\VPG\ppo_actor.pth")
pol = Feedforward(env.observation_space.shape[0], env.action_space.shape[0])
pol.load_state_dict(checkpoint)


def rollout(policy, env):
    while True:
        obs = env.reset()
        done = False
        ep_rew = 0
        while not done:
            env.render()
            action = policy(obs).detach().numpy()
            obs, rew, done, _ = env.step(action)
            ep_rew += rew
        yield ep_rew

for i in rollout(pol, env):
    print(i)
    print("-----------")
