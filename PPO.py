import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.distributions import MultivariateNormal

import gym
import numpy as np


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


class PPO:
    """
    """
    def __init__(self, env, policy):
        
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]

        self.actor = policy(self.obs_dim, self.action_dim)
        self.critic = policy(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=0.001)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.001)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.gamma = 0.9
        self.update_iteration = 5
        self.clip = 0.2


    def train(self, timestamps):
        """
        """
        t_count = 0

        while t_count <= timestamps:
            b_obs, b_acts, b_log_probs, b_lens, b_rtgs = self.batch_simulator()

            V, _ = self.critic_evaluate(b_obs, b_acts)
            A = b_rtgs - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-10)

            for i in range(self.update_iteration):
                v , iter_log_prob = self.critic_evaluate(b_obs, b_acts)

                ratio = torch.exp(iter_log_prob - b_log_probs)
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(v, b_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()



            t_count += sum(b_lens)
    
    def critic_evaluate(self, batch_o, batch_a):
        """
        """
        V = self.critic(batch_o).squeeze()
        mean = self.actor(batch_o)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_a)
        # Return predicted values V and log probs log_probs
        return V, log_probs


    def batch_simulator(self, batch_maxlength = 4800):
        """
        """
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = [], [], [], [], [], []
        batch_t = 0
        while batch_t < batch_maxlength:
            ep_rew, ep_log_prob, ep_obs, ep_length, ep_action = self.episode_simulator()

            batch_rews.append(ep_rew)
            batch_log_probs.extend(ep_log_prob)
            batch_obs.extend(ep_obs)
            batch_lens.append(ep_length)
            batch_acts.extend(ep_action)

            batch_t += ep_length

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_lens, batch_rtgs

    def rtgs(self, rewards):
        """
        """
        res_rtgs = list()

        for ep_t in rewards:
            discounted_r = 0

            for rew in reversed(ep_t):
                discounted_r = rew + discounted_r*self.gamma
                # print(f"for {rew} in {discounted_r}")
                res_rtgs.insert(0, discounted_r)
        
        return torch.tensor(res_rtgs, dtype=torch.float)
    

    def episode_simulator(self, render=False, episodic_maxlength=1600):
        """
        """
        ep_rews, ep_logprob, ep_ob, ep_actions = [], [] , [], []
        obs = self.env.reset()
        done = False
        
        for ep_t in range(episodic_maxlength):
            if render:
                self.env.render()
            
            action, log_prob = self.action_sampler(obs)
            obs, rew, done, _ = self.env.step(action)

            ep_ob.append(obs)
            ep_rews.append(rew)
            ep_logprob.append(log_prob)
            ep_actions.append(action)

            if done:
                break
        
        return ep_rews, ep_logprob, ep_ob, ep_t, ep_actions

    def action_sampler(self, observation):
        """
        """
        action_mean = self.actor(observation)

        dist = MultivariateNormal(action_mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()


if __name__ == '__main__':
    e1 = gym.make('Pendulum-v1')
    p1 = PPO(e1, Feedforward)
    p1.train(100)

    # r = p1.episode_simulator(True, 10)
    