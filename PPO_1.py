import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.distributions import MultivariateNormal

import gym
import numpy as np
import matplotlib.pyplot as plt


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

        self.gamma = 0.95
        self.update_iteration = 10
        self.clip = 0.2


    def train(self, timestamps):
        """
        """
        t_count = 0
        actor_loss_l , batch_mean_r = list(), list()

        while t_count <= timestamps:
            b_obs, b_acts, b_log_probs, b_lens, b_rtgs, b_rews = self.rollout() 

            V, _ = self.critic_evaluate(b_obs, b_acts)
            A = b_rtgs - V.detach()
            A = (A - A.mean()) / (A.std() + 1e-10)

            for i in range(self.update_iteration):
                V , iter_log_prob = self.critic_evaluate(b_obs, b_acts)

                ratio = torch.exp(iter_log_prob - b_log_probs)
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, b_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()

            print(f"Average Batch rewards: {np.mean([np.sum(i) for i in b_rews])}")
            batch_mean_r.append(np.mean([np.sum(i) for i in b_rews]))
            actor_loss_l.append(actor_loss.detach())


            t_count += sum(b_lens)

        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')
    
        return batch_mean_r, actor_loss_l
    
    def critic_evaluate(self, batch_o, batch_a):
        """
        """
        V = self.critic(batch_o).squeeze()
        mean = self.actor(batch_o)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_a)
        # Return predicted values V and log probs log_probs
        return V, log_probs


    def batch_simulator(self, batch_maxlength = 20000):
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

        return batch_obs, batch_acts, batch_log_probs, batch_lens, batch_rtgs, batch_rews

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


    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < 4800:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(1800):
                # If render is specified, render the environment
                # if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                #     self.env.render()

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.action_sampler(obs)
                obs, rew, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.rtgs(batch_rews)                                                              # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        # self.logger['batch_rews'] = batch_rews
        # self.logger['batch_lens'] = batch_lens

        print( np.mean([np.sum(ep_rews) for ep_rews in batch_rews]))

        return batch_obs, batch_acts, batch_log_probs, batch_lens, batch_rtgs, batch_rews


if __name__ == '__main__':
    e1 = gym.make('Pendulum-v1')
    p1 = PPO(e1, Feedforward)
    bm, am = p1.train(5000000)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    ax_right = ax.twinx()
    ax.plot(bm, color='black')
    ax_right.plot(am, color='red')
    plt.show()

    # r = p1.episode_simulator(True, 10)
    