"""
    The file contains the PPO class to train with.
    NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
            It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

from tkinter import Variable
import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import torch.nn.functional as F
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
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
        """
        self.eval_actor_loss, self.critic_loss, self.eval_avgreturn = [], [], []
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor_network = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.critic_network = policy_class(self.obs_dim, 1)
        self.actor_optim = Adam(self.actor_network.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic_network.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(), 't_total': 0, 'i_total': 0,         
            'batch_lens': [], 'batch_rews': [], 'actor_losses': [], 'critic_losses':[]  
        }

    def train(self, total_steps):
        """
        """
        t_total ,i_total = 0,0 
        while t_total < total_steps:                                                        
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.batch_simulator()                     

            t_total += np.sum(batch_lens)
            i_total += 1

            self.logger['t_total'] = t_total
            self.logger['i_total'] = i_total

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()                                                                      
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):                                                       
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())

            self._log_summary()

            # Save our model
            if i_total % self.save_freq == 0:
                torch.save(self.actor_network.state_dict(), './ppo_actor.pth')
                torch.save(self.critic_network.state_dict(), './ppo_critic.pth')

        return self.eval_actor_loss, self.critic_loss, self.eval_avgreturn

    def batch_simulator(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = [] 
            obs = self.env.reset()
            done = False

            b_ob, b_act, b_log_prob, ep_rew, ep_t = self.episode_simulator(obs)
            t += ep_t
            # print(f"episode end {ep_t}")
            # print(f"total timestamp count: {t}")
            batch_obs.extend(b_ob)
            batch_acts.extend(b_act)
            batch_log_probs.extend(b_log_prob)
            batch_rews.append(ep_rew)
            batch_lens.append(ep_t+1)


        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.calc_rtgs(batch_rews)                                                             
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        print( np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']]))

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def episode_simulator(self, obs):

        b_obs, b_acts, b_log_probs, ep_rews = [], [], [], []
        for ep_t in range(self.max_timesteps_per_episode):
            
            # ti += 1 
            # print(f"total timestamp count: {ti}")
            # print(f"episodic count: {ep_t}")
            b_obs.append(obs)
            action, log_prob = self.action_sampler(obs)
            obs, rew, done, _ = self.env.step(action)
            
            ep_rews.append(rew)
            b_acts.append(action)
            b_log_probs.append(log_prob)

            if done:
                break
        
        return b_obs, b_acts, b_log_probs, ep_rews, ep_t

    def calc_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go 
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def action_sampler(self, obs):
        """
        """
        mean = self.actor_network(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic_network(batch_obs).squeeze()
        mean = self.actor_network(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 4800                 
        self.max_timesteps_per_episode = 1600           
        self.n_updates_per_iteration = 5                
        self.lr = 0.005                                
        self.gamma = 0.95                               
        self.clip = 0.2                                

        # Miscellaneous parameters
        self.render = True                             
        self.render_every_i = 10                       
        self.save_freq = 10                            
        self.seed = None                               

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_total = self.logger['t_total']
        i_total = self.logger['i_total']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])
        

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))


        print(flush=True)
        print(f"-------------------- Iteration #{i_total} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_total}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'], self.logger['batch_rews'], self.logger['actor_losses'], self.logger['critic_losses'] = [],[],[],[]

        self.eval_actor_loss.append(avg_actor_loss)
        self.eval_avgreturn.append(avg_ep_rews)
        self.critic_loss.append(avg_critic_loss)

class VPG():

    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
        """
        self.eval_actor_loss, self.eval_avgreturn = [], []
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor_network = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.actor_optim = Adam(self.actor_network.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(), 't_total': 0, 'i_total': 0,         
            'batch_lens': [], 'batch_rews': [], 'actor_losses': []
        }

    def train(self, total_steps):
        """
        """
        t_total ,i_total = 0,0 
        while t_total < total_steps:                                                        
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.batch_simulator()                     

            t_total += np.sum(batch_lens)
            i_total += 1

            self.logger['t_total'] = t_total
            self.logger['i_total'] = i_total

            # V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs                                                                     
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            surr1 = batch_log_probs * A_k
            actor_loss = (-surr1).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.logger['actor_losses'].append(actor_loss.detach())
            self._log_summary()

            # Save our model
            if i_total % self.save_freq == 0:
                torch.save(self.actor_network.state_dict(), './VPG/ppo_actor.pth')

        return self.eval_actor_loss, self.eval_avgreturn

    def batch_simulator(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = [] 
            obs = self.env.reset()
            done = False

            b_ob, b_act, b_log_prob, ep_rew, ep_t = self.episode_simulator(obs)
            t += ep_t
            # print(f"episode end {ep_t}")
            # print(f"total timestamp count: {t}")
            batch_obs.extend(b_ob)
            batch_acts.extend(b_act)
            batch_log_probs.extend(b_log_prob)
            batch_rews.append(ep_rew)
            batch_lens.append(ep_t+1)


        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, requires_grad=True)
        batch_rtgs = self.calc_rtgs(batch_rews)                                                             
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        print( np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']]))

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def episode_simulator(self, obs):

        b_obs, b_acts, b_log_probs, ep_rews = [], [], [], []
        for ep_t in range(self.max_timesteps_per_episode):
            
            # ti += 1 
            # print(f"total timestamp count: {ti}")
            # print(f"episodic count: {ep_t}")
            b_obs.append(obs)
            action, log_prob = self.action_sampler(obs)
            obs, rew, done, _ = self.env.step(action)
            
            ep_rews.append(rew)
            b_acts.append(action)
            b_log_probs.append(log_prob)

            if done:
                break
        
        return b_obs, b_acts, b_log_probs, ep_rews, ep_t

    def calc_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go 
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def action_sampler(self, obs):
        """
        """
        mean = self.actor_network(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic_network(batch_obs).squeeze()
        mean = self.actor_network(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 20000                 
        self.max_timesteps_per_episode = 4800           
        self.n_updates_per_iteration = 5                
        self.lr = 0.005                                
        self.gamma = 0.95                               
        self.clip = 0.2                                

        # Miscellaneous parameters
        self.render = True                             
        self.render_every_i = 10                       
        self.save_freq = 10                            
        self.seed = None                               

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_total = self.logger['t_total']
        i_total = self.logger['i_total']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
       

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
       

        print(flush=True)
        print(f"-------------------- Iteration #{i_total} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_total}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'], self.logger['batch_rews'], self.logger['actor_losses'] = [],[],[]

        self.eval_actor_loss.append(avg_actor_loss)
        self.eval_avgreturn.append(avg_ep_rews)



if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    model = PPO(Feedforward, env)
    actor_loss, c_loss, avg_return = model.train(45000)

    # model = VPG(Feedforward, env)
    # actor_loss, avg_return = model.train(450000)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    ax_right = ax.twinx()
    ax.plot(avg_return, color='black')
    ax_right.plot(actor_loss, color='red')
    ax.plot(c_loss, color='red')

    plt.show()


