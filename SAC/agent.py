import os
import sys
from pathlib import Path
import logging
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import copy
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import ReplayBuffer
from src.networks import BaseActorNetwork, DoubleQCriticNetwork, mlp, cnn

class ActorNetwork(BaseActorNetwork):
    def __init__(self, state_size, action_size, fc_hidden_units=None, conv_layers=None, 
                 log_std_interval=None, use_batchnorm=False, dropout_rate=0.0):
        super().__init__(state_size, action_size, fc_hidden_units, conv_layers, 
                         use_batchnorm, dropout_rate)
        self.log_std_interval = log_std_interval
        self.tanh_transform = TanhTransform(cache_size=1)

    def create_cnn(self, input_dim, action_size, conv_layers, fc_hidden_units, 
                   use_batchnorm, dropout_rate):
        return cnn(input_channels=input_dim,
                   output_dim=2 * action_size,
                   conv_layers=conv_layers,
                   fc_hidden_units=fc_hidden_units,
                   use_batchnorm=use_batchnorm,
                   dropout_rate=dropout_rate)

    def create_mlp(self, input_dim, action_size, fc_hidden_units, dropout_rate):
        return mlp(input_dim=input_dim,
                   output_dim=2 * action_size,
                   hidden_units=fc_hidden_units,
                   dropout_rate=dropout_rate)

    def forward(self, state):
        # Add batch dimension if processing a single image
        state = state.unsqueeze(0) if state.dim() == 3 and self.image_input else state
        mu, log_std = self.actor(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, *self.log_std_interval)
        std = torch.exp(log_std)
        return TransformedDistribution(Normal(mu, std), self.tanh_transform)


class SACAgent:
    def __init__(self, 
                 state_size,
                 action_size,
                 hidden_units=None,
                 conv_layers=None,
                 log_std_interval=None,
                 use_batchnorm=False,
                 dropout_rate=0.0,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 temperature_lr=1e-4,
                 gamma=0.99,
                 tau=0.005,
                 init_temperature=0.1,
                 actor_update_freq=2,
                 critic_target_update_freq=2,
                 replay_buffer_size=int(1e5),
                 batch_size=128,
                 data_type=torch.float32,
                 device=torch.device('cpu'),
                 checkpoint_dir='./checkpoints'):
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_size
        
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        
        self.actor = ActorNetwork(state_size=state_size,
                                  action_size=action_size,
                                  fc_hidden_units=hidden_units,
                                  conv_layers=conv_layers,
                                  log_std_interval=log_std_interval, 
                                  use_batchnorm=use_batchnorm,
                                  dropout_rate=dropout_rate).to(self.device, data_type)
        self.double_critic = DoubleQCriticNetwork(state_size=state_size,
                                                  action_size=action_size,
                                                  fc_hidden_units=hidden_units,
                                                  conv_layers=conv_layers, 
                                                  ).to(self.device, data_type)
        self.double_critic_target = copy.deepcopy(self.double_critic).to(self.device, data_type)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.double_critic_optimizer = optim.Adam(self.double_critic.parameters(), lr=critic_lr)
        self.temperature_optimizer = optim.Adam([self.log_alpha], lr=temperature_lr)
        
        # Freeze target network with respect to optimizers to avoid unnecessary computations
        for param in self.double_critic_target.parameters():
            param.requires_grad = False
        
        self.replay_buffer = ReplayBuffer(state_size, action_size, self.device, data_type, replay_buffer_size)
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_model_path = None
        self.last_model_path = None
        self.best_reward = float('-inf')
        
        self.rewards_file = None
        self.losses_file = None
        self.replay_buffer_file = None
        
        self.to_train_mode()
        self.update_counter = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def to_train_mode(self):
        self.actor.train()
        self.double_critic.train()

    def to_eval_mode(self):
        self.actor.eval()
    
    def to_cpu(self):
        self.actor.to('cpu')

    def take_action(self, state):
        policy = self.actor(state)
        return policy.sample().squeeze().detach().cpu().numpy()
    
    def save_training_data(self, rewards_memory, losses, save_replay_buffer=True):
        with open(str(self.rewards_file), 'wb') as f:
            pickle.dump(rewards_memory, f)
        with open(str(self.losses_file), 'wb') as f:
            pickle.dump(losses, f)
        
        if save_replay_buffer:
            with open(str(self.replay_buffer_file), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
    
    def load_training_data(self, experiment_number):
        self.rewards_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_rewards.pkl')
        self.losses_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_losses.pkl')
        self.replay_buffer_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_replay_buffer.pkl')
        
        if os.path.exists(self.rewards_file):
            with open(self.rewards_file, 'rb') as f:
                rewards_memory = pickle.load(f)
        else:
            rewards_memory = []
        if os.path.exists(self.losses_file):
            with open(self.losses_file, 'rb') as f:
                losses = pickle.load(f)
        else:
            losses = {'critic': [], 'actor': [], 'temperature': []}
        if os.path.exists(self.replay_buffer_file):
            with open(self.replay_buffer_file, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        return rewards_memory, losses
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        next_policies = self.actor(next_states)
        next_actions = next_policies.sample()
        log_prob_next_actions = next_policies.log_prob(next_actions)
        with torch.no_grad():
            q1_target, q2_target = self.double_critic_target(next_states, next_actions)
            v_target = torch.min(q1_target, q2_target) - self.alpha.detach() * log_prob_next_actions
            q_target = rewards + self.gamma * (~dones) * v_target.mean(dim=-1, keepdim=True)
            q_target = q_target.detach()
        
        current_q1, current_q2 = self.double_critic(states, actions)
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)
        
        self.double_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.double_critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor_and_temperature(self, states):
        policies = self.actor(states)
        actions = policies.rsample()
        log_prob_actions = policies.log_prob(actions)
        
        current_q1, current_q2 = self.double_critic(states, actions)
        current_q = torch.min(current_q1, current_q2)
        
        actor_loss = (self.alpha.detach() * log_prob_actions - current_q).mean(dim=-1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        temperature_loss = -(self.alpha * (log_prob_actions + self.target_entropy).detach()).mean()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        
        return actor_loss.item(), temperature_loss.item()
    
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self, losses):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # rewards = F.normalize(rewards, dim=0)
        
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        losses['critic'].append(critic_loss)
        
        if self.update_counter % self.actor_update_freq == 0:
            actor_loss, temperature_loss = self._update_actor_and_temperature(states)
            losses['actor'].append(actor_loss)
            losses['temperature'].append(temperature_loss)
                        
        if self.update_counter % self.critic_target_update_freq == 0:
            self._update_target_network(self.double_critic_target, self.double_critic)
            
        self.update_counter += 1
        
    def train(self, env, start_episode, max_episodes, warmup_steps, gradient_steps,
              print_interval, checkpoint_interval, experiment_number,
              save_replay_buffer=False, checkpoint_training_data=False):
        
        rewards_memory, losses = self.load_training_data(experiment_number)
        
        step = 0
        for episode in range(start_episode, max_episodes):
            state, done = env.reset()
            episode_reward = 0
            while not done:
                step += 1
                
                self.to_eval_mode()
                action = self.take_action(state.to(self.device))
                self.to_train_mode()
                
                next_state, reward, done = env.step(action)
                
                self.replay_buffer.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if step > warmup_steps:
                    for _ in range(gradient_steps):
                        self.update(losses)
                                
            rewards_memory.append(episode_reward)
            
            if (episode + 1) % print_interval == 0:
                logging.info(f'Episode {episode + 1} - Reward: {episode_reward:.2f} - Alpha: {self.alpha.item():.5f} - Steps: {step + 1}')

            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode, episode_reward)
                if checkpoint_training_data:
                    self.save_training_data(rewards_memory, losses, save_replay_buffer)
                
            # Clear GPU cache after each episode
            torch.cuda.empty_cache()
                
        self.save_checkpoint(max_episodes, rewards_memory[-1])
        self.save_training_data(rewards_memory, losses, save_replay_buffer)
        
        return rewards_memory, losses
    
    def save_checkpoint(self, episode, reward, experiment_number='42'):
        if self.best_model_path is None:
            self.best_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_best_agent.pth')
        if self.last_model_path is None:
            self.last_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_last_checkpoint.pth')
        
        torch.save({
            'episode': episode,
            'reward': reward,
            'log_alpha': self.log_alpha,
            'actor_state_dict': self.actor.state_dict(),
            'double_critic_state_dict': self.double_critic.state_dict(),
            'double_critic_target_state_dict': self.double_critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'double_critic_optimizer_state_dict': self.double_critic_optimizer.state_dict(),
            'temperature_optimizer_state_dict': self.temperature_optimizer.state_dict(),
        }, self.last_model_path)
        
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save({
                'episode': episode,
                'reward': reward,
                'log_alpha': self.log_alpha,
                'actor_state_dict': self.actor.state_dict(),
                'double_critic_state_dict': self.double_critic.state_dict(),
            }, self.best_model_path)
        
    def load_checkpoint(self, best=False, experiment_number='42'):
        self.best_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_best_agent.pth')
        self.last_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_last_checkpoint.pth')
        path = self.best_model_path if best else self.last_model_path
        
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No checkpoint found at {path}")

            checkpoint = torch.load(path)    
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.double_critic.load_state_dict(checkpoint['double_critic_state_dict'])
            self.log_alpha = copy.deepcopy(checkpoint['log_alpha']).to(self.device)
            self.log_alpha.requires_grad = True
            reward = checkpoint['reward']
            
            logging.info(f"Loaded model with reward: {reward:.2f} and alpha: {self.alpha.item():.5f}")
            
            if not best:
                self.double_critic_target.load_state_dict(checkpoint['double_critic_state_dict'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.double_critic_optimizer.load_state_dict(checkpoint['double_critic_optimizer_state_dict'])
                self.temperature_optimizer.load_state_dict(checkpoint['temperature_optimizer_state_dict'])
                logging.info(f"Loaded last checkpoint from episode {checkpoint['episode'] + 1}.")
                return checkpoint['episode'] + 1
            else:
                logging.info("Loaded best agent checkpoint.")
                return None
            
        except FileNotFoundError:
            logging.info("No pre-trained model found, training from scratch.\n")
            return 0
