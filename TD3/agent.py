import os
import sys
from pathlib import Path
import torch
import pickle
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import ReplayBuffer
from src.networks import BaseActorNetwork, DoubleQCriticNetwork, mlp, cnn

class ActorNetwork(BaseActorNetwork):
    def __init__(self, state_size, action_size, fc_hidden_units=None, conv_layers=None, 
                 use_batchnorm=False, dropout_rate=0.0):
        super().__init__(state_size, action_size, fc_hidden_units, conv_layers,
                         use_batchnorm, dropout_rate)

    
    def create_cnn(self, input_dim, action_size, conv_layers, fc_hidden_units, 
                use_batchnorm, dropout_rate):
        return cnn(input_channels=input_dim,
                output_dim=action_size,
                conv_layers=conv_layers,
                fc_hidden_units=fc_hidden_units,
                use_batchnorm=use_batchnorm,
                dropout_rate=dropout_rate, 
                output_activation=nn.Tanh())
    
    def create_mlp(self, input_dim, action_size, fc_hidden_units, dropout_rate):
        return mlp(input_dim=input_dim,
                output_dim=action_size,
                hidden_units=fc_hidden_units, 
                dropout_rate=dropout_rate,
                output_activation=nn.Tanh())

    def forward(self, state):
        state = state.unsqueeze(0) if state.dim() == 3 and self.image_input else state
        return self.actor(state)


class TD3Agent:
    def __init__(self, 
                 state_size,
                 action_size,
                 hidden_units=None,
                 conv_layers=None,
                 use_batchnorm=False,
                 dropout_rate=0.0,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 exploration_noise=0.1,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 replay_buffer_size=int(1e5),
                 batch_size=128,
                 device=torch.device('cpu'),
                 data_type=torch.float32,
                 checkpoint_dir='./checkpoints'):
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = ActorNetwork(state_size=state_size,
                                  action_size=action_size,
                                  fc_hidden_units=hidden_units,
                                  conv_layers=conv_layers, 
                                  use_batchnorm=use_batchnorm,
                                  dropout_rate=dropout_rate).to(self.device, data_type)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.double_critic = DoubleQCriticNetwork(state_size=state_size,
                                                  action_size=action_size,
                                                  fc_hidden_units=hidden_units,
                                                  conv_layers=conv_layers, 
                                                  use_batchnorm=use_batchnorm,
                                                  dropout_rate=dropout_rate).to(self.device, data_type)
        self.double_critic_target = copy.deepcopy(self.double_critic).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.double_critic_optimizer = optim.Adam(self.double_critic.parameters(), lr=critic_lr)

        # Freeze target networks with respect to optimizers to avoid unnecessary computations
        for param in self.actor_target.parameters():
            param.requires_grad = False
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
        self.evaluation_rewards_file = None
        
        self.to_train_mode()
        self.update_counter = 0
        self.steps = 0

    def to_train_mode(self):
        self.actor.train()
        self.actor_target.train()

    def to_eval_mode(self):
        self.actor.eval()
        
    def to_cpu(self):
        self.actor.to('cpu')
    
    def take_action(self, state, noise=0.0):
        action = self.actor(state).detach().cpu().numpy()
        if noise > 0: # Action space is normalized to [-1, 1]
            action += noise * np.random.randn(self.action_size)
        return np.clip(action, -1, 1) 

    def save_training_data(self, rewards_memory, losses, evaluation_rewards, save_replay_buffer=True):
        with open(str(self.rewards_file), 'wb') as f:
            pickle.dump(rewards_memory, f)
        with open(str(self.losses_file), 'wb') as f:
            pickle.dump(losses, f)
        with open(str(self.evaluation_rewards_file), 'wb') as f:
            pickle.dump(evaluation_rewards, f)
        
        if save_replay_buffer:
            with open(str(self.replay_buffer_file), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
    
    def load_training_data(self, experiment_number):
        self.rewards_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_rewards.pkl')
        self.losses_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_losses.pkl')
        self.replay_buffer_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_replay_buffer.pkl')
        self.evaluation_rewards_file = os.path.join(self.checkpoint_dir, f'{experiment_number}_evaluation_rewards.pkl')
        
        if os.path.exists(self.rewards_file):
            with open(self.rewards_file, 'rb') as f:
                rewards_memory = pickle.load(f)
        else:
            rewards_memory = []
        if os.path.exists(self.losses_file):
            with open(self.losses_file, 'rb') as f:
                losses = pickle.load(f)
        else:
            losses = {'critic': [], 'actor': []}
        if os.path.exists(self.replay_buffer_file):
            with open(self.replay_buffer_file, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        if os.path.exists(self.evaluation_rewards_file):
            with open(self.evaluation_rewards_file, 'rb') as f:
                evaluation_rewards = pickle.load(f)
        else:
            evaluation_rewards = []        
        
        return rewards_memory, losses, evaluation_rewards

    def _update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = self.policy_noise * torch.randn(next_actions.shape).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_actions = (next_actions + noise).clamp(-1, 1)
            
            target_q1, target_q2 = self.double_critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (~dones) * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.double_critic(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.double_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.double_critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states):
        actor_loss = -self.double_critic(states, self.actor(states))[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, losses):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # rewards = F.normalize(rewards, dim=0)
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        losses['critic'].append(critic_loss)

        if self.update_counter % self.policy_delay == 0:
            actor_loss = self._update_actor(states)
            losses['actor'].append(actor_loss)
            self._update_target_network(self.actor_target, self.actor)
            self._update_target_network(self.double_critic_target, self.double_critic)

        self.update_counter += 1
    
    def evaluate(self, env, episodes=10, verbose=False):
        rewards = []
        seed = np.random.randint(0, 1000)
        self.to_eval_mode()
        with torch.no_grad():
            for _ in range(episodes):
                state, done = env.reset(seed)
                episode_reward = 0
                while not done:
                    action = self.take_action(state.to(self.device))
                    state, reward, done = env.step(action)
                    episode_reward += reward
                rewards.append(episode_reward)
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            if verbose:
                logging.info(f"Reward over {episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
            return avg_reward
    
    def train(self, env, start_episode, max_episodes, warmup_steps, gradient_steps,
              print_interval, checkpoint_interval, experiment_number,
              save_replay_buffer=False, checkpoint_training_data=False, 
              evaluation_interval=10): #doubles the total number of episodes
        
        rewards_memory, losses, evaluation_rewards = self.load_training_data(experiment_number)
        
        for episode in range(start_episode, max_episodes):
            state, done = env.reset()
            episode_reward = 0
            while not done:
                self.steps += 1
                
                self.to_eval_mode()
                action = self.take_action(state.to(self.device), noise=self.exploration_noise)
                self.to_train_mode()
                
                next_state, reward, done = env.step(action)
                
                self.replay_buffer.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if self.steps and len(self.replay_buffer) > warmup_steps:
                    for _ in range(gradient_steps):
                        self.update(losses)
            
            if episode % evaluation_interval == 0:
                evaluation_rewards.append(self.evaluate(env))

            rewards_memory.append(episode_reward)
            
            if (episode + 1) % print_interval == 0:
                logging.info(f'Episode {episode + 1} - Reward: {episode_reward:.2f} - Steps: {self.steps + 1}')

            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode, episode_reward, self.steps)
                if checkpoint_training_data:
                    self.save_training_data(rewards_memory, losses, evaluation_rewards, save_replay_buffer)

            torch.cuda.empty_cache()

        self.save_checkpoint(max_episodes, rewards_memory[-1], self.steps)
        self.save_training_data(rewards_memory, losses, evaluation_rewards, save_replay_buffer)
        
        return rewards_memory, losses, evaluation_rewards

    def save_checkpoint(self, episode, reward, steps, experiment_number='42'):
        if self.best_model_path is None:
            self.best_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_best_agent.pth')
        if self.last_model_path is None:
            self.last_model_path = os.path.join(self.checkpoint_dir, f'{experiment_number}_last_checkpoint.pth')
        
        torch.save({
            'episode': episode,
            'steps': steps,
            'reward': reward,
            'actor_state_dict': self.actor.state_dict(),
            'double_critic_state_dict': self.double_critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'double_critic_target_state_dict': self.double_critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'double_critic_optimizer_state_dict': self.double_critic_optimizer.state_dict(),
        }, self.last_model_path)
        
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save({
                'episode': episode,
                'reward': reward,
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
            reward = checkpoint['reward']
            
            logging.info(f"Loaded model with reward: {reward:.2f}")
            
            if not best:
                self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                self.double_critic_target.load_state_dict(checkpoint['double_critic_target_state_dict'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.double_critic_optimizer.load_state_dict(checkpoint['double_critic_optimizer_state_dict'])
                self.steps = checkpoint['steps']
                logging.info(f"Loaded last checkpoint from episode {checkpoint['episode'] + 1}.")
                return checkpoint['episode'] + 1
            else:
                logging.info("Loaded best agent checkpoint.")
                return None
            
        except FileNotFoundError:
            logging.info("No pre-trained model found, training from scratch.\n")
            return 0