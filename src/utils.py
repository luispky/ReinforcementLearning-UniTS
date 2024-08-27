import logging
import os
from pathlib import Path
from collections import deque
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
import torch
import random

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, data_type, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0 
        self.device = device
        self.dtype = data_type
        
        np_dtype = np.float32 if data_type == torch.float32 else np.float64
        
        self.states = np.zeros((max_size, *state_dim), dtype=np_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=np_dtype)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np_dtype)
        self.rewards = np.zeros((max_size, 1), dtype=np_dtype)
        self.dones = np.zeros((max_size, 1), dtype=np.bool_)

    def __len__(self):
        return self.size
    
    def remember(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.states[indices], dtype=self.dtype).to(self.device),
            torch.as_tensor(self.actions[indices], dtype=self.dtype).to(self.device),
            torch.as_tensor(self.rewards[indices], dtype=self.dtype).to(self.device),
            torch.as_tensor(self.next_states[indices], dtype=self.dtype).to(self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.bool).to(self.device)
        )

def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def running_std(x, window_size):
    std = []
    for i in range(len(x) - window_size + 1):
        std.append(np.std(x[i:i+window_size]))
    return np.array(std)

def plot_results(rewards, losses, window_size=8, save_fig=False, path='results', name='results.png', algorithm='SAC'):
    
    # Dynamically determine the base directory based on script location
    parent_directory = Path(__file__).resolve().parent.parent
    path = os.path.join(parent_directory, algorithm, path)  
    os.makedirs(path, exist_ok=True)
    
    rewards = np.array(rewards)
    critic_loss = np.array(losses['critic'])
    actor_loss = np.array(losses['actor'])
    size = 0
    temperature_loss = None
    if losses.get('temperature') is not None:
        temperature_loss = np.array(losses['temperature'])
        size = 4
    else:
        size = 3
    
    # Calculate smoothed data
    rewards_rm = running_mean(rewards, window_size)
    rewards_std = running_std(rewards, window_size)
    
    # Plotting
    fig, ax = plt.subplots(1, size, figsize=(5*size, 7))

    # Plot rewards
    # ax[0].plot(rewards, color='blue', alpha=0.3)
    ax[0].plot(rewards_rm, color='blue', label='Average Return')
    ax[0].fill_between(range(len(rewards_rm)), rewards_rm - rewards_std, rewards_rm + rewards_std, color='blue', alpha=0.3)
    ax[0].set_title('Rewards')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    ax[0].legend(loc='lower left')
    
    # Plot critic loss
    ax[1].plot(critic_loss, color='red')
    ax[1].set_title('Critic Loss')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Loss')

    # Plot actor loss
    ax[2].plot(actor_loss, color='red')
    ax[2].set_title('Actor Loss')
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel('Loss')
    
    if size == 4: 
        # Plot temperature loss
        ax[3].plot(temperature_loss, color='red')
        ax[3].set_title('Temperature Loss')
        ax[3].set_xlabel('Steps')
        ax[3].set_ylabel('Loss')

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(os.path.join(path, f'{name}_results.png'))
        
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(secs)}s"
    else:
        return f"{int(secs)}s"

def test_policy(agent, env, num_episodes):
    episode_total_rewards = np.zeros(num_episodes)
    seed = np.random.randint(0, 1000)
    agent.to_cpu()
    agent.to_eval_mode()
    with torch.no_grad():
        for i in range(num_episodes):
            state, done = env.reset(seed)
            total_reward = 0
            while not done:
                action = agent.take_action(state)
                state, reward, done = env.step(action)
                total_reward += reward
            episode_total_rewards[i] = total_reward
        avg_reward = np.mean(episode_total_rewards)
        std_reward = np.std(episode_total_rewards)
        
    logging.info(f"Reward over {num_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")

def policy_visualization(agent, env, save_gif=False, save_video=False, path='results', name=None, algorithm='SAC'):
    '''Visualize the policy learned by the agent and save as either a GIF or an MP4 video.'''
    
    # Dynamically determine the base directory based on script location
    parent_directory = Path(__file__).resolve().parent.parent
    path = os.path.join(parent_directory, algorithm, path)  
    os.makedirs(path, exist_ok=True)
    
    seed = np.random.randint(0, 1000)
    state, done = env.reset(seed)
    total_reward = 0
    frames = []
    agent.to_cpu()
    agent.to_eval_mode()
    with torch.no_grad():
        while not done:
            frame = env.render()
            
            # Ensure the frame size is divisible by 16 for MP4 video
            if save_video:
                h, w, _ = frame.shape
                new_h = (h + 15) // 16 * 16
                new_w = (w + 15) // 16 * 16
                frame_resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                frame_resized[:h, :w, :] = frame
                frames.append(frame_resized)
            else:
                frames.append(frame)
            
            action = agent.take_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        
    env.close()
    logging.info(f"Total Reward of Policy Evaluation: {total_reward:.2f}")

    if save_gif:
        gif_path = os.path.join(path, f'{name}_policy.gif')
        imageio.mimsave(gif_path, frames, fps=30)
    
    if save_video:
        video_path = os.path.join(path, f'{name}.mp4')
        
        # Use 'ffmpeg' plugin for MP4 video creation
        writer = imageio.get_writer(video_path, fps=30, codec='libx264')
        
        for frame in frames:
            writer.append_data(frame)
        
        writer.close()

def load_config(env_id, algorithm, config_files):
    """Load the configuration based on the environment ID and algorithm."""
    config_file = config_files.get(env_id)
    if not config_file:
        raise ValueError(f"Unknown environment ID '{env_id}'. Valid options are: {', '.join(config_files.keys())}.")
    
    parent_directory = Path(__file__).resolve().parent.parent
    # Construct the path to the config file based on the algorithm
    config_path = os.path.join(parent_directory, algorithm, 'configs', config_file)
    
    # Read the configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)