import logging
import os
from pathlib import Path
from collections import deque
import imageio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
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

def cumulative_mean(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)

sns.set(style="whitegrid")

def plot_results(rewards, losses, eval_rewards, evaluation_interval=10, window_size=15, save_fig=False, path='results', name='results.png', algorithm='SAC'):
    
    # Dynamically determine the base directory based on script location
    parent_directory = Path(__file__).resolve().parent.parent
    path = os.path.join(parent_directory, algorithm, path)  
    os.makedirs(path, exist_ok=True)
    
    rewards = np.array(rewards)
    eval_rewards = np.array(eval_rewards)
    critic_loss = np.array(losses['critic'])
    actor_loss = np.array(losses['actor'])
    size = 0
    temperature_loss = None
    if losses.get('temperature') is not None:
        temperature_loss = np.array(losses['temperature'])
        size = 5
    else:
        size = 4
    
    rewards_rm = running_mean(rewards, window_size)
    rewards_cm = cumulative_mean(rewards)
    
    eval_rewards_rm = running_mean(eval_rewards, window_size//2)
    eval_rewards_cm = cumulative_mean(eval_rewards//2)
    
    critic_loss_rm = running_mean(critic_loss, window_size)
    critic_loss_cm = cumulative_mean(critic_loss)
    
    actor_loss_rm = running_mean(actor_loss, window_size)
    actor_loss_cm = cumulative_mean(actor_loss)
    
    temperature_loss_rm = None
    temperature_loss_cm = None
    if temperature_loss is not None:
        temperature_loss_rm = running_mean(temperature_loss, window_size)
        temperature_loss_cm = cumulative_mean(temperature_loss)
    
    fig, ax = plt.subplots(1, size, figsize=(5*size, 7))

    for a in ax:
        a.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        a.xaxis.set_major_locator(MaxNLocator(nbins=5))
        a.yaxis.set_major_locator(MaxNLocator(nbins=5))
        a.grid(True, which='both', linestyle='--', linewidth=0.6) 
        a.tick_params(axis='both', which='major', labelsize=12)  


    sns.lineplot(x=np.arange(len(rewards_rm)), y=rewards_rm, ax=ax[0], color='blue', linewidth=2, alpha=0.7)
    sns.lineplot(x=np.arange(len(rewards_cm)), y=rewards_cm, ax=ax[0], color='red', linewidth=2)
    ax[0].set_title('Training Rewards', fontsize=14)
    ax[0].set_xlabel('Episodes', fontsize=12)
    ax[0].set_ylabel('Reward', fontsize=12)
    
    sns.lineplot(x=evaluation_interval*np.arange(len(eval_rewards_rm)), y=eval_rewards_rm, ax=ax[1], color='blue', linewidth=2, alpha=0.7)
    sns.lineplot(x=evaluation_interval*np.arange(len(eval_rewards_cm)), y=eval_rewards_cm, ax=ax[1], color='red', linewidth=2)
    ax[1].set_title('Evaluation Rewards', fontsize=14)
    ax[1].set_xlabel('Episodes', fontsize=12)
    ax[1].set_ylabel('Reward', fontsize=12)
    
    sns.lineplot(x=np.arange(len(critic_loss_rm)), y=critic_loss_rm, ax=ax[2], color='blue', label='Running Mean', linewidth=2, alpha=0.7)
    sns.lineplot(x=np.arange(len(critic_loss_cm)), y=critic_loss_cm, ax=ax[2], color='red', label='Cumulative Mean', linewidth=2)
    ax[2].set_title('Critic Loss', fontsize=14)
    ax[2].set_xlabel('Steps', fontsize=12)
    ax[2].set_ylabel('Loss', fontsize=12)
    ax[2].legend(loc='upper right', fontsize=12)

    sns.lineplot(x=np.arange(len(actor_loss_rm)), y=actor_loss_rm, ax=ax[3], color='blue', linewidth=2, alpha=0.7)
    sns.lineplot(x=np.arange(len(actor_loss_cm)), y=actor_loss_cm, ax=ax[3], color='red', linewidth=2)
    ax[3].set_title('Actor Loss', fontsize=14)
    ax[3].set_xlabel('Steps', fontsize=12)
    ax[3].set_ylabel('Loss', fontsize=12)
    
    if temperature_loss_rm is not None and temperature_loss_cm is not None:
        sns.lineplot(x=np.arange(len(temperature_loss_rm)), y=temperature_loss_rm, ax=ax[4], color='blue', linewidth=2, alpha=0.7)
        sns.lineplot(x=np.arange(len(temperature_loss_cm)), y=temperature_loss_cm, ax=ax[4], color='red', linewidth=2)
        ax[4].set_title('Temperature Loss', fontsize=14)
        ax[4].set_xlabel('Steps', fontsize=12)
        ax[4].set_ylabel('Loss', fontsize=12)

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