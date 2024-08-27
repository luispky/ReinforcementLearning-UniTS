import numpy as np
import gymnasium as gym
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.frame_stack import FrameStack
import torch

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation, dtype=np.float32) / 255.0

class Environment:
    '''Wrapper class around a gymnasium environment for convenience.'''
    def __init__(self, 
                 env_name, 
                 render_mode='rgb_array', 
                 seed=None, 
                 data_type=torch.float32, 
                 hardcore=False, 
                 image_processing=False,  # New flag to enable image processing
                 image_size=84,           # Default image size for resizing
                 stack_frames=4):          # Default number of frames to stack
        self.env_name = env_name
        env = None
        if 'BipedalWalker' in env_name:
            env = gym.make(env_name, render_mode=render_mode, hardcore=hardcore)
        else:
            env = gym.make(env_name, render_mode=render_mode)
        
        # Apply RescaleAction only for continuous action spaces
        if isinstance(env.action_space, gym.spaces.Box):
            self.env = RescaleAction(env, min_action=-1, max_action=1)
        else:
            self.env = env

        self.image_processing = image_processing
        self._dtype = data_type
        
        if self.image_processing:
            # Apply image preprocessing wrappers
            self.env = GrayScaleObservation(self.env)
            self.env = ResizeObservation(self.env, image_size)
            self.env = FrameStack(self.env, stack_frames)
            self.env = NormalizeObservation(self.env)
        
        obs_space = self.env.observation_space
        
        if isinstance(obs_space, gym.spaces.Box):
            self.state_size = obs_space.shape
        else:
            raise NotImplementedError("Unsupported observation space type.")
        
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Box):
            self.action_size = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_size = action_space.n
        else:
            raise NotImplementedError("Action space is not Box or Discrete. Behavior not implemented.")

        self._seed = seed
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        
    def reset(self, seed=None):
        if 'Car' in self.env_name:
            # Iterate the first 50 steps to get a good starting point
            state = self.env.reset(seed=seed)[0] if seed is not None else self.env.reset(seed=self._seed)[0]
            for _ in range(50):
                state, _, _, _, _ = self.env.step([0.0, 0.0, 0.0])
            return self._to_tensor(state), False
        
        state = self.env.reset(seed=seed)[0] if seed is not None else self.env.reset(seed=self._seed)[0]
        return self._to_tensor(state), False
    
    def step(self, action):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action = int(action)  # Ensure action is an integer for discrete spaces
        state, reward, terminated, truncated, _ = self.env.step(action)
        return self._to_tensor(state), reward, terminated or truncated

    def get_random_action(self):
        return self.env.action_space.sample()
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
    
    def _to_tensor(self, state):
        """Converts the state to a tensor."""
        return torch.tensor(state, dtype=self._dtype)
