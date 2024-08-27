import sys
import os
import logging
import argparse
import torch
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from SAC.agent import SACAgent
from TD3.agent import TD3Agent
from src.environment import Environment
from src.utils import policy_visualization, plot_results, format_time, load_config, test_policy, set_seed

# Set device and data type based on availability of GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLOAT_DTYPE = torch.float32 if DEVICE.type == 'cuda' else torch.float64

CONFIG_FILES = {
    'pendulum': 'config_Pendulum.yaml',
    'walker': 'config_BipedalWalker.yaml',
    'halfcheetah': 'config_HalfCheetah.yaml',
    'humanoid': 'config_Humanoid.yaml',
    'ant': 'config_Ant.yaml',
    'walker2d': 'config_Walker2d.yaml',
    'car': 'config_CarRacing.yaml'
}

def run(config, algorithm):
    # Configuration variables
    max_episodes = config['train']['max_episodes']
    env_name = config['environment']['env_name']
    hardcore = config['environment'].get('hardcore', False)
    load_best = config['agent'].get('load_best', False)
    experiment_number = config['experiment_number']
    checkpoint_dir = f"checkpoints/{env_name + '_hardcore' if hardcore else env_name}"
    results_dir = f"results/{env_name + '_hardcore' if hardcore else env_name}"
    seed=config['environment'].get('seed', None)
    
    parent_directory = Path(__file__).resolve().parent.parent
    checkpoint_dir = os.path.join(parent_directory, algorithm, checkpoint_dir)
    
    # Set seeds
    set_seed(seed)
    
    # Initialize environment
    env = Environment(
        env_name=env_name,
        data_type=FLOAT_DTYPE,
        hardcore=hardcore, 
        image_processing=config['environment'].get('image_processing', False),
        image_size=config['environment'].get('image_size', 84),
        stack_frames=config['environment'].get('stack_frames', 4),
        seed=seed
    )
    
    # Initialize agent based on the chosen algorithm
    if algorithm == 'SAC':
        agent = SACAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            hidden_units=config['agent']['hidden_units'],
            conv_layers=config['agent']['conv_layers'],
            log_std_interval=config['agent']['log_std_interval'],
            actor_lr=config['agent']['actor_lr'],
            critic_lr=config['agent']['critic_lr'],
            temperature_lr=config['agent']['temperature_lr'],
            gamma=config['agent']['gamma'],
            tau=config['agent']['tau'],
            init_temperature=config['agent']['init_temperature'],
            actor_update_freq=config['agent']['actor_update_freq'],
            critic_target_update_freq=config['agent']['critic_target_update_freq'],
            replay_buffer_size=config['agent']['replay_buffer_size'],
            batch_size=config['agent']['batch_size'],
            checkpoint_dir=checkpoint_dir,
            device=DEVICE,
            data_type=FLOAT_DTYPE,
        )
    elif algorithm == 'TD3':
        agent = TD3Agent(
            state_size=env.state_size,
            action_size=env.action_size,
            hidden_units=config['agent']['hidden_units'],
            conv_layers=config['agent']['conv_layers'],
            actor_lr=config['agent']['actor_lr'],
            critic_lr=config['agent']['critic_lr'],
            gamma=config['agent']['gamma'],
            tau=config['agent']['tau'],
            exploration_noise=config['agent']['exploration_noise'],
            policy_noise=config['agent']['policy_noise'],
            noise_clip=config['agent']['noise_clip'],
            policy_delay=config['agent']['policy_delay'],
            replay_buffer_size=config['agent']['replay_buffer_size'],
            batch_size=config['agent']['batch_size'],
            checkpoint_dir=checkpoint_dir,
            device=DEVICE,
            data_type=FLOAT_DTYPE,
        )
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    # Load model checkpoint
    start_episode = agent.load_checkpoint(best=load_best, experiment_number=experiment_number)

    # Train the agent if not fully trained or if loading the best model
    if start_episode is not None and start_episode < max_episodes:
        start_time = time.time()

        rewards, losses = agent.train(
                                    env=env,
                                    start_episode=start_episode,
                                    max_episodes=max_episodes,
                                    warmup_steps=config['train']['warmup_steps'],
                                    gradient_steps=config['train']['gradient_steps'],
                                    print_interval=config['train']['print_interval'],
                                    checkpoint_interval=config['train']['checkpoint_interval'],
                                    experiment_number=experiment_number,
                                )

        end_time = time.time()
        logging.info(f"Training time: {format_time(end_time - start_time)}")

        # Save and plot results
        plot_results(rewards, losses, save_fig=True, name=experiment_number, path=results_dir, algorithm=algorithm)
    
    # Test and visualize policy
    test_policy(agent=agent, env=env, num_episodes=config['train']['test_episodes'])
    policy_visualization(agent=agent, env=env, save_gif=True, path=results_dir, name=f"{experiment_number}_{'best' if load_best else 'last'}", algorithm=algorithm)

    if not load_best:
        agent.load_checkpoint(best=True, experiment_number=experiment_number)
        policy_visualization(agent=agent, env=env, save_gif=True, path=results_dir, name=f"{experiment_number}_best", algorithm=algorithm)

def main():
    parser = argparse.ArgumentParser(description="Train an agent using Deep Actor-Critic algorithms.")
    parser.add_argument('-env', '--environment-id', type=str, required=True, choices=CONFIG_FILES.keys(), help="The environment ID to run the algorithm on.")
    parser.add_argument('-alg', '--algorithm', type=str, required=True, choices=['sac', 'td3'], help="The algorithm to run.")
    args = parser.parse_args()
    algorithm = args.algorithm.upper()

    # Load configuration
    config = load_config(args.environment_id, algorithm, CONFIG_FILES)
    
    env_name = config['environment']['env_name']
    experiment_number = config['experiment_number']
    
    # Setup logging
    log_dir = f"logs/{env_name + '_hardcore' if config['environment'].get('hardcore', False) else env_name}"
    parent_directory = Path(__file__).resolve().parent.parent
    log_dir = os.path.join(parent_directory, algorithm, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{experiment_number}_training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log configuration if starting fresh
    log_file = os.path.join(log_dir, f"{experiment_number}_training.log")
    if not config['agent'].get('load_best', False) and os.path.getsize(log_file) == 0:
        logging.info(f"{algorithm} Agent Configuration:")
        for key, value in config['agent'].items():
            logging.info(f"{key}: {value}")
        logging.info("\nTrain Configuration:")
        for key, value in config['train'].items():
            logging.info(f"{key}: {value}")
        logging.info(f"\nEnvironment Seed: {config['environment'].get('seed', None)}")
        
        logging.info(f"\nStarting {algorithm} Training\n")

    run(config, algorithm)

if __name__ == '__main__':
    main()
