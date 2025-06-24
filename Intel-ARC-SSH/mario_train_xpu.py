import csv
import signal
from datetime import datetime

import gym_super_mario_bros
import intel_extension_for_pytorch as ipex
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from custom_wrapper import SkipFrameWrapper

# Global variables to track progress
current_timesteps = 0
total_timesteps = 1000000
ppo_params = {}


def main():
    global ppo_params, current_timesteps, total_timesteps

    print(ipex.__version__)

    # Environment setup
    vec_env = SubprocVecEnv([set_env for _ in range(4)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')

    # PPO parameters
    ppo_params = {
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 512,
        'ent_coef': 0.1,
        'n_epochs': 10,
        'target_kl': 0.1,
        'verbose': 1,
        'tensorboard_log': 'mario_train_xpu_logs',
        'device': 'xpu'
    }

    # Save initial record
    data = {
        'time': datetime.now().isoformat(),
        'status': 'training started',
        'current_timesteps': 0,
        'total_timesteps': total_timesteps,
        **ppo_params
    }
    save_to_csv(data)

    # Signal handler for interruptions
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize and train the model
    model = PPO('CnnPolicy', vec_env, **ppo_params)

    def progress_callback(local_timesteps, total_timesteps_):
        global current_timesteps
        current_timesteps = local_timesteps

    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    model.save('mario_model_xpu')

    # Save final record
    data = {
        'time': datetime.now().isoformat(),
        'status': 'training completed',
        'current_timesteps': current_timesteps,
        'total_timesteps': total_timesteps,
        **ppo_params
    }
    save_to_csv(data)


def set_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = SkipFrameWrapper(env, skip=8)
    env = ResizeObservation(env, shape=(84, 84))
    env = Monitor(env)
    return env


def save_to_csv(data, file_path='training_log.csv'):
    # Check if the file exists
    file_exists = False
    try:
        with open(file_path, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    # Write data to the CSV file
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()  # Write header if file doesn't exist
        writer.writerow(data)


def signal_handler(sig, frame):
    global ppo_params, current_timesteps, total_timesteps
    data = {
        'time': datetime.now().isoformat(),
        'status': 'interrupted',
        'current_timesteps': current_timesteps,
        'total_timesteps': total_timesteps,
        **ppo_params
    }
    save_to_csv(data)
    print("Training interrupted. Progress saved.")
    exit(0)


if __name__ == '__main__':
    main()
