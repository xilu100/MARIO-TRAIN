import csv
import os
import time
from datetime import datetime

import gym_super_mario_bros
import intel_extension_for_pytorch as ipex
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_wrapper import SkipFrameWrapper


def main():
    print(ipex.__version__)

    checkpoint_path = 'mario_model_xpu.zip'
    log_csv_path = 'training_log.csv'
    vec_env = SubprocVecEnv([set_env for _ in range(4)])

    # PPO 参数
    ppo_params = {
        'learning_rate': 1e-4,
        'n_steps': 1024,
        'batch_size': 128,
        'gamma': 0.99,
        'clip_range': 0.2,
        'ent_coef': 1e-4,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'clip_range_vf': 0.2,
        'verbose': 1,
        'tensorboard_log': 'mario_train_xpu_logs',
        'device': 'xpu'
    }


    user_input = ''
    while user_input not in ['continue', 'restart']:
        user_input = input(
            "Do you want to 'continue' training from last checkpoint \nor 'restart' training? \nPlease type 'continue' or 'restart': ").strip().lower()

    if user_input == 'continue' and os.path.exists(checkpoint_path):
        print("Loading existing model...")
        model = PPO.load(checkpoint_path, env=vec_env, device='xpu')
        log_training_event(log_csv_path, "Continue", training_step=model.num_timesteps)
    else:
        print("Creating a new model...")
        model = PPO('CnnPolicy', vec_env, **ppo_params)
        log_training_event(log_csv_path, "New", training_step=0)

    # 训练过程
    try:
        model.learn(total_timesteps=10000000,
                    callback=CheckpointCallback(checkpoint_path, log_csv_path, save_interval_seconds=600))
    except KeyboardInterrupt:
        print("Training paused, saving current state...")
        model.save(checkpoint_path)
        log_training_event(log_csv_path, "Interrupted", training_step=model.num_timesteps)

    print("Training complete or interrupted.")
    model.save('mario_model_xpu_final')
    log_training_event(log_csv_path, "Final Saved", training_step=model.num_timesteps)


def set_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = SkipFrameWrapper(env, skip=8)
    env = ResizeObservation(env, shape=(84, 84))
    env = Monitor(env)
    return env


class CheckpointCallback:

    def __init__(self, checkpoint_path, log_csv_path, save_interval_seconds=600):
        self.checkpoint_path = checkpoint_path
        self.log_csv_path = log_csv_path
        self.save_interval = save_interval_seconds
        self.last_save_time = time.time()
        self.save_index = 0

    def __call__(self, locals_, globals_):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            print("Saving model checkpoint (time-based)...")
            locals_['self'].save(self.checkpoint_path)
            self.last_save_time = current_time

            self.save_index += 1
            num_timesteps = locals_['self'].num_timesteps if 'self' in locals_ else None

            log_training_event(
                self.log_csv_path,
                "Checkpoint Saved (Time)",
                save_index=self.save_index,
                training_step=num_timesteps
            )
        return True


def log_training_event(log_csv_path, event_type, save_index=None, training_step=None):
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Timestamp', 'Event', 'Training_Step'])

    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            save_index if save_index is not None else '',
            timestamp,
            event_type,
            training_step if training_step is not None else ''
        ])


if __name__ == '__main__':
    main()
