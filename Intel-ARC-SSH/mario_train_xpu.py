import gym_super_mario_bros
import intel_extension_for_pytorch as ipex
import torch
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log='mario_train_xpu_logs', device=device)
model.learn(total_timesteps=100000)
model.save('mario_model_xpu')
