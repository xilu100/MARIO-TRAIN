import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log='mario_train_logs')
model.learn(total_timesteps=1000)
model.save('mario_model')
