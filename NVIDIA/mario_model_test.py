import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO.load('mario_model.zip', env=env)
#model = PPO.load('/home/lxz/Desktop/MARIO-TRAIN/Intel-ARC-SSH/mario_model_xpu_final.zip', env=env)
obs = env.reset()
for i in range(1000):
    obs = obs.copy()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
