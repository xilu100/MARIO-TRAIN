import gym_super_mario_bros
import intel_extension_for_pytorch as ipex
from gym.wrappers import GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from custom_wrapper import SkipFrameWrapper


def main():
    print(ipex.__version__)
    vec_env = SubprocVecEnv([set_env for _ in range(4)])
    model = PPO('CnnPolicy', vec_env, verbose=1, tensorboard_log='mario_train_xpu_logs', device="xpu")
    model.learn(total_timesteps=1000000)
    model.save('mario_model_xpu')


def set_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = SkipFrameWrapper(env, skip=8)
    env = ResizeObservation(env, shape=(84, 84))

    return env


if __name__ == '__main__':
    main()
