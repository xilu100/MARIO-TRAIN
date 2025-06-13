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
    vec_env = SubprocVecEnv([set_env for _ in range(4)])

    ppo_params = {
        'learning_rate': 1e-4,
        'n_steps': 1024,
        'batch_size': 128,
        'gamma': 0.99,
        'clip_range': 0.2,
        'target_kl': 0.03,
        'ent_coef': 1e-3,  # 探索鼓励
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'clip_range_vf': 0.2,
        'verbose': 1,
        'tensorboard_log': 'mario_train_xpu_logs',
        'device': 'xpu'
    }

    model = PPO('CnnPolicy', vec_env, **ppo_params)

    model.learn(total_timesteps=30000000)
    model.save('mario_model_xpu')
    print("model saved")


def set_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = SkipFrameWrapper(env, skip=8)
    env = ResizeObservation(env, shape=(84, 84))
    env = Monitor(env)
    return env


if __name__ == '__main__':
    main()
