# Train-Logs

## June,12, 2025 : The environment needs to be pre-processed

- [x] Grayscale processing

```python
from gym.wrappers import GrayScaleObservation
```

- [x] Frame Skipping
- [custom_wrapper.py](Intel-ARC-SSH/custom_wrapper.py)

```python
import gym


class CustomWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self.skip = skip

    def skip_step(self, action):
        obs, total_reward, done, info = None, 0, False, None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
```

- [x] Resize the frame (data)  
  Using downsampling technology

```python
from gym.wrappers import ResizeObservation
```

## June,12, 2025 : Adding multithreading

```python
vec_env = SubprocVecEnv([set_env for _ in range(4)])
```

## June,25, 2025 : Add VecFrameStack with 4 frames and channels_order='last'.

```python
vec_env = VecFrameStack(vec_env, 4, channels_order='last')
```

## June,25, 2025 : Added hyperparameter logging.
```python
data = {
        'time': datetime.now().isoformat(),
        'status': 'interrupted',
        'current_timesteps': current_timesteps,
        'total_timesteps': total_timesteps,
        **ppo_params
    }
```