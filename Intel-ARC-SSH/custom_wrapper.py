import gym

class SkipFrameWrapper(gym.Wrapper):
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