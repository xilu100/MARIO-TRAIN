# MARIO-TRAIN

This is a project for research on Super-Mario-Bros reinforcement learning.  
The purpose and research focus of this project is on RL (`Reinforcement Learning`) for completing levels in
Super Mario (tentative).  
Using `NVIDIA` GPUs for minimal training/testing, with `Intel ARC` GPUs as the primary training setup.  
The project(temporary) originates from this author. Thanks to the author for open-sourcing it.     
URL:https://github.com/Kautenja/gym-super-mario-bros  
The environment setup for this project(temporary) is based on the quick setup guide provided by the author below. Thanks
to the author.  
URL:https://github.com/jusway/RL_SuperMario  
In the future, it may be necessary to update and reconfigure the entire environment based on newer versions of the
reinforcement learning modules.

## Environment-setup

**Attention**:This project is only compatible with `Windows/Linux(Ubuntu 22.04 LTS)` and works with `CPU`,`NVIDIA` GPUs
with `CUDA`, `Intel ARC`
series GPUs, and `Intel` integrated graphics.  
For compatibility reasons, NVIDIA GPUs are best run on Windows.   
(On Linux, installing NVIDIA GPU drivers may result in a black screen that prevents access to the graphical interface.
If you don't use the graphical interface, Linux is recommended,
as setting up the environment on Linux can be much more efficient.)  
To date*, `AMD` GPUs have only partially supported Windows `ROCm` with the RX 9000 series.
Therefore, I have not tested the performance of AMD GPUs on either Windows or Linux.  
`Apple Silicon` **MacOS**  cannot utilize `Metal` acceleration due to the inability to download the required version
of OpenCV correctly.  
*:June,03, 2025

### Step 1. Download Conda and create a Conda environment

URL:https://www.anaconda.com/download/success

#### Windows:

1. Click the download link/button.  
   https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe
2. Double-click to open the installation package.
3. Double-click the installation package ending with `.exe`, then click  
   `Next` -> `I Agree` -> `Just Me` -> `Next` -> `Next` -> `Add to PATH` -> `Install` -> `Finish`
4. Open the `Terminal`/`PowerShell*`/`CMD`(It is not recommended to use CMD).
5. Type
   ```shell
   conda init powershell
   ```

6. Restart the terminal.  
   Execute the version check.
   ```shell
   conda --version
   ```
   If you encounter issues similar to the ones listed below, please refer to the following link.
   (Occurs in PowerShell version 7.5.0 and above.)
   ```shell
   usage: conda-script.py [-h] [-v] [--no-plugins] [-V] COMMAND ...
   conda-script.py: error: argument COMMAND: invalid choice: ''(choose from activate, clean, commands, compare, config, create, deactivate, env, 
   export, info, init, install, list, notices, package, content-trust, doctor, repoquery, remove, uninstall, rename, run, search, update, upgrade)
   ```
   Solution:https://github.com/conda/conda/issues/14537  
   The correct output should be similar to:
   ```shell 
   conda 24.11.3
   ```
7. If all goes well, it should be in this format:  
   `(base) PS PATH\PATH\PATH>`
8. Create a Conda environment based on Python 3.8.8.
   ```shell
   conda create -n mario python=3.8.8
   ```  
   (The environment name after `-n` can be anything you like, as long as you can remember it.
   Here, we will use "mario" consistently.)
9. If all goes well, it should be in this format:  
   Input:
   ```shell
   conda activate mario
   ```  
   Output:
   ```shell
   (mario) PS PATH\PATH\PATH>
   ```
10. It is recommended to perform all the following operations within this environment;
    otherwise, compatibility issues may occur.

#### Linux/WSL2

```PASS```

### Step 2. Configure accelerated computing frameworks / hardware acceleration platforms.

#### CUDA

##### Windows

First, ensure that your NVIDIA GPU drivers are updated to the latest or a relatively recent version.

1. Check
   ```shell
   nvidia-smi
   ``` 
   to see if you get an output similar to the following:
   ```shell
   Tue Jun  3 13:35:24 2025
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 576.52                 Driver Version: 576.52         CUDA Version: 12.9     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  NVIDIA GeForce RTX 6090 ...  WDDM  |   00000000:04:00.0  On |                  N/A |
   | 35%   41C    P0             49W / 1050W |    2447MiB / 524288MiB |  0.004%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+
   ```
2. Check the maximum CUDA version supported in the top right corner.
   If the maximum supported `CUDA version` shown is less than `11.8`,
   please use a compatible GPU or accept significant compatibility risks.
3. Go to the official website to download CUDA version 11.8.  
   Windows11:  
   https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local  
   Windows10:  
   https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
4. Open the downloaded `.exe` file.
5. Click the following buttons in order:(If administrator permission is required, of course, click “Agree” or “Allow.”)
   `OK`(Wait for the all progress bar to complete.)->`AGREE AND CONTINUE`->`Express (Recommended)`->`NEXT`->
   `I Understand,...`+`NEXT`(May appear)->`NEXT`->`CLOSE`
6. Type
   ```shell
   nvcc --version
   ```
7. You will get output similar to the following:
   ```shell
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2022 NVIDIA Corporation
   Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
   Cuda compilation tools, release 11.8, V11.8.89
   Build cuda_11.8.r11.8/compiler.31833905_0
   ```
   Check whether the version, whether in Release or Build, is equal to 11.8.
8. Important: Check again whether the current Conda environment is set to `(mario)`.
   ```shell
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Wait for all installations to complete.
9. Enter the following command to check the output:
   ```shell
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```
10. You should get output similar to the following:
    ```shell
    PyTorch version: 2.4.1+cu118
    CUDA available: True
    CUDA device name: NVIDIA GeForce RTX 6090 Ti
    ```
11. Congratulations, the CUDA environment has been successfully installed!

##### Linux/WSL2

```PASS```

#### IPEX

##### Windows

```PASS```

##### Linux/WSL2

1. Configure the OneAPI environment.
   ```bash
   sudo apt-get update
   sudo apt install build-essential
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
   sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
   sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
   sudo apt-get install -y libze-dev intel-ocloc
   clinfo | grep "Device Name"
   sudo gpasswd -a ${USER} render
   newgrp render
   ```
2. If all goes well, you should get the following output:
   ```bash
    Device Name                                   Intel(R) Arc(TM) A770 Graphics
    Device Name                                   Intel(R) Arc(TM) A770 Graphics
    Device Name                                   Intel(R) Arc(TM) A770 Graphics
    Device Name                                   Intel(R) Arc(TM) A770 Graphics
    Adding user ${USER} to group render
   ```
3. Theoretically, the OneAPI installation is now complete.
4. Configure PyTorch based on IPEX.  
   Pip:
   ```bash
   python -m pip install torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
   ```
   Conda:
   ```bash
   conda install intel-extension-for-pytorch=2.3.110 pytorch=2.3.1 -c https://software.repos.intel.com/python/conda -c conda-forge
   ```
5. Enter the following code to check if the installation was successful:
   ```bash
   export OCL_ICD_VENDORS=/etc/OpenCL/vendors
   export CCL_ROOT=${CONDA_PREFIX}
   python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
   ```
6. You will get output similar to the following:
   ```bash
   export CCL_ROOT=${CONDA_PREFIX}
   python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"
   2.3.1+cxx11.abi
   2.3.110+xpu
   [0]: _XpuDeviceProperties(name='Intel(R) Arc(TM) A770 Graphics', platform_name='Intel(R) Level-Zero', type='gpu', driver_version='1.3.33276', total_memory=15473MB, max_compute_units=512, gpu_eu_count=512, gpu_subslice_count=32, max_work_group_size=1024, max_num_sub_groups=128, sub_group_sizes=[8 16 32], has_fp16=1, has_fp64=0, has_atomic64=1)
   ```
7. Congratulations, the PyTorch environment based on Intel GPUs has been successfully configured!
8. However, since Intel ARC series GPUs have not been thoroughly tested in the market, there may be uninstalled
   environment configurations and various compatibility issues. Therefore, if you do not want to spend time resolving
   compatibility or configuration issues, it is recommended to avoid using Intel GPUs.
   If you enjoy tackling challenges, or if an Intel GPU is your only option due to financial constraints, please follow
   the step-by-step guide in the link below to configure the environment.  
   GitHub : https://github.com/intel/intel-extension-for-pytorch  
   Install Guide: (Actual tests show that neither pip nor Conda can fully install PyTorch based on IPEX. Therefore, it
   is recommended to use a combination of pip and Conda modes.)  
   pip:  
   https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.3.110%2Bxpu&os=linux%2Fwsl2&package=pip  
   Conda:  
   https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.3.110%2Bxpu&os=linux%2Fwsl2&package=conda

#### Integrated Graphics

##### Windows:

```shell
pip3 install torch torchvision torchaudio
pip install torch torchvision torchaudio
```

##### Linux/WSL2:

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3. Configure `pip` libraries and dependencies.
0. Be sure to install the VS Studio 2022 tool(Desktop C/C++).
1. Downgrade the following packages:
   ```
   pip install setuptools==65.5.0
   pip install wheel==0.38.4
   pip install pip==20.2.4
   ```
2. Directly install the following dependencies:
   ```
   pip install -r requirements.txt 
   ```
   (If `requirements.txt` is not in the current folder, please enter the full path.Exp:
   `pip install -r PATH/PATH/requirements.txt`)
    - [requirements.txt](requirements.txt)
3. If no errors are reported, the pip dependencies have been successfully configured.
4. This is only a temporary environment configuration for an early version; `Gym` has now migrated to `Gymnasium`.  
   So, if you encounter any compatibility issues, try Googling or, with a bit of luck, try changing the version numbers.

---
The following section focuses on project implementation, leaning towards a log-style format.

## Test whether the Mario based on NES can run.

### June,05, 2025: Successfully configured the environment and ran the test program successfully.

- [test_run_mario.py](NVIDIA/TestEnv/test_run_mario.py)

### June,07, 2025: Prior knowledge required: Q-Learning, DQN (Deep Q-Network) , and PPO (Proximal Policy Optimization).

#### From Q-Learning To DQN (Deep Q-Network)

Worth exploring in depth.

#### PPO (Proximal Policy Optimization)

Worth exploring in depth.  
This project will temporarily focus on studying PPO.  
Reference URL(s):  
https://en.wikipedia.org/wiki/Proximal_policy_optimization  
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

##### History(The history and development process are crucial.)

The predecessor to PPO, Trust Region Policy Optimization (TRPO), was published in 2015.
It addressed the instability issue of another algorithm, the Deep Q-Network (DQN), by using the trust region method
to limit the KL divergence between the old and new policies.
However, TRPO uses the Hessian matrix (a matrix of second derivatives) to enforce the trust region,
but the Hessian is inefficient for large-scale problems.

##### Algorithm Overview

Math:  
Worth exploring in depth.  
Code(Simple Invocation):  
Fictional Example:

```python
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.atari_wrappers import AtariWrapper

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = AtariWrapper(env)
```

```python
from stable_baselines3 import PPO

model = PPO("CnnPolicy", env, verbose=1, n_steps=256, batch_size=64, n_epochs=4, learning_rate=2.5e-4, gamma=0.99)

model.learn(total_timesteps=1000000)

model.save("ppo_mario")
```

Code(Actual Implementation):  
AI-generated content; rigorous analysis will not be conducted for now—this section is for placeholder purposes only.

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

GAMMA = 0.99
EPS_CLIP = 0.2
LR = 3e-4
K_EPOCHS = 4
BATCH_SIZE = 64


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(value), dist_entropy


def collect_trajectories(env, policy, batch_size):
    states = []
    actions = []
    rewards = []
    dones = []
    logprobs = []

    state = env.reset()
    for _ in range(batch_size):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, logprob = policy.act(state_tensor)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        logprobs.append(logprob.detach())

        state = next_state if not done else env.reset()
    return states, actions, rewards, dones, logprobs


def compute_returns_advantages(rewards, dones, values, gamma=GAMMA):
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values.detach()
    return returns, advantages


def ppo_update(policy, optimizer, states, actions, old_logprobs, returns, advantages):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_logprobs = torch.stack(old_logprobs)
    returns = returns.detach()
    advantages = advantages.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(K_EPOCHS):
        logprobs, state_values, dist_entropy = policy.evaluate(states, actions)

        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - state_values).pow(
            2).mean() - 0.01 * dist_entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    max_training_steps = 10000
    batch_size = 2048
    print_interval = 1000

    timestep = 0
    while timestep < max_training_steps:
        states, actions, rewards, dones, old_logprobs = collect_trajectories(env, policy, batch_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(states)
            _, values = policy.forward(state_tensor)
            values = values.squeeze()

        returns, advantages = compute_returns_advantages(rewards, dones, values)

        ppo_update(policy, optimizer, states, actions, old_logprobs, returns, advantages)

        timestep += batch_size

        if timestep % print_interval == 0:
            total_reward = sum(rewards) / batch_size
            print(f"Step: {timestep} Avg Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()

```

### June,07, 2025: Noticing Mario’s actions.

While studying the code , I noticed that the original author added `wrappers` for three types of Mario’s `actions`.

```python
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```

- [actions_ex.py](Referenced%20file/actions_ex.py)  
  Actual path:(YOUR ENV PATH)\mario\Lib\site-packages\gym_super_mario_bros\actions.py

The original author divided them into three types: `RIGHT_ONLY`, `SIMPLE_MOVEMENT`, and `COMPLEX_MOVEMENT`,
respectively.

### June,07-10, 2025: Noticed the operating mechanism.

```python 
done = True  # Whether the environment is ended
for step in range(5000):  # Execute 5000 steps
    if done:  # If it ends
        state = env.reset()  # Environment Restart
    state, reward, done, info = env.step(
        env.action_space.sample())  # Returns four variables: status, reward, whether it is finished, and information
    env.render()  # Render into image(s)

env.close()  # Close the entire environment

```

### June, 10, 2025: Test Run Stable Baselines3 v1.6.2

```python
import gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
```

Run successfully.

### June,11, 2025: Custom Trainer and Tester

Constructed using the standard syntax of `Stable Baselines3`.  
This project adopts a train-test separation approach to facilitate parameter updates.  
Two files were created, namely:

- [mario_model_train.py](NVIDIA/mario_model_train.py)
- [mario_model_test.py](NVIDIA/mario_model_test.py)

To unify the running environment, importing the Gym Mario environment is necessary.

```python
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
```

First is the basic definition of training:

```python
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log='mario_train_logs')
model.learn(total_timesteps=100)
model.save('mario_model')
```

Next is the basic definition of the model testing:

```python
model = PPO.load('mario_model.zip', env=env)

obs = env.reset()
for i in range(1000):
    obs = obs.copy()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
```

If the execution passes, then everything inside is just adding,copying, pasting, and replacing😂.  
Replaceable parameters include PPO, CNN, total steps, and so on.  
# At this point, the installation of the project is complete.
The parameter debugging logs of the project are in another file.  
- [PROJECTLOGS.md](PROJECTLOGS.md)

