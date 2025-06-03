# MARIO-TRAIN

This is a project for research on Super-Mario-Bros reinforcement learning.  
The purpose and research focus of this project is on RL (`Reinforcement Learning`) for completing levels in
Super Mario (tentative).

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
    - [Python configuration file](requirements.txt)
3. If no errors are reported, the pip dependencies have been successfully configured.
4. This is only a temporary environment configuration for an early version; `Gym` has now migrated to `Gymnasium`.  
   So, if you encounter any compatibility issues, try Googling or, with a bit of luck, try changing the version numbers.

## Test whether the Mario based on NES can run.

```python
pass
```

   

