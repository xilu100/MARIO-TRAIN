# MARIO-TRAIN
This is a project for research on Super-Mario-Bros reinforcement learning.  
The purpose and research focus of this project is on RL (`Reinforcement Learning`) for completing levels in 
Super Mario (tentative).

The project originates from this author. Thanks to the author for open-sourcing it.   
URL:https://github.com/Kautenja/gym-super-mario-bros
## Environment-setup
**Attention**:This project is only compatible with `Windows/Linux` and works with `CPU`,`NVIDIA` GPUs with `CUDA`, `Intel ARC` 
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
   "Next -> I Agree -> Just Me -> Next-> Next -> Add to PATH -> Install -> finish"
4. Open the `Terminal`/`PowerShell*`/`CMD`(It is not recommended to use CMD).
5. Type ```conda init powershell```.
6. Restart the terminal.
7. If all goes well, it should be in this format:`(base) PS PATH\PATH\PATH>`
8. Create a Conda environment based on Python 3.8.8.  
```conda create -n mario python=3.8.8```  
   (The environment name after `-n` can be anything you like, as long as you can remember it. 
   Here, we will use "mario" consistently.)
9. If all goes well, it should be in this format:`(mario) PS PATH\PATH\PATH>`
10. It is recommended to perform all the following operations within this environment; 
    otherwise, compatibility issues may occur.
#### Linux/WSL2
```PASS```
### Step 2. Configure accelerated computing frameworks / hardware acceleration platforms.
#### CUDA
```PASS```
#### IPEX
```PASS```
#### Integrated Graphics
```PASS```

 

