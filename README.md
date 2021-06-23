# 1. IDEs and Git

## 1.1 Installing Visual Studio Code

* [Visual Studio Code Download](https://code.visualstudio.com/download)
* 64 bit System Installer (`VSCodeSetup-x64-1.57.1.exe` - 78KB)
* Default Installation Path (`C:\Program Files\Microsoft VS Code`)
* Installation Time (~5 mins)

<img width="50%" alt="vscode_python_extension" src="https://user-images.githubusercontent.com/1005582/122739187-53191b80-d2b5-11eb-892c-6bb43e0ea1dc.png">

### Installing Extensions for Visual Studio Code
* [Python Extension for VSCode Instruction](https://code.visualstudio.com/docs/python/python-tutorial#_install-visual-studio-code-and-the-python-extension)
* Remote - SSH Extension
* Remote X11 (SSH) Extension

## 1.2 Install git
* [Latest Version of Git for Windows](https://git-scm.com/download/win)
* Don't forget to add git to the system PATH

# 2. Conda and Unit Specific Packages

## 2.1 Installing Anaconda
* [Anaconda Installation Instructions](https://docs.anaconda.com/anaconda/install/windows/)
* Individual Version (`Anaconda3-2021.05-Windows-x86_64.exe` - 488KB)
* Installation Size (2.9 GB)
* Installed for all users at `C:\ProgramData\Anaconda3`
* Installation Time (~30 mins)
* Launch Anaconda Navigator

<img width="50%" alt="anaconda_navigator" src="https://user-images.githubusercontent.com/1005582/122739014-1f3df600-d2b5-11eb-95db-4cf21f80c1d5.png">

You can start a CMD or POWERSHELL console using the navigator, or following the steps 1 and 2 in the screenshot below to start a CMD or POWERSHELL. If you are intalling packages, you can right click the arrow to bring up a pop-up menu, `run as adminstrator` (Step 3a) or `pin on taskbar` (Step 3b) for future convenience.

<img width="50%" alt="conda_powershell" src="https://user-images.githubusercontent.com/1005582/123036101-110be900-d41f-11eb-8162-ffa8f300c694.png">

### Create Anaconda Environment
Need admin access to write to C drive (Run Conda Powershell as Administrator - right click on the icon)
1. `conda create -p c:\envs\cits4012\ python=3.8`
2. `conda activate c:\envs\cits4012`

### Use the virtual environment in VSCode
* [Instructions on how to use environment in VSCode](https://code.visualstudio.com/docs/python/environments#_conda-environments) 
* Test to see if the CITS4012_base environment is available from VSCode

## 2.2 Install NLP packages
### 2.2.1 Install Spacy
1. Go back to Conda CMD.exe, check to see if you have `pip` installed using 
* `conda list`
* `pip install -U spacy`
* `python -m spacy download en_core_web_sm`
 
2. Find the Spacy version (we want v3+): 

```
# Windows CMD
C:\> conda list | findstr "spacy"`

# Windows PowerShell
C:\> conda list | Select-String "spacy"

# Linux
$ conda list | grep "spacy"
```
### 2.2.2 Install PyTorch

#### Check for Cuda compatible Graphics Card on Windows
1. Click Start.
2. On the Start menu, click Run.
3. In the Open box, type "dxdiag" (without the quotation marks), and then click OK.
4. The DirectX Diagnostic Tool opens. ...
5. On the Display tab, information about your graphics card is shown in the Device section.

My laptop has NVIDIA GeForce MX130.

#### Install Pytorch 
[Pytorch Website](https://pytorch.org)  

* with GPU 

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge`

* CPU only 

`conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge`

#### Install Tensorboard
`conda install -c conda-forge tensorboard`

#### Install GraphViz on Windows
[2.47.3 EXE installer for Windows 10 (64-bit)](https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/2.47.3/stable_windows_10_cmake_Release_x64_graphviz-install-2.47.3-win64.exe)

Download the exe file and install, make sure it is added to the system PATH (Windows - Edit the Windows Environment Variables.

<img width="50%" alt="GraphVizInstall" src="https://user-images.githubusercontent.com/1005582/122881303-9767f280-d36d-11eb-8188-0163c59eab01.png">

#### Install torchviz
`pip install torchviz`

### 2.2.3 Install NLTK
`pip install nltk`

and then download the data and models 

`python -m nltk.downloader -d c:\envs\cits4012\nltk_data all`

#### Install truecase 
install this after NLTK installation pls.

`pip install truecase`

## 2.3 Install Jupyterlab
`conda install -c conda-forge jupyterlab`

## 2.4 Install Scikit-learn
`pip install -U scikit-learn`

Verify if it works:

`python -c "import sklearn; sklearn.show_versions()"`

