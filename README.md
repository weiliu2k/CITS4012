# Installing Anaconda

* [Anaconda Installation Instructions](https://docs.anaconda.com/anaconda/install/windows/)
* Individual Version (`Anaconda3-2021.05-Windows-x86_64.exe` - 488KB)
* Installation Size (2.9 GB)
* Installed for all users at `C:\ProgramData\Anaconda3`
* Installation Time (~30 mins)
* Launch Anaconda Navigator

# Installing Visual Studio Code

* [Visual Studio Code Download](https://code.visualstudio.com/download)
* 64 bit System Installer (`VSCodeSetup-x64-1.57.1.exe` - 78KB)
* Default Installation Path (`C:\Program Files\Microsoft VS Code`)
* Installation Time (~5 mins)

# Installing Python Extension for Visual Studio Code
* [Python Extension for VSCode Instruction](https://code.visualstudio.com/docs/python/python-tutorial#_install-visual-studio-code-and-the-python-extension)
* Installation Time (~2-3 mins)

# Create Anaconda Environment
1. `conda create -n CITS4012_base python=3.8`
2. `conda activate CITS4012_base`

# Use the virtual environment in VSCode
* [Instructions on how to use environment in VSCode](https://code.visualstudio.com/docs/python/environments#_conda-environments) 
* Test to see if the CITS4012_base environment is available from VSCode

# Install NLP packages
## Install Spacy
1. Go back to Conda CMD.exe, check to see if you have `pip` installed using 
`conda list`
`pip install -U spacy`
`python -m spacy download en_core_web_sm`
 
2. Find the Spacy version (we want v3+): 

```
# Windows CMD
C:\> conda list | findstr "spacy"`

# Windows PowerShell
C:\> conda list | Select-String "spacy"

# Linux
$ conda list | grep "spacy"
```
## Install PyTorch
### Check for Cuda compatible Graphics Card on Windows
1. Click Start.
2. On the Start menu, click Run.
3. In the Open box, type "dxdiag" (without the quotation marks), and then click OK.
4. The DirectX Diagnostic Tool opens. ...
5. On the Display tab, information about your graphics card is shown in the Device section.

My laptop has NVIDIA GeForce MX130.<img width="1976" alt="anaconda_navigator" src="https://user-images.githubusercontent.com/1005582/122739014-1f3df600-d2b5-11eb-95db-4cf21f80c1d5.png">
<img width="779" alt="vscode_python_extension" src="https://user-images.githubusercontent.com/1005582/122739187-53191b80-d2b5-11eb-892c-6bb43e0ea1dc.png">

