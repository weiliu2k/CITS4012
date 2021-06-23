# Frequently encounted problems in Windows

1. File not found when files are actually there - this might be to do with the limitation of path length (<=260 characters) in Windows. See below for a solution

* [How to set group policies in Windows to allow for long path](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/)

2. When trying to install some pip packages you may get the error stating:

> Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio": https://visualstudio.microsoft.com/downloads/

Scroll down the page to find "All Tools", expand "Tools for Visual Studio 2019" and find "Build Tools for Visual Studio 2019", download and install as an admin for all users.

<img width="50%" alt="vs-build-tools" src="https://user-images.githubusercontent.com/1005582/123034052-6d6d0980-d41b-11eb-8fe2-2f541825375d.png">

Warning: The installation of the Build Tools might take a while (~20mins).
 
# Create a new environment

`conda create -p c:\envs\cits4012_py37 python=3.7`
`conda activate c:\envs\cits4012_py37`

# Install Flair
Flair requires different versions of numpy and torch, so it is better to isolate it from the normal environment

`pip install flair`

# Install Neuralcoref
Follow the "compile from source instruction" on the github page as it requires Python 3.7 and Spacy 2.0+

```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```
You may run into the frequent problem 2 above. Solve it by installing C++ compiler suitable for your OS. 

# Install Jupyterlab
`pip install jupyterlab`
