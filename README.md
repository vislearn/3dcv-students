# Exercises for 3D Computer Vision

## Requirements
* Python 3.10 (Maybe works with Python 3.8)
* It is recommended to set up Miniconda or Anaconda

## Installation

### Anaconda / Miniconda (Unix/Windows/Mac)
After installing Anacodna / Miniconda (follow guide: https://docs.anaconda.com/anaconda/install/)
create an Virtual Environment
```sh
conda create -n 3dcv-students python=3.10
```
wait until it installs and activate the environment with:
```sh
conda activate 3dcv-students
```
Than install all requirements as usual
```sh
pip install -r requirements.txt
```

### Unix (Python only)
For the installation I recommend a python venv.
```sh
git clone git@github.com:vislearn/3dcv-students.git
cd 3dcv/
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

### Windows (Python only)
Download Python [here](https://www.python.org/downloads/windows/).
Make sure that the installer adds Python to your PATH.

In PowerShell run
```bat
git clone git@github.com:vislearn/3dcv-students.git
```
(if you don't use git, just download and extract the zip file)
```bat
cd .\3dcv\
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip install -r .\requirements.txt
```

## Usage

In order to edit the notebooks run
```sh
jupyter-lab
```
to start the local webserver.
If the notebook home page does not show up immediately, open `http://localhost:8888` in your browser.
Now, just open one of the task notebooks and start editing.

## Project Organization
    
    ├── 1.0-tl-scientific-python.ipynb          <- The Notebooks, containing your tasks
    │
    ├── ...
    │
    ├── LICENSE                                 <- The License
    │
    ├── README.md                               <- The top-level README with installation instructions
    │
    ├── data                                    <- The necessary data files, e.g. datasets
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt                        <- The requirements file to setup the environment
    │
    ├── task.pdf                               <- Detailed information about your tasks
    │
    ├── vll                                     <- Boilerplate source code provided by us
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── model                               <- Scripts defining the neural network models
    │   │
    │   ├── utils                               <- Scripts utilities used during data generation or training
    │   │
    │   ├── validate                            <- Scripts to validate models
    │   │
    │   └── visualize                           <- Scripts to create exploratory and results oriented visualizations
