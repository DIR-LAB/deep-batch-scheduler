# deep-batch-scheduler
repo for the deep batch scheduler research

## Installation

### Required Software
* Python 3.6
```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```
* OpenMPI 
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

* Tensorflow
```bash
sudo apt install python3.6-dev python3-pip
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3.6 ./venv
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
pip install --upgrade tensorflow
```

### Install GYM

```bash
git clone https://github.com/openai/gym.git
pip install -e ./gym
```

### Install SpinningUp
```bash
git clone https://github.com/openai/spinningup.git
pip install -e ./spinningup
```

### SSH Key Install
```bash
eval `ssh-agent -s`
ssh-add ~/...
```
### Install Deep Batch Scheduler
```bash
git clone git@github.com:DIR-LAB/deep-batch-scheduler.git
```