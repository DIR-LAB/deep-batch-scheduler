# deep-batch-scheduler
repo for the deep batch scheduler research on Ubuntu 16.04

## Installation

### Required Software
* Python 3.7
```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
* OpenMPI 
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

* Tensorflow
```bash
sudo apt install python3.7-dev python3-pip
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3.7 ./venv
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

### Default Training

```bash
python ppo-pick-algms.py --exp_name what-ever-name-you-want --trajs 500 --seed 2
```
There are manyother parameters in the source file.

vpg.py and ppo.py are traning the RL agent to pick a job. On the other hand, vpg-pick-algms.py is to train the RL agent to pick a sort algorithm to use to schedule current job queue. 
