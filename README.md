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


### File Structure

```
data/: Contains a series of workload and real-world traces.
cluster.py: Contains Machine and Cluster classes.
job.py: Contains Job and Workloads classed. 
compare-pick-jobs.py: Test training results and compare it with different policies.
HPCSimPickJobs.py: SchedGym Environment.
ppo-pick-jobs.py: Train RLScheduler using PPO algorithm.
```

### Default Training

```bash
python ppo-pick-jobs.py --workload "./data/lublin_256.swf" --exp_name your-exp-name --trajs 500 --seed 0
```
There are many other parameters in the source file.

### Monitor Training 

After running Default Training, a folder named `logs/your-exp-name/` will be generated under `./data`. 

```bash
python spinningup/spinup/utils/plot.py ./data/logs/your-exp-name/
```

It will plot the training curve.

### Test and Compare

After RLScheduler converges, you can test the result and compare it with different policies such as SJF, F1 and so on.

```bash
python compare-pick-jobs.py --rlmodel "./data/logs/your-exp-name/your-exp-name_s0/" --workload "./data/lublin_256.swf --len 2048 --iter 10"
```
