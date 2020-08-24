# deep-batch-scheduler
This repo includes the deep batch scheduler source code and necessary datasets to run the experiments/tests. 

The code has been tested on Ubuntu 18.04/16.04 with Tensorflow 1.14 and SpinningUp 0.2. Newer version of Tensorflow (such as 2.x) does not work because of the new APIs. Windows 10 should be OK to run the code, only the installation of the dependencies (such as Gym environment and SpinningUp could be bumpy).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3879814.svg)](https://doi.org/10.5281/zenodo.3879814)

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

* Virtualenv
```bash
sudo apt install python3.7-dev python3-pip
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3.7 ./venv
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
```

### Clone Deep Batch Scheduler
```bash
git clone https://github.com/DIR-LAB/deep-batch-scheduler.git
```

### Install Dependencies
```shell script
cd deep-batch-scheduler
pip install -r requirements.txt
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

To change the hyper-parameters, such as `MAX_OBSV_SIZE` or the trajectory length during training, you can change them in HPCSimPickJobs.py. You can also change to different neural networks (MLP and LeNet) in HPCSimPickJob.py. 

### Training
To train a RL model based on a job trace, run this command:
```bash
python ppo-pick-jobs.py --workload "./data/lublin_256.swf" --exp_name your-exp-name --trajs 500 --seed 0
```
There are many other parameters in the source file.
* `--model`, specify a saved trained model (for two-step training and re-training)
* `--pre_trained`, specify whether this trainig will be a twp-step training or re-training
* `--score_type`, specify which scheduling metrics you are optimizing for: [0]：bounded job slowdown；[1]: job waiting time; [2]: job response time; [3] system resource utilization.

### Monitor Training 

After running Default Training, a folder named `logs/your-exp-name/` will be generated under `./data`. 

```bash
python spinningup/spinup/utils/plot.py ./data/logs/your-exp-name/
```

It will plot the training curve.

### Test and Compare

After RLScheduler converges, you can test the result and compare it with different policies such as FCFS, SJF, WFP3, UNICEP, and F1.

```bash
python compare-pick-jobs.py --rlmodel "./data/logs/your-exp-name/your-exp-name_s0/" --workload "./data/lublin_256.swf --len 2048 --iter 10"
```
There are many parameters you can use:
* `--seed`, the seed for random sampling
* `--iter`, how many iterations for the testing
* `--backfil`, enable/disable backfilling during the test
* `--score_type`, specify the scheduling metrics. [0]：bounded job slowdown；[1]: job waiting time; [2]: job response time; [3] system resource utilization.

## A Step-By-Step Example

Here, we give a step-by-step example to show the complete training/monitoring/testing workflow of RLScheduler.

* Step 1: Train a model using Lublin_256 data trace and name the experiment as lublin256-seed0 
```bash
python ppo-pick-jobs.py --workload "./data/lublin_256.swf" --exp_name lublin256-seed0 --trajs 500 --seed 0
```
In this experiment, we have `seed=0`, collect 500 trajectories in each epoch, and optimize average bounded slowdown. 

* Step 2: Monitor the training by checking the training curves
```bash
python plot.py ./data/logs/lublin256-seed0 -x Epoch -s 1
```
It will output something like this:
<figure>
	<img align="middle" src="https://github.com/DIR-LAB/deep-batch-scheduler/blob/master/trained_models/resources/lublin256_training_epoch.png" alt="Lublin256 Training Curve"/ width="400">
</figure>

* Step 3: Schedule 10 randomly sampled job sequence from the job trace
```bash
python compare-pick-jobs.py --rlmodel "./data/logs/lublin256-seed0/lublin256-seed0_s0/" --workload "./data/lublin_256.swf" --seed 1 --len 1024 --iter 10
```
In this scheduling case, we randomly select 10 job sequences using `seed=1`. It will output something like this for comparing different scheduling results:
<figure>
	<img align="middle" src="https://github.com/DIR-LAB/deep-batch-scheduler/blob/master/trained_models/resources/lublin256_1024.png" alt="Lublin256 Training Curve"/ width="400">
</figure>
We use the average to produce the performance tables in the paper.

## Reproduce Results in Paper

We provide the script and several trained models to help reproduce the key results shown in the paper, particularly Table V and Table VI. 

### Results of Scheduling Towards average bounded slowdown
```shell script
python make_table_script.py --score_type "bsld"
```

| Trace               | FCFS    | WFP3     | UNI      | SJF     | F1      | RL     |
|---------------------|---------|----------|----------|---------|---------|--------|
| Without backfilling |         |          |          |         |         |        |
| Lublin-1            | 7273.77 | 19753.53 | 22274.74 | 277.35  | 258.37  | **254.67** |
| SDSC-SP2            | 1727.54 | 3000.88  | 1848.45  | 2680.55 | 1232.06 | **466.44** |
| HPC2N               | 297.18  | 426.99   | 609.77   | 157.71  | 118.01  | **117.01** |
| Lublin-2            | 7842.47 | 9523.18  | 11265.31 | 787.89  | **698.34**  | 724.51 |
| With backfilling    |         |          |          |         |         |        |
| Lublin-1            | 235.82  | 133.87   | 307.23   | 73.31   | 75.07   | **58.64**  |
| SDSC-SP2            | 1595.12 | 1083.12  | 548.01   | 2167.84 | 1098.22 | **397.82** |
| HPC2N               | 127.38  | 97.39    | 175.12   | 122.04  | **71.95**   | 86.14  |
| Lublin-2            | 247.61  | 318.35   | 379.59   | **91.99**   | 148.25  | 118.79 |

### Results of Scheduling Towards resource utilization
```shell script
python make_table_script.py --score_type "utilization"
```

| Trace               | FCFS  | WFP3  | UNI   | SJF   | F1    | RL    |
|---------------------|-------|-------|-------|-------|-------|-------|
| Without backfilling |       |       |       |       |       |       |
| Lublin-1            | 0.657 | 0.747 | 0.691 | 0.762 | **0.816** | 0.714 |
| SDSC-SP2            | 0.670 | 0.658 | **0.688** | 0.645 | 0.674 | 0.671 |
| HPC2N               | 0.638 | 0.636 | 0.636 | 0.640 | 0.637 | **0.640** |
| Lublin-2            | 0.404 | 0.543 | 0.510 | 0.562 | 0.478 | **0.562** |
| With backfilling    |       |       |       |       |       |       |
| Lublin-1            | 0.868 | 0.864 | **0.883** | 0.778 | 0.840 | 0.850 |
| SDSC-SP2            | 0.682 | 0.681 | 0.706 | 0.661 | 0.677 | **0.707** |
| HPC2N               | 0.639 | 0.637 | 0.638 | 0.641 | 0.638 | **0.642** |
| Lublin-2            | 0.587 | 0.583 | 0.587 | 0.593 | 0.552 | **0.593** |
