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

## Tensorflow and all Driver install

### On Ubuntu 16.04 with Tesla GPUs 

查看这一页：https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8

查看是否安装好driver
```
nvidia-smi
```

安装Nvidia Modprobe
```bash
sudo apt-add-repository multiverse
sudo apt update
sudo apt install nvidia-modprobe
```

增加Nvidia的repo
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-387
```

安装CUDA和cuDNN
https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07

Spinningup会安装Tensorflow。覆盖掉之前安装的Tensorflow-GPU.

16.04 安装 Python3.6
```
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
```

判断Tensorflow是不是在使用GPU
```
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
```
