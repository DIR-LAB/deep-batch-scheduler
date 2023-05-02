from HPCSimPickJobs import *
import numpy as np
import os
import torch
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import PPO
from torch.nn.functional import relu, softmax
from torch.nn import MultiheadAttention


class CustomTorchModel(nn.Module):

    def __init__(self, observation_space, action_space, actor_model='kernel', critic_model='critic_lg'):
        """ Initialize the custom model
        :param observation_space: (gym.spaces.Box)
        :param action_space: (gym.spaces.Discrete)
        :param actor_model: (str) the name of the actor model
        :param critic_model: (str) the name of the critic model"""

        super(CustomTorchModel, self).__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.m = int(np.sqrt(MAX_QUEUE_SIZE))
        self.latent_dim_pi = action_space.n
        self.latent_dim_vf = action_space.n

        self.attn_layer = MultiheadAttention(JOB_FEATURES, 1, batch_first=True)

        self.kernel_net = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(), 
            nn.Linear(in_features=8, out_features=1),
            nn.Sigmoid()
        )

        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16, out_features=self.latent_dim_pi)
        )
        self.critic_lg1 = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1)
        )
        self.critic_lg2 = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.latent_dim_vf)
        )
        self.critic_un_lg = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE*JOB_FEATURES, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim_vf)
        )
        self.critic_un = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE*JOB_FEATURES, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.latent_dim_vf)
        )
        self.critic_sm = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE*JOB_FEATURES, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.latent_dim_vf)
        )
        self.models = {
            'kernel': self.kernel_net,
            'conv': self.lenet,
            'critic_lg': [self.critic_lg1, self.critic_lg2],
            'critic_un_lg': self.critic_un_lg,
            'critic_un': self.critic_un,
            'critic_sm': self.critic_sm,
        }

    def forward(self, observation):
        """Forward pass for both actor and critic"""
        return self.forward_actor(observation), self.forward_critic(observation)

    def forward_actor(self, x): 
        """Forward pass for actor
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor"""
        if self.actor_model=='conv':
            x = torch.reshape(x, (-1, self.m, self.m, JOB_FEATURES))
            x = self.models[self.actor_model](x)
        elif self.actor_model in ['kernel', 'attn']:
            x = x.reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
            if self.actor_model == 'attn':
                queries = relu(nn.Linear(in_features=JOB_FEATURES, out_features=32)(x))
                keys = relu(nn.Linear(in_features=JOB_FEATURES, out_features=32)(x))
                values = relu(nn.Linear(in_features=JOB_FEATURES, out_features=32)(x))
                output = self.attn_layer(queries, keys, values)[0]
                x = softmax(output, dim=1)
                x = relu(nn.Linear(in_features=x.shape[-1], out_features=16)(x))
                x = relu(nn.Linear(in_features=16, out_features=8)(x))
                x = nn.Linear(in_features=8, out_features=self.latent_dim_pi)(x)
            else:
                x = self.models[self.actor_model](x)
                x = torch.squeeze(x, dim=-1)
            return x
        else:
            #TODO: Add error handling
            pass

        return x

    def forward_critic(self, x):
        if not self.critic_model == 'critic_lg':
            x = torch.reshape(x, shape=(-1, MAX_QUEUE_SIZE*JOB_FEATURES))
            x = self.models[self.critic_model](x)
        else:
            x = x.reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
            x = self.models[self.critic_model][0](x)
            x = torch.squeeze(x, dim=-1)
            x = self.models[self.critic_model][1](x)
        return x

class CustomActorCriticPolicy(ActorCriticPolicy):
  """Custom Actor Critic Policy for HPC RL"""

  def __init__( self, observation_space, action_space,
        lr_schedule, net_arch = None, activation_fn= nn.Tanh, 
        *args, **kwargs):
    """Initialize the policy based on Stable Baselines' ActorCriticPolicy"""

    super(CustomActorCriticPolicy, self).__init__(
        observation_space, action_space, lr_schedule,
        net_arch, activation_fn, *args, **kwargs)
    
  def _build_mlp_extractor(self) -> None:
    """Build the feature extractor using custom Torch model"""
    self.mlp_extractor = CustomTorchModel(self.observation_space, self.action_space)

def init_env(workload_path, args, check=True):
    customEnv = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, 
                    job_score_type=args.score_type, batch_job_slice=args.batch_job_slice, build_sjf=False)
    customEnv.seed(args.seed)
    customEnv.my_init(workload_file=workload_path)
    if check:
        check_env(customEnv)
    return customEnv
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model_dir', type=str, default='./data/')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)

    args = parser.parse_args()

    
    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    #logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    env = init_env(workload_file, args)
    if args.trained_model is not None:
        model = PPO.load(args.trained_model, env=env)
        PPO(CustomActorCriticPolicy, env, learning_rate=3e-4, seed=0, n_epochs=50, gamma=0.99, clip_range=0.2, gae_lambda=0.97, target_kl=0.01, policy_kwargs=dict(), normalize_advantage=True, tensorboard_log=os.getcwd())
    else:
        
        model = PPO(CustomActorCriticPolicy, env, learning_rate=3e-4, seed=0, n_epochs=50, gamma=0.99, clip_range=0.2, gae_lambda=0.97, target_kl=0.01, policy_kwargs=dict(), tensorboard_log=os.getcwd(), normalize_advantage=True)
        callback = ReplayBuffer(args.trajs, observation_space=model.observation_space, action_space=model.action_space)
        model.learn(200000, progress_bar=True)
        model.save(f"{args.model_dir}ppo_HPC")
