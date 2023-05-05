from HPCSimPickJobs import *
import numpy as np
import os
import torch
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from torch.nn.functional import relu, softmax
from torch.nn import MultiheadAttention
from torch.distributions import Categorical
from functools import partial

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
        self.latent_dim_pi = 8
        self.latent_dim_vf = 8

        self.attn_layer = MultiheadAttention(JOB_FEATURES-1, 1, batch_first=True)

        self.kernel_net = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES-1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU()
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
            nn.Linear(in_features=JOB_FEATURES-1, out_features=32),
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
            nn.ReLU()
        )
        self.critic_un_lg = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE*(JOB_FEATURES-1), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim_vf)
        )
        self.critic_un = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE*(JOB_FEATURES-1), out_features=32),
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
            nn.Linear(in_features=MAX_QUEUE_SIZE*(JOB_FEATURES-1), out_features=32),
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
        pi, mask = self.forward_actor(observation)
        return pi, mask, self.forward_critic(observation)

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
            mask = x[:, :, -1].flatten()
            x = x[:, :, :-1]
            if self.actor_model == 'attn':
                queries = relu(nn.Linear(in_features=JOB_FEATURES-1, out_features=32)(x))
                keys = relu(nn.Linear(in_features=JOB_FEATURES-1, out_features=32)(x))
                values = relu(nn.Linear(in_features=JOB_FEATURES-1, out_features=32)(x))
                output = self.attn_layer(queries, keys, values)[0]
                x = softmax(output, dim=1)
                x = relu(nn.Linear(in_features=x.shape[-1], out_features=16)(x))
                x = relu(nn.Linear(in_features=16, out_features=8)(x))
                x = nn.Linear(in_features=8, out_features=self.latent_dim_pi)(x)
            else:
                x = self.models[self.actor_model](x)
            return x, mask
        else:
            #TODO: Add error handling
            pass

        return x

    def forward_critic(self, x):
        """Forward pass for critic
        Args:
            x (torch.Tensor): input tensor
        Returns:    
            torch.Tensor: output tensor"""
        if not self.critic_model == 'critic_lg':
            x = torch.reshape(x, shape=(-1, MAX_QUEUE_SIZE, JOB_FEATURES))
            x = x[:, :, :-1]
            x = torch.reshape(x, shape=(-1, MAX_QUEUE_SIZE*(JOB_FEATURES-1)))
            x = self.models[self.critic_model](x)
        else:
            x = x.reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
            x = x[:, :, :-1]
            x = self.models[self.critic_model][0](x)
            #print(x.shape)
            x = torch.squeeze(x, dim=2)
            #print(x.shape)
            x = self.models[self.critic_model][1](x)
            #print(x.shape)
            #print('\n')
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

    def _predict(self, obs, deterministic):
        """Get the action and the value for a given observation"""
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, mask = self.mlp_extractor.forward_actor(features)
        actions = self.ret_actions(latent_pi, mask, deterministic=deterministic)
        return actions


    def ret_actions(self, latent_pi, mask, deterministic=False, ret_dist=False, sample=True):
        """Sample actions from the policy"""
        actions = torch.squeeze(latent_pi)
        actions = self.action_net(latent_pi)
        actions = torch.flatten(actions)
        actions = actions+(mask*1000000)
        distribution = Categorical(logits=actions)
        if sample:
            if deterministic:
                actions = torch.argmax(distribution.probs)
            else:
                actions = distribution.sample()
        if ret_dist:
            return actions, distribution
        return actions


    def forward(self, obs):
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, mask, latent_vf = self.mlp_extractor(features)
        actions, distribution = self.ret_actions(latent_pi, mask, 
                                                 False, ret_dist=True)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        values = torch.flatten(values)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, mask, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        values = torch.flatten(values)
        _, distribution = self.ret_actions(latent_pi, mask, 
                                                      sample=False, ret_dist=True)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def actor_head(self, latent_pi):
        """Forward pass for actor network
        Args:
            latent_pi (torch.Tensor): latent pi tensor
        Returns:
            torch.Tensor: output tensor"""
        net = nn.Linear(in_features=latent_pi, out_features=1)
        return net
    
    def value_head(self, latent_vf):
        """Forward pass for critic network
        Args:
            latent_vf (torch.Tensor): latent vf tensor
        Returns:
            torch.Tensor: output tensor"""
        net = nn.Linear(in_features=latent_vf, out_features=1)
        return net
    
    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.actor_head(latent_dim_pi)

        self.value_net = self.value_head(self.mlp_extractor.latent_dim_vf)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

def init_env(workload_path, args, check=True):
    customEnv = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, 
                    job_score_type=args.score_type, batch_job_slice=args.batch_job_slice, build_sjf=False)
    customEnv.seed(args.seed)
    customEnv.my_init(workload_file=workload_path)
    if check:
        check_env(customEnv)
    return customEnv
 
def init_dir_from_args(args):
    score_type_dict = {0: 'bsld', 1: 'wait_time', 2: 'turnaround_time', 3: 'resource_utilization'}
    workload_name = args.workload.split('/')[-1].split('.')[0]
    current_dir = os.getcwd()

    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, args.log_dir)
    model_dir = f'{args.model_dir}/{score_type_dict[args.score_type]}/{workload_name}/'
    #model_dir = os.path.join(current_dir, model_dir)
    return model_dir, log_data_dir, workload_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model_dir', type=str, default='./trained_models')
    parser.add_argument('--log_dir', type=str, default='./data/logs/lublin_256')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rollout_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_rollouts', type=int, default=400)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--stats_window_size', type=int, default=5)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    args = parser.parse_args()

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(args)

    # create environment
    env = init_env(workload_file, args)
    if args.trained_model is not None:
        model = PPO.load(args.trained_model, env=env)
    else:
        model = PPO(CustomActorCriticPolicy, env, learning_rate=args.lr, 
                    seed=args.seed, n_epochs=args.epochs, batch_size=args.batch_size, 
                    n_steps=args.rollout_steps, gamma=args.gamma, 
                    clip_range=args.clip_range, gae_lambda=args.gae_lambda, 
                    target_kl=args.target_kl, policy_kwargs=dict(), tensorboard_log=log_data_dir, 
                    normalize_advantage=False, device=args.device, verbose=args.verbose, 
                    stats_window_size=args.stats_window_size)
        env_steps = args.rollout_steps*args.num_rollouts
        model.learn(env_steps, progress_bar=True)
        model.save(f"{model_dir}ppo_HPC")
        print(f"Trained model saved at: {model_dir}ppo_HPC")
