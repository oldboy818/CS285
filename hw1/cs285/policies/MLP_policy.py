import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy

from torch.distributions import Categorical, Normal

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # 확인: obs가 NumPy 배열인 경우 차원을 확인하여 적절히 변환
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]  # 관측값이 하나인 경우 배치 차원 추가

        # TODO return the action that the policy prescribes
        # NumPy 배열을 PyTorch 텐서로 변환
        obs_tensor = ptu.from_numpy(observation)

        # 모델을 사용하여 행동을 결정하고, 결과로 나오는 텐서를 샘플링
        # 여기서 self.forward는 신경망 모델을 나타냄
        action_tensor = self.forward(obs_tensor)
        # action = action_tensor.sample()
        
        # PyTorch 텐서를 NumPy 배열로 변환
        action = ptu.to_numpy(action_tensor)

        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!

    def forward(self, observation: torch.FloatTensor) -> Any:
        if self.discrete:
            # Use logits network for discrete action space
            logits = self.logits_na(observation)
            # return Categorical(logits=logits)
            return logits
        else:
            # Use mean network and logstd for continuous action space
            mean = self.mean_net(observation)
            # Create a normal distribution with the mean and the standard deviation
            std = torch.exp(self.logstd)
            # return Normal(mean, std)
            return mean


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # # TODO: update the policy and return the loss
        # loss = TODO
        # return {
        #     # You can add extra logging information here, but keep this line
        #     'Training Loss': ptu.to_numpy(loss),
        # }
        # Retrieve relevant objects from self
        loss_fn = self.loss
        optimizer = self.optimizer

        # Setup our optimizer for this train step
        optimizer.zero_grad()

        # Convert our obs into a form usable by our model
        obs_pt = ptu.from_numpy(observations)

        # Retrieve model output actions
        model_actions = self(obs_pt)

        # Convert loss inputs to a form usable by the loss object
        actions_pt = ptu.from_numpy(actions)

        # Calculate loss
        loss = loss_fn(model_actions, actions_pt)

        # Update parameters
        loss.backward()
        optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
