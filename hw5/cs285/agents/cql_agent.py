from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        # soft-maxmimum Q-value 계산에 스무딩 효과를 주는 역할
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        
        # standard TD error loss
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        ##########################################################################################
        # loss = loss + ...

        # extract Q-values from variables. 
        # qa_values : 데이터셋에서 state-action에 대한 Q, q_values : 모든 가능한 action에 대한 Q
        qa_values = variables["qa_values"]
        q_values = variables["q_values"]

        # CQL regularization terms
        # Log-sum-exp term over all actions for each state. log(sum_a{exp(q_values)})를 계산.
        logsumexp_q = torch.logsumexp(q_values / self.cql_temperature, dim=1).mean()

        # Q-value for the actions taken in the dataset
        dataset_q = qa_values.mean()

        # CQL loss 계산
        cql_loss = self.cql_alpha * (logsumexp_q - dataset_q)

        # 기존 손실에 CQL loss 추가. 즉, TD loss에 CQL loss 추가
        loss = loss + cql_loss

        # Update metrics
        metrics["cql_loss"] = cql_loss.item()
        ##########################################################################################
        return loss, metrics, variables
