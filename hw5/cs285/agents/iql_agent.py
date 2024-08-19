from typing import Optional
import torch
from torch import nn
from cs285.agents.awac_agent import AWACAgent

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_value_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.value_critic = make_value_critic(observation_shape)
        self.target_value_critic = make_value_critic(observation_shape)
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        self.expectile = expectile

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): Compute advantage with IQL
        ################################################################################
        with torch.no_grad():
            # 각 상태(obs)에서 행동에 해당하는 Q값
            # self.critic(obs): critic 신경망으로 관찰된 상태의 Q값을 계산. (batch_size, num_actions)
            # actions.unsqueeze(1): 텐서 차원 늘림. (batch_size,) -> (batch_size, 1)
            # gather(1, actions.unsqueeze(1)): 신경망 출력 Q값 중 각 상태에서 실제 행동에 해당하는 Q값만 선택. (batch_size, )
            q_values = self.critic(observations).gather(1, actions.unsqueeze(1)).squeeze(1) # (batch_size, )
            
            # self.value_critic(obs): value critic 신경망으로 각 상태의 V값 계산. (batch_size, 1)
            v_values = self.value_critic(observations).squeeze(1)   # (batch_size, )

            # advantage = Q(s,a) - V(s)
            advantages = q_values - v_values
        return advantages
        ################################################################################

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Update Q(s, a) to match targets (based on V)
        ################################################################################
        with torch.no_grad():
            # V(s')
            target_v = self.target_value_critic(next_observations).squeeze(1)   # (batch_size, )
            # Q(s, a) <-- r(s,a) + V(s')
            target_q = rewards + (1.0 - dones.float()) * target_v
        
        # Q(s,a)
        q_values = self.critic(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        # MSE loss
        loss = self.critic_loss(q_values, target_q)
        ################################################################################
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": self.critic_loss(q_values, target_q).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_q.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }
        # metrics = {
        #     "q_loss": loss.item(),
        #     "q_values": q_values.mean().item(),
        #     "target_values": target_q.mean().item(),
        #     "q_grad_norm": grad_norm.item(),
        # }
        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Compute the expectile loss
        ################################################################################
        # expectile loss = (1-tau) * x^2 (x>0), tau * x^2 (x<=0)

        # x = V(s) - Q(s,a)
        x = vs - target_qs
        weight = torch.where(x > 0, expectile, 1 - expectile)
        
        loss = weight * (x ** 2)

        return loss.mean()
        ################################################################################
    
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)
        """
        # TODO(student): Compute target values for V(s)
        ################################################################################
        with torch.no_grad():
            # Q(s,a)
            q_values = self.critic(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        ################################################################################

        # TODO(student): Update V(s) using the loss from the IQL paper
        ################################################################################
        # V(s)
        v_values = self.value_critic(observations).squeeze(1)
        # expectile loss (V(s), Q(s, a))
        loss = self.iql_expectile_loss(self.expectile, v_values, q_values)
        ################################################################################
        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (v_values - q_values).mean().item(),
            "vs": v_values.mean().item(),
            "target_values": q_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(observations, actions, rewards, next_observations, dones)
        metrics["actor_loss"] = self.update_actor(observations, actions)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()
        
        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
