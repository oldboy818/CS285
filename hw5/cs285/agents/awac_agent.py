from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn

from cs285.agents.dqn_agent import DQNAgent

class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            ##################################################################################
            # 다음 상태에서의 정책 분포. 즉, actor distribution
            next_action_dist = self.actor(next_observations)
            # 정책에서 샘플링된 액션으로 다음 상태에서의 Q 값 계산
            next_qa_values = self.target_critic(next_observations)
            
            # Use the actor to compute a critic backup
            # 모든 가능한 행동에 대한 Q값을 정책 분포의 확률로 가중 평균된 Q값 계산
            next_actions_probs = next_action_dist.probs # .probs : PyTorch 분포 객체의 확률 값을 반환
            # compute E[Q(s', a')]
            next_qs = torch.sum(next_actions_probs * next_qa_values, dim=-1)
            ##################################################################################
            
            # TODO(student): Compute the TD target
            ##################################################################################
            target_values = rewards + self.discount * next_qs * (dones)
            ##################################################################################
        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        ##################################################################################
        qa_values = self.critic(observations)
        q_values = qa_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)
        ##################################################################################

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        ##################################################################################
        # advantage 함수는 '(정책 하의 Q값의 기대값) - (현재 Q값)'
        # 현재 Q 값 계산
        qa_values = self.critic(observations).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # 현재 정책 하의 Q 값의 기대값
        q_values = self.critic(observations)
        values = torch.sum(action_dist.probs * q_values, dim=-1)

        advantages = qa_values - values
        ##################################################################################
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        ##################################################################################
        # compute log actor distribution(policy)
        action_dist = self.actor(observations)
        log_prob = action_dist.log_prob(actions)

        # compute weight
        advantage = self.compute_advantage(observations, actions, action_dist)
        weight = torch.exp(advantage / self.temperature)

        # compute loss
        loss = -torch.mean(log_prob * weight)
        ##################################################################################
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
