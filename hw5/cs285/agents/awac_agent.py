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
            # compute Q values for next_observations using the critic
            next_qa_values = self.critic(next_observations)     # (batch, num_act)

            # Use the actor to compute a critic backup
            # compute the actor distribution(policy) for next_observations
            next_action_dist = self.actor(next_observations)   # (batch, num_act)
            # E_{a'~pi} [Q(s', a')]
            next_qs = torch.sum(next_action_dist.probs * next_qa_values, dim=-1)    # (batch, )
            ##################################################################################

            # TODO(student): Compute the TD target, r + γ E_{a'~pi(a'|s')}[Q(s', a')]
            ##################################################################################
            target_values = rewards + (1.0 - dones.float()) * self.discount * next_qs   # (batch, )
            ##################################################################################
        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        ##################################################################################
        # Q(s, a)
        qa_values = self.critic(observations)   # (batch, num_act)

        # actions는 (batch,) 형태이이고 .unsqueeze(-1)을 통해 (batch, 1) 형태로 변환
        # gather를 통해 각 상태에서 실제 행동에 해당하는 Q값만 선택하여 (batch,) 형태로 반환
        # .squeeze(-1)을 통해 불필요한 차원 제거하여 다시 (batch,) 형태로 변환
        q_values = qa_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)   # (batch, )
        assert q_values.shape == target_values.shape

        # MSE 손실 계산
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
        # Q(s,a). 현재 Q 값 계산 
        qa_values = self.critic(observations)   # Q(s,·) (batch, num_act)
        q_values = qa_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)   # Q(s,a) (batch, )
        
        # E[Q(s,a)] = V(s). 현재 정책 하의 Q 값의 기대값, 즉 V(s) 계산
        values = torch.sum(action_dist.probs * qa_values, dim=-1)   # (batch, )

        # compute advantage A(s,a) = Q(s,a) - E[Q(s,a)] = Q - V & normalize
        advantages = q_values - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ##################################################################################
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        ##################################################################################
        # compute log actor distribution(policy). log pi(·|s)
        action_dist = self.actor(observations)
        log_prob = action_dist.log_prob(actions)

        # compute weight
        advantage = self.compute_advantage(observations, actions, action_dist)  # 현재 정책의 advantage A(s,a)
        # torch.clamp(input, min, max):     input 텐서의 모든 원소 값을 min과 max 사이의 범위로 제한하여 수치적 안정성 확보
        # .detach():    현재 계산 그래프(computation graph)에서 텐서를 분리(detach)합니다. 
        # 즉, 이 텐서로부터는 더 이상 그래디언트가 역전파되지 않도록 만듭니다. weight 텐서에 .detach()를 적용하면, 
        # 액터 손실로부터 계산된 그래디언트가 weight를 통과하지 못하고 액터 네트워크까지만 흘러가게 됩니다. 
        # 이는 액터 업데이트가 크리틱 파라미터에 영향을 주지 않도록 보장하는 역할
        weight = torch.exp(torch.clamp(advantage / self.temperature, -10, 10)).detach()

        # compute loss
        loss = -torch.mean(log_prob * weight)
        ##################################################################################
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        # super().update(...)를 호출하여 부모 클래스(DQNAgent)의 업데이트 로직(critic, target update)을 먼저 수행
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
