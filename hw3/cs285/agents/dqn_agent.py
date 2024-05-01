from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        ########################################################################################################

        with torch.no_grad():
            # get Q values from critic
            q_vals = self.critic(observation)
            # get action that maximize Q values
            best_action = torch.argmax(q_vals, dim=1).item()

        if np.random.random() < epsilon:    # best_action이 아닌 경우
            # choose a random action excluding the best action
            possible_actions = [act for act in range(self.num_actions) if act != best_action]
            action = np.random.choice(possible_actions)

        else:
            action = best_action
        ########################################################################################################
        # return ptu.to_numpy(action).squeeze(0).item()
        return action

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            ##################################################################################
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                raise NotImplementedError
            else:
                next_action = torch.argmax(next_qa_values, dim=1)
            
            # select corresponding Q values directly using indexing
            next_q_values = next_qa_values[torch.arange(batch_size), next_action]
            # calculate target Q values for the current action
            # done = True면, '1 - done'은 0이 되어 'next_q_values'의 영향을 받지 않도록 한다.
            # 즉, 에피소드가 끝났을 때 미래 보상을 고려하지 않고 현재 보상만 고려하게 된다.
            target_values = reward + self.discount * next_q_values * done
            ##################################################################################

        # TODO(student): train the critic with the target values
        ##################################################################################
        qa_values = self.critic(obs)
        # 'qa_values'의 dim=1 즉, 각 배치 사이즈의 액션 차원에 대해 action.unsqueeze(1)이 제공하는 인덱스를 선택
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1) # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)
        ##################################################################################

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        ##################################################################################
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # Update the target network when needed
        if step % self.target_update_period == 0:
            self.update_target_critic()
        ##################################################################################
        return critic_stats
