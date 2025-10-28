import pickle
import time
import argparse

from cs285.agents.dqn_agent import DQNAgent
import cs285.env_configs
from cs285.envs import Pointmass

import os
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.agents import agents
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer

from scripting_utils import make_logger, make_config

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()  # 학습에 사용된 GYM 환경(ex. Pointmass) 생성
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"
    
    # DQNAgent, CQLAgent, IQLAgent, AWACAgent 클래스 가져옴
    agent_cls = agents[config["agent"]] 
    # CQLAgent 객체를 생성합니다. 이때 YAML 파일에 정의된 agent_kwargs
    # (예: cql_alpha, hidden_size, learning_rate 등)이 에이전트 생성자에 전달됩니다.
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    with open(os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl"), "rb") as f:
        # exploration method에 따라 수집된 오프라인 데이터셋(pkl파일, config['dataset_name']) 로드
        # 이 dataset 객체는 ReplayBuffer 클래스의 인스턴스이며, 수집한 모든 (s,a,r,s',dones)튜플들을 담고 있습니다.
        dataset = pickle.load(f)

    for step in tqdm.trange(config["training_steps"], dynamic_ncols=True):
        # Train with offline RL
        batch = dataset.sample(config["batch_size"])    # replay buffer에서 batch_size만큼 샘플링

        ###########################################################
        # 데이터 shape 디버깅 출력. shape 확인하기 위함.
        if step == 0:
            print({k: tuple(v.shape) for k, v in batch.items()})
        ###########################################################
        
        # NumPy 배열에서 PyTorch 텐서로 변환
        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        # XX-Agent의 update 메서드를 호출하여 에이전트의 신경망 파라미터를 업데이트
        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            # print(f"Logging metrics at step {step}")  # 디버깅 출력
            for k, v in metrics.items():    # agent.update가 반환한 metrics (TD 손실, CQL 손실 등)를 로거를 통해 기록
                logger.log_scalar(v, k, step)
                # print(f"Logged {k}: {v}")  # 로그 기록 후 출력

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            env_pointmass: Pointmass = env.unwrapped
            logger.log_figures(
                [env_pointmass.plot_trajectory(trajectory["next_observation"]) for trajectory in trajectories],
                "trajectories",
                step,
                "eval"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw5_offline_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
