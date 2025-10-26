import yaml
import os
import time

import cs285.env_configs
from cs285.infrastructure.logger import Logger
# ----------------------------------------------------------------------
import importlib
# ----------------------------------------------------------------------

def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    base_config_name = config_kwargs.pop("base_config")
    return cs285.env_configs.configs[base_config_name](**config_kwargs)

# # 기존 make_config 함수를 아래 코드로 교체
# def make_config(config_file: str) -> dict:
#     # 1. 실험 YAML 파일 로드
#     with open(config_file, "r") as f:
#         # YAML 로더 변경 (원본 코드 방식)
#         config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

#     # 2. YAML 파일에서 log_name 값을 미리 저장 (덮어쓰기 위해)
#     user_log_name = config_kwargs.get("log_name", None)

#     # 3. base_config 이름 가져오기 (pop으로 제거)
#     if "base_config" not in config_kwargs:
#         raise ValueError("Config file must specify 'base_config'")
#     base_config_name = config_kwargs.pop("base_config")

#     # 4. base_config 함수 찾기
#     if base_config_name not in cs285.env_configs.configs:
#         raise ValueError(f"Unknown base_config '{base_config_name}'")
#     base_config_func = cs285.env_configs.configs[base_config_name]

#     # 5. base_config 함수를 *호출*하여 기본 설정 딕셔너리 생성 (원본 코드 방식)
#     #    YAML에서 읽은 값들 중 base_config 함수가 필요로 하는 인자들을 전달
#     #    (pop으로 제거되지 않은 나머지 값들이 전달됨)
#     final_config = base_config_func(**config_kwargs) # <-- 원본처럼 **kwargs 사용!

#     # 6. YAML 파일에 log_name이 있었다면, base_config가 설정한 log_name을 덮어쓰기 (핵심!)
#     if user_log_name is not None:
#         final_config["log_name"] = user_log_name
#     # 만약 YAML에 log_name이 없고 base_config에도 없었다면 기본값 생성
#     elif "log_name" not in final_config:
#         # 이 부분은 base_config가 log_name을 항상 정의하므로 거의 실행되지 않음
#         log_name = "{env_name}_{agent}_{timestamp}".format(
#             env_name=final_config.get("env_name", "UnknownEnv"),
#             agent=final_config.get("agent", "UnknownAgent"),
#             timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
#         )
#         final_config["log_name"] = log_name

#     # agent_kwargs 오버라이드 처리는 base_config_func(**config_kwargs) 호출 시
#     # 내부적으로 처리되거나, 혹은 base_config 함수 자체가 agent.* 키를 무시할 수 있음
#     # (hw5의 base_config들은 agent.* 처리가 필요 없어 보임)
#     # 따라서 이전 턴의 agent_kwargs 처리 로직은 불필요

#     return final_config

def make_logger(logdir_prefix: str, config: dict) -> Logger:
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    
    logdir = (
        logdir_prefix + config["log_name"] + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    return Logger(logdir)
