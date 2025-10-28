#!/bin/bash
#
# HW5 모든 과제 실행 스크립트 (최종 수정본: PDF 방식 + 올바른 --dataset_dir 전달)
#

echo "===== HW5 BATCH SCRIPT START ====="
echo "===== PHASE 1: DATA GENERATION (Sec 3 & 4.3) ====="
echo "Launching all data generation jobs in parallel..."

# # --- 섹션 3.1 & 3.2 (변경 없음) ---
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_easy_random.yaml \
#   --dataset_dir datasets/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_random.yaml \
#   --dataset_dir datasets/ &
python3 cs285/scripts/run_hw5_explore.py \
  -cfg experiments/exploration/pointmass_hard_random.yaml \
  --dataset_dir datasets/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_easy_rnd.yaml \
#   --dataset_dir datasets/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_rnd.yaml \
#   --dataset_dir datasets/ &
python3 cs285/scripts/run_hw5_explore.py \
  -cfg experiments/exploration/pointmass_hard_rnd.yaml \
  --dataset_dir datasets/ &
  
# # --- 섹션 4.3: Data Ablation용 데이터 생성 (수정) ---
# # (YAML 파일 이름 확인: pointmass_medium_rnd_1k.yaml 등)
# echo "Launching Data Ablation (Generation)..."
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_rnd_1k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_1000/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_rnd_5k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_5000/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_rnd_10k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_10000/ &
# python3 cs285/scripts/run_hw5_explore.py \
#   -cfg experiments/exploration/pointmass_medium_rnd_20k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_20000/ &

echo "Waiting for ALL data generation (Phase 1) to complete..."
wait
echo "===== PHASE 1 COMPLETE ====="

# ---
#
# PHASE 2: [수정됨] 모든 작업에 --dataset_dir 인수를 정확히 전달
#
# ---

echo "===== PHASE 2: OFFLINE RL & FINETUNING (Sec 4 & 5) ====="
echo "Launching all training jobs in parallel..."

# # --- 섹션 4.1: 기본 DQN/CQL 실행 ---
# # (기본 'datasets/' 폴더를 사용)
# echo "Launching Section 4.1: DQN/CQL (Training)..."
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_easy_dqn.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_easy_cql.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_dqn.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql.yaml \
#   --dataset_dir datasets &
python3 cs285/scripts/run_hw5_offline.py \
  -cfg experiments/offline/pointmass_hard_dqn.yaml \
  --dataset_dir datasets &
python3 cs285/scripts/run_hw5_offline.py \
  -cfg experiments/offline/pointmass_hard_cql.yaml \
  --dataset_dir datasets &
  
# # --- 섹션 4.1: CQL Alpha Sweep (YAML 사용) ---
# # (알파 스윕은 기본 'datasets/' 폴더를 사용)
# echo "Launching Section 4.1: CQL Alpha Sweep (Training)..."
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_alpha_0.0.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_alpha_0.1.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_alpha_1.0.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_alpha_5.0.yaml \
#   --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_alpha_10.0.yaml \
#   --dataset_dir datasets &

# echo "Waiting for CQL Alpha Sweep jobs to complete..."
# wait

# # --- 섹션 4.2: IQL 및 AWAC (기본) ---
# # (모두 기본 'datasets/' 폴더를 사용)
# echo "Launching Section 4.2: IQL/AWAC..."
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_iql.yaml --dataset_dir datasets &
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_awac.yaml --dataset_dir datasets &
python3 cs285/scripts/run_hw5_offline.py \
  -cfg experiments/offline/pointmass_hard_iql.yaml --dataset_dir datasets &
python3 cs285/scripts/run_hw5_offline.py \
  -cfg experiments/offline/pointmass_hard_awac.yaml --dataset_dir datasets &

# echo "Waiting for Section 4.2 IQL/AWAC jobs to complete..."
# wait

# # --- 섹션 4.3: Data Ablation 학습 (YAML + --dataset_dir 사용) ---
# # (YAML 파일에 'exp_name'이, 여기서는 '--dataset_dir'이 지정됨)
# echo "Launching Section 4.3: Data Ablation (Training)..."
# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_1k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_1000/ &
# sleep 3

# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_5k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_5000/ &
# sleep 3

# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_10k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_10000/ &
# sleep 3

# python3 cs285/scripts/run_hw5_offline.py \
#   -cfg experiments/offline/pointmass_medium_cql_20k.yaml \
#   --dataset_dir datasets/medium_rnd_steps_20000/ &

# echo "Waiting for Section 4.3 Data Ablation jobs to complete..."
# wait

# --- 섹션 5: Online Fine-Tuning ---
# (기본 'datasets/' 폴더를 사용)
echo "Launching Section 5: Online Fine-Tuning (Hard)..."
python3 cs285/scripts/run_hw5_finetune.py \
  -cfg experiments/finetuning/pointmass_hard_cql_finetune.yaml \
  --use_reward \
  --dataset_dir datasets/finetune_runs

# 2단계(모든 학습)가 끝날 때까지 대기
echo "All training jobs (Phase 2) are launched. Waiting for completion..."
wait

echo "===== HW5 BATCH SCRIPT COMPLETE ====="