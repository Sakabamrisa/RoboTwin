#!/bin/bash
set -euo pipefail

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

policy_name=pi05
task_name=${1:-adjust_bottle}
task_config=${2:-demo_clean}
seed=${3:-0}
gpu_id=${4:-0}
ckpt_src=${5:-/mnt/data/zq/RoboTwin/policy/pi05/checkpoints/policy_ckpt/fulldata}

train_config_name=pi05_aloha_robotwin_mulitask_clean_wowrist
model_name=policy_ckpt
checkpoint_id=.
ckpt_setting=fulldata

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [ ! -d "${ckpt_src}" ]; then
    echo "Checkpoint directory not found: ${ckpt_src}" >&2
    exit 1
fi

if [ ! -f "${ckpt_src}/model.safetensors" ]; then
    echo "model.safetensors not found in: ${ckpt_src}" >&2
    exit 1
fi

link_parent="${repo_root}/policy/pi05/checkpoints/${train_config_name}"
link_path="${link_parent}/${model_name}"
mkdir -p "${link_parent}"

if [ -L "${link_path}" ]; then
    ln -sfn "${ckpt_src}" "${link_path}"
elif [ -e "${link_path}" ]; then
    echo "Path exists and is not a symlink: ${link_path}" >&2
    exit 1
else
    ln -s "${ckpt_src}" "${link_path}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mcheckpoint: ${ckpt_src}\033[0m"

cd "${repo_root}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --checkpoint_id "${checkpoint_id}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
