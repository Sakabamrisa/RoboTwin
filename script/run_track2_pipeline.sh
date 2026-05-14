#!/bin/bash
set -euo pipefail

repo_root=/mnt/data/zq/RoboTwin
task_name=${TASK_NAME:-${5:-adjust_bottle}}
task_config=${TASK_CONFIG:-demo_clean}
gpu_id=${1:-0}
seed=${2:-0}
output_dir=${OUTPUT_DIR:-${4:-${repo_root}/eval_result/${task_name}_min_pipeline_gpu${gpu_id}}}
run_collect=${RUN_COLLECT:-0}

dataset_root=${DATASET_ROOT:-/mnt/data/cyx/datasets/robotwin/aloha_tars}
if [[ -n "${COLLECTED_DATA_DIR:-}" ]]; then
    collected_data_dir="${COLLECTED_DATA_DIR}"
else
    case "${task_config}" in
        demo_clean)
            dataset_variant=${DATASET_VARIANT:-aloha-agilex_clean_50}
            ;;
        demo_randomized)
            dataset_variant=${DATASET_VARIANT:-aloha-agilex_randomized_500}
            ;;
        *)
            dataset_variant=${DATASET_VARIANT:-${task_config}}
            ;;
    esac
    collected_data_dir="${dataset_root}/${task_name}/${dataset_variant}"
fi
seed_file=${SEED_FILE:-${collected_data_dir}/seed.txt}
instruction_dir=${INSTRUCTION_DIR:-${collected_data_dir}/instructions}

policy_name=pi05
train_config_name=pi05_aloha_robotwin_mulitask_clean_wowrist
model_name=policy_ckpt
checkpoint_id=.
ckpt_setting=10data
policy_host=${POLICY_HOST:-127.0.0.1}
policy_port=${POLICY_PORT:-8000}

target_fail_num=${TARGET_FAIL_NUM:-1} #目标失败数量
max_rollout_tries=${MAX_ROLLOUT_TRIES:-20}

export LD_LIBRARY_PATH=/share_storages/miniconda3/envs/cuda128/lib:/share_storages/miniconda3/envs/cuda128/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd "${repo_root}"

if [[ "${run_collect}" == "1" ]]; then
    bash collect_data.sh "${task_name}" "${task_config}" "${gpu_id}"
else
    echo "[pipeline] Skip collect_data.sh; using existing data from ${collected_data_dir}"
fi

if [[ ! -f "${seed_file}" ]]; then
    echo "[pipeline] Missing seed file: ${seed_file}" >&2
    exit 1
fi

if [[ ! -d "${instruction_dir}" ]]; then
    echo "[pipeline] Missing instruction dir: ${instruction_dir}" >&2
    exit 1
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --host "${policy_host}" \
    --port "${policy_port}" \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --train_config_name "${train_config_name}" \
    --model_name "${model_name}" \
    --checkpoint_id "${checkpoint_id}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --test_num "${target_fail_num}" \
    --target_fail_num "${target_fail_num}" \
    --max_rollout_tries "${max_rollout_tries}" \
    --seed_file "${seed_file}" \
    --instruction_dir "${instruction_dir}" \
    --expert_check False \
    --collect_wrist_camera False \
    --merge_rollout_hdf5 False \
    --save_failed_only True \
    --eval_video_log False \
    --output_dir "${output_dir}"

rollout_dir="${output_dir}/rollout_data"

python script/convert_rollout_pkl_to_hdf5_video.py "${rollout_dir}" --remove-cache --stride 6

source /share_storages/miniconda3/etc/profile.d/conda.sh
conda activate flowwam
python script/export_robot_only_hdf5.py "${rollout_dir}" --render-resolution 320 240
