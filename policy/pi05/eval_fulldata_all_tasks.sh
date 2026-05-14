#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH=/mnt/data/miniconda3/envs/cuda128/lib:/mnt/data/miniconda3/envs/cuda128/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

policy_name=pi05
task_config=${1:-demo_clean}
seed=${2:-0}
gpu_id=${3:-0}
ckpt_src=${4:-/mnt/data/zq/RoboTwin/policy/pi05/checkpoints/policy_ckpt/fulldata}
shard_index=${SHARD_INDEX:-0}
shard_count=${SHARD_COUNT:-1}

train_config_name=pi05_aloha_robotwin_mulitask_clean_wowrist
model_name=policy_ckpt
checkpoint_id=.
ckpt_setting=fulldata
test_num=3

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

TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

if [ "${shard_count}" -lt 1 ]; then
    echo "SHARD_COUNT must be >= 1, got: ${shard_count}" >&2
    exit 1
fi

if [ "${shard_index}" -lt 0 ] || [ "${shard_index}" -ge "${shard_count}" ]; then
    echo "SHARD_INDEX must be in [0, SHARD_COUNT), got: ${shard_index}/${shard_count}" >&2
    exit 1
fi

tasks=()
for idx in "${!TASKS[@]}"; do
    if [ $((idx % shard_count)) -eq "${shard_index}" ]; then
        tasks+=("${TASKS[$idx]}")
    fi
done

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mcheckpoint: ${ckpt_src}\033[0m"
echo -e "\033[33mtask config: ${task_config}, seed: ${seed}, shard: ${shard_index}/${shard_count}, tasks: ${#tasks[@]}/${#TASKS[@]}\033[0m"

cd "${repo_root}"

run_time="$(date '+%Y-%m-%d_%H-%M-%S')"
log_dir="${repo_root}/eval_result/pi05_fulldata_all_tasks_${run_time}_shard${shard_index}of${shard_count}"
mkdir -p "${log_dir}"
summary_file="${log_dir}/summary.txt"

success_count=0
fail_count=0

for task_name in "${tasks[@]}"; do
    echo "===== ${task_name} =====" | tee -a "${summary_file}"

    if PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
            --overrides \
            --task_name "${task_name}" \
            --task_config "${task_config}" \
            --train_config_name "${train_config_name}" \
            --model_name "${model_name}" \
            --checkpoint_id "${checkpoint_id}" \
            --ckpt_setting "${ckpt_setting}" \
            --seed "${seed}" \
            --policy_name "${policy_name}" \
            --test_num "${test_num}"; then
        echo "OK ${task_name}" | tee -a "${summary_file}"
        success_count=$((success_count + 1))
    else
        echo "FAIL ${task_name}" | tee -a "${summary_file}"
        fail_count=$((fail_count + 1))
    fi
done

echo "Done. success=${success_count}, fail=${fail_count}, logs=${log_dir}" | tee -a "${summary_file}"
