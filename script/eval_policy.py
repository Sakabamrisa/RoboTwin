import sys
import os
import subprocess
import json

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def load_seed_list(seed_file):
    with open(seed_file, "r", encoding="utf-8") as file:
        return [int(seed) for seed in file.read().split()]


def select_instruction(instruction_data, instruction_type):
    candidates = instruction_data.get(instruction_type, [])
    if candidates:
        return np.random.choice(candidates)

    for fallback_type in ("seen", "unseen"):
        candidates = instruction_data.get(fallback_type, [])
        if candidates:
            return np.random.choice(candidates)

    instruction = instruction_data.get("instruction")
    if instruction:
        return instruction

    raise ValueError(f"No valid instruction found for type: {instruction_type}")


def load_pregenerated_instruction(args, source_episode_idx, instruction_type):
    instruction_dir = args.get("instruction_dir")
    if instruction_dir is None:
        instruction_dir = os.path.join("data", args["task_name"], args["task_config"], "instructions")

    instruction_path = os.path.join(instruction_dir, f"episode{source_episode_idx}.json")
    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"Instruction file not found: {instruction_path}")

    with open(instruction_path, "r", encoding="utf-8") as file:
        instruction_data = json.load(file)

    return instruction_data, select_instruction(instruction_data, instruction_type)


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    runtime_arg_keys = (
        "expert_check",
        "target_fail_num",
        "target_success_num",
        "max_rollout_tries",
        "merge_rollout_hdf5",
        "save_failed_only",
        "seed_file",
        "instruction_dir",
        "output_dir",
        "test_num",
        "eval_video_log",
    )
    for key in runtime_arg_keys:
        if key in usr_args:
            args[key] = usr_args[key]

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    if "collect_wrist_camera" in usr_args:
        args["camera"]["collect_wrist_camera"] = as_bool(usr_args["collect_wrist_camera"])
    if "collect_head_camera" in usr_args:
        args["camera"]["collect_head_camera"] = as_bool(usr_args["collect_head_camera"])
    if "wrist_camera_type" in usr_args:
        args["camera"]["wrist_camera_type"] = usr_args["wrist_camera_type"]
    if "head_camera_type" in usr_args:
        args["camera"]["head_camera_type"] = usr_args["head_camera_type"]
    if "eval_video_log" in args:
        args["eval_video_log"] = as_bool(args["eval_video_log"])

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    output_dir = args.get("output_dir")
    if output_dir is None:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    else:
        save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rollout_save_dir = save_dir / "rollout_data"
    rollout_save_dir.mkdir(parents=True, exist_ok=True)
    args["save_data"] = True
    args["save_path"] = str(rollout_save_dir)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, eval_attempt_num, rollout_fail_count = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Attempts: {eval_attempt_num}\n")
        file.write(f"Success: {suc_num}\n")
        file.write(f"Fail: {rollout_fail_count}\n")
        if eval_attempt_num > 0:
            file.write(f"Success Rate: {suc_num / eval_attempt_num}\n")

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = as_bool(args.get("expert_check", False))
    merge_rollout_hdf5 = as_bool(args.get("merge_rollout_hdf5", False))
    save_failed_only = as_bool(args.get("save_failed_only", True))
    target_fail_num = int(args.get("target_fail_num", test_num))
    target_success_num = args.get("target_success_num", None)
    if target_success_num is not None:
        target_success_num = int(target_success_num)

    default_max_tries = max(test_num, target_fail_num) * 20
    max_rollout_tries = int(args.get("max_rollout_tries", default_max_tries))

    seed_file = args.get("seed_file")
    if seed_file is None and not expert_check:
        default_seed_file = os.path.join("data", args["task_name"], args["task_config"], "seed.txt")
        if os.path.exists(default_seed_file):
            seed_file = default_seed_file

    seed_list = []
    if seed_file is not None:
        seed_list = load_seed_list(seed_file)
        if not seed_list:
            raise ValueError(f"Seed file is empty: {seed_file}")
        print(f"\033[94mUsing seed file:\033[0m {seed_file} ({len(seed_list)} seeds)")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    rollout_success_count = 0
    rollout_fail_count = 0
    rollout_try_count = 0
    seed_attempt_count = 0
    seed_cursor = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    def target_reached():
        if rollout_fail_count < target_fail_num:
            return False
        if target_success_num is not None and rollout_success_count < target_success_num:
            return False
        return True

    while not target_reached():
        if seed_attempt_count >= max_rollout_tries:
            raise RuntimeError(
                f"Reached max_rollout_tries={max_rollout_tries} before target: "
                f"success={rollout_success_count}/{target_success_num}, "
                f"fail={rollout_fail_count}/{target_fail_num}"
            )

        if seed_list:
            if seed_cursor >= len(seed_list):
                raise RuntimeError(
                    f"Seed file exhausted before target: "
                    f"success={rollout_success_count}/{target_success_num}, "
                    f"fail={rollout_fail_count}/{target_fail_num}"
                )
            now_seed = seed_list[seed_cursor]
            source_episode_idx = seed_cursor
            seed_cursor += 1
        else:
            source_episode_idx = seed_attempt_count

        seed_attempt_count += 1
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        episode_info = None
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                import traceback

                print("\n========== error occurs ==========", flush=True)
                print("Error:", repr(e), flush=True)
                traceback.print_exc()

                try:
                    TASK_ENV.close_env()
                except Exception:
                    print("close_env also failed:", flush=True)
                    traceback.print_exc()

                args["render_freq"] = render_freq
                raise
            # except Exception as e:
            #     # stack_trace = traceback.format_exc()
            #     # print(" -------------")
            #     # print("Error: ", e)
            #     # print(" -------------")
            #     TASK_ENV.close_env()
            #     now_seed += 1
            #     args["render_freq"] = render_freq
            #     print("error occurs !")
            #     continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        instructions_dir = os.path.join(args["save_path"], "instructions")
        os.makedirs(instructions_dir, exist_ok=True)

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except UnStableError as e:
            TASK_ENV.close_env()
            print(f"\033[93mSkip unstable rollout seed {now_seed}: {e}\033[0m")
            now_seed += 1
            continue

        if expert_check:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            if not results:
                raise RuntimeError(f"No instruction generated for seed: {now_seed}")
            instruction_data = results[0]
            instruction = select_instruction(instruction_data, instruction_type)
        else:
            instruction_data, instruction = load_pregenerated_instruction(
                args, source_episode_idx, instruction_type)

        with open(os.path.join(instructions_dir, f"episode{now_id}.json"), "w", encoding="utf-8") as file:
            json.dump(instruction_data, file, ensure_ascii=False, indent=2)

        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if args.get("save_data", False) and hasattr(TASK_ENV, "folder_path"):
            if merge_rollout_hdf5:
                TASK_ENV.merge_pkl_to_hdf5_video()
                TASK_ENV.remove_data_cache()
            elif succ and save_failed_only:
                TASK_ENV.remove_data_cache()
        elif args.get("save_data", False):
            print("\033[93mWarning: no rollout frames were saved for this episode.\033[0m")

        if succ:
            rollout_success_count += 1
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            rollout_fail_count += 1
            print("\033[91mFail!\033[0m")

        rollout_try_count += 1
        keep_episode = (not succ) or (not save_failed_only)
        if keep_episode:
            now_id += 1

        TASK_ENV.close_env(clear_cache=((TASK_ENV.test_num + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"fail target: \033[96m{rollout_fail_count}/{target_fail_num}\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, TASK_ENV.test_num, rollout_fail_count


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
