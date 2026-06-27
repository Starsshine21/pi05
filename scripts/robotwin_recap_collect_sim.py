#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class EpisodeMeta:
    episode_index: int
    prompt: str
    success: bool
    num_frames: int
    task_name: str
    task_config: str
    train_config: str
    model_name: str
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect RoboTwin sim rollouts for ReCap-style offline processing.')
    parser.add_argument('--task-name', default='adjust_bottle')
    parser.add_argument('--task-config', default='demo_clean')
    parser.add_argument('--train-config-name', default='pi05_aloha_full_base')
    parser.add_argument('--model-name', default='model_robotwin')
    parser.add_argument('--checkpoint-id', default='30000')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--output-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    root = Path('/nfs_global/S/yangrongzheng/pi05/external/RoboTwin').resolve()
    os.chdir(root)
    import sys
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'script'))
    sys.path.insert(0, str(root / 'policy' / 'pi05' / 'src'))
    sys.path.insert(0, str(root / 'policy' / 'pi05' / 'packages' / 'openpi-client' / 'src'))

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from script.eval_policy import class_decorator, eval_function_decorator, get_embodiment_config

    gei_path = root / 'description' / 'utils' / 'generate_episode_instructions.py'
    spec = importlib.util.spec_from_file_location('robotwin_generate_episode_instructions', gei_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {gei_path}')
    gei = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gei)
    generate_episode_descriptions = gei.generate_episode_descriptions

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    raw_dir = output_dir / 'raw_rollouts'
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(root / 'task_config' / f'{args.task_config}.yml', 'r', encoding='utf-8') as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    task_args['task_name'] = args.task_name
    task_args['task_config'] = args.task_config
    task_args['ckpt_setting'] = args.model_name
    task_args['policy_name'] = 'pi05'
    task_args['eval_mode'] = True
    task_args['render_freq'] = 0
    task_args['eval_video_save_dir'] = None

    embodiment_type = task_args.get('embodiment')
    embodiment_config_path = Path(CONFIGS_PATH) / '_embodiment_config.yml'
    with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb):
        robot_file = embodiment_types[emb]['file_path']
        if robot_file is None:
            raise RuntimeError('No embodiment files')
        return robot_file

    if len(embodiment_type) == 1:
        task_args['left_robot_file'] = get_embodiment_file(embodiment_type[0])
        task_args['right_robot_file'] = get_embodiment_file(embodiment_type[0])
        task_args['dual_arm_embodied'] = True
    elif len(embodiment_type) == 3:
        task_args['left_robot_file'] = get_embodiment_file(embodiment_type[0])
        task_args['right_robot_file'] = get_embodiment_file(embodiment_type[1])
        task_args['embodiment_dis'] = embodiment_type[2]
        task_args['dual_arm_embodied'] = False
    else:
        raise RuntimeError('embodiment items should be 1 or 3')

    task_args['left_embodiment_config'] = get_embodiment_config(task_args['left_robot_file'])
    task_args['right_embodiment_config'] = get_embodiment_config(task_args['right_robot_file'])

    task_env = class_decorator(args.task_name)
    get_model = eval_function_decorator('pi05', 'get_model')
    reset_model = eval_function_decorator('pi05', 'reset_model')
    eval_func = eval_function_decorator('pi05', 'eval')

    usr_args = {
        'task_name': args.task_name,
        'task_config': args.task_config,
        'ckpt_setting': args.model_name,
        'policy_name': 'pi05',
        'instruction_type': 'unseen',
        'train_config_name': args.train_config_name,
        'model_name': args.model_name,
        'checkpoint_id': args.checkpoint_id,
        'pi0_step': 10,
        'seed': args.seed,
    }
    model = get_model(usr_args)

    now_seed = 100000 * (1 + args.seed)
    collected = 0
    while collected < args.num_episodes:
        try:
            task_env.setup_demo(now_ep_num=collected, seed=now_seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            success = bool(task_env.plan_success and task_env.check_success())
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            continue
        except Exception:
            task_env.close_env()
            raise

        if not success:
            now_seed += 1
            continue

        task_env.setup_demo(now_ep_num=collected, seed=now_seed, is_test=True, **task_args)
        episode_info_list = [episode_info['info']]
        results = generate_episode_descriptions(task_args['task_name'], episode_info_list, 1)
        prompt = np.random.choice(results[0]['unseen'])
        task_env.set_instruction(prompt)

        frames = []
        reset_model(model)
        while task_env.take_action_cnt < task_env.step_lim:
            obs = task_env.get_obs()
            frame = {
                'image': np.asarray(obs['observation']['head_camera']['rgb'], dtype=np.uint8),
                'wrist_image': np.asarray(obs['observation']['right_camera']['rgb'], dtype=np.uint8),
                'state': np.asarray(obs['joint_action']['vector'], dtype=np.float32),
                'prompt': str(prompt),
                'timestamp': float(task_env.take_action_cnt / 10.0),
            }
            eval_func(task_env, model, obs)
            action = getattr(task_env, 'action', None)
            if action is None:
                action = np.zeros((frame['state'].shape[0],), dtype=np.float32)
            frame['actions'] = np.asarray(action, dtype=np.float32).reshape(-1)
            frames.append(frame)
            if task_env.eval_success:
                break

        final_success = bool(task_env.eval_success)
        ep_dir = raw_dir / f'episode_{collected:06d}'
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ep_dir / 'frames.npz',
            image=np.stack([f['image'] for f in frames], axis=0),
            wrist_image=np.stack([f['wrist_image'] for f in frames], axis=0),
            state=np.stack([f['state'] for f in frames], axis=0),
            actions=np.stack([f['actions'] for f in frames], axis=0),
            timestamp=np.asarray([f['timestamp'] for f in frames], dtype=np.float32),
        )
        meta = EpisodeMeta(
            episode_index=collected,
            prompt=prompt,
            success=final_success,
            num_frames=len(frames),
            task_name=args.task_name,
            task_config=args.task_config,
            train_config=args.train_config_name,
            model_name=args.model_name,
            seed=now_seed,
        )
        (ep_dir / 'meta.json').write_text(json.dumps(asdict(meta), indent=2))
        thumb = cv2.cvtColor(frames[0]['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_dir / 'thumb.png'), thumb)
        print(f'Saved {ep_dir} success={final_success} frames={len(frames)}', flush=True)
        task_env.close_env(clear_cache=((collected + 1) % 10 == 0))
        collected += 1
        now_seed += 1

    print(raw_dir)


if __name__ == '__main__':
    main()
