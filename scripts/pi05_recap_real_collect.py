#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from scripts.pi05_real_robot_infer import PI05RealRobotRunner, RobotConfig


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class EpisodeMeta:
    episode_index: int
    prompt: str
    success: bool
    num_frames: int
    control_hz: float
    robot_ip: str
    hand_port: str
    started_at: float
    ended_at: float


class ReCapRealCollector(PI05RealRobotRunner):
    def __init__(self, checkpoint_dir: str, train_config_name: str, prompt: str, robot_cfg: RobotConfig, output_dir: pathlib.Path):
        super().__init__(checkpoint_dir, train_config_name, prompt, robot_cfg)
        self.logger = _build_logger(self.__class__.__name__)
        self.output_dir = output_dir
        self.raw_dir = self.output_dir / 'raw_rollouts'
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_episode(self, episode_index: int, max_steps: int | None = None) -> pathlib.Path:
        self.logger.info('Starting episode %s. Press Ctrl+C to stop the whole collector after current save.', episode_index)
        frames = []
        step = 0
        started_at = time.time()
        while True:
            obs, joints, hand = self._get_obs()
            action = self.policy.infer(obs)['actions']
            if getattr(action, 'ndim', 1) > 1:
                action = action[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            frame = {
                'image': np.asarray(obs['image'], dtype=np.uint8),
                'wrist_image': np.asarray(obs['wrist_image'], dtype=np.uint8),
                'state': np.asarray(obs['state'], dtype=np.float32),
                'actions': action.astype(np.float32),
                'prompt': str(obs['prompt']),
                'timestamp': float(step / self.robot_cfg.control_hz),
            }
            frames.append(frame)
            self._apply_action(action, joints, hand)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
            time.sleep(self.dt)
        ended_at = time.time()
        success = input(f'Episode {episode_index} finished. Mark success? [y/N]: ').strip().lower() in {'y', 'yes'}
        ep_dir = self.raw_dir / f'episode_{episode_index:06d}'
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
            episode_index=episode_index,
            prompt=self.prompt,
            success=success,
            num_frames=len(frames),
            control_hz=self.robot_cfg.control_hz,
            robot_ip=self.robot_cfg.robot_ip,
            hand_port=self.robot_cfg.hand_port,
            started_at=started_at,
            ended_at=ended_at,
        )
        (ep_dir / 'meta.json').write_text(json.dumps(asdict(meta), indent=2))
        thumb = cv2.cvtColor(frames[0]['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_dir / 'thumb.png'), thumb)
        self.logger.info('Saved raw episode to %s (frames=%s success=%s)', ep_dir, len(frames), success)
        return ep_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect real-robot PI05 rollouts for ReCap.')
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--train-config', default='pi05_pickplace_full_pytorch')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_real_collect')
    parser.add_argument('--robot-ip', default='192.168.1.109')
    parser.add_argument('--hand-port', default='/dev/ttyUSB0')
    parser.add_argument('--control-hz', type=float, default=10.0)
    parser.add_argument('--arm-speed', type=float, default=0.1)
    parser.add_argument('--arm-acceleration', type=float, default=0.1)
    parser.add_argument('--episode-start-index', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = RobotConfig(
        robot_ip=args.robot_ip,
        hand_port=args.hand_port,
        control_hz=args.control_hz,
        arm_speed=args.arm_speed,
        arm_acceleration=args.arm_acceleration,
    )
    collector = ReCapRealCollector(args.checkpoint_dir, args.train_config, args.prompt, cfg, out)
    try:
        for ep in range(args.episode_start_index, args.episode_start_index + args.num_episodes):
            collector.collect_episode(ep, max_steps=args.max_steps)
    finally:
        try:
            collector.robot.stop()
        except Exception:
            pass
        for dev in (collector.hand, collector.head_camera, collector.wrist_camera):
            try:
                dev.close()
            except Exception:
                pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()
