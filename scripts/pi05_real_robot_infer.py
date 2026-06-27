#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from openpi.policies import policy_config as policy_config_lib
from openpi.training import config as train_config_lib


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


class UR5eRTDE:
    def __init__(self, robot_ip: str, acceleration: float = 0.1, speed: float = 0.1):
        import rtde_control
        import rtde_receive

        self.robot_ip = str(robot_ip)
        self.acceleration = float(acceleration)
        self.speed = float(speed)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)

    def get_pos_j(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)

    def get_pos_eef(self) -> np.ndarray:
        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float32)

    def set_pos_j(self, target_qpos, servo: bool = True):
        target_qpos = np.asarray(target_qpos, dtype=np.float64).tolist()
        if servo:
            self.rtde_c.servoJ(target_qpos, self.speed, self.acceleration, 1 / 500, 0.1, 300)
        else:
            self.rtde_c.moveJ(target_qpos, self.speed, self.acceleration)

    def stop(self):
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass


class InspireHandSerial:
    REGDICT = {"posSet": 1474, "posAct": 1534}

    def __init__(self, port: str, baudrate: int = 115200, hand_id: int = 1):
        import serial

        self.serial_mod = serial
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.hand_id = int(hand_id)
        self.ser = None

    def open(self):
        self.ser = self.serial_mod.Serial()
        self.ser.port = self.port
        self.ser.baudrate = self.baudrate
        self.ser.open()

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def _write_register(self, add: int, num: int, values):
        packet = [0xEB, 0x90, self.hand_id, num + 3, 0x12, add & 0xFF, (add >> 8) & 0xFF]
        packet.extend(int(v) & 0xFF for v in values)
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        self.ser.write(packet)
        time.sleep(0.01)
        self.ser.read_all()

    def _read_register(self, add: int, num: int):
        packet = [0xEB, 0x90, self.hand_id, 0x04, 0x11, add & 0xFF, (add >> 8) & 0xFF, num]
        checksum = sum(packet[2:]) & 0xFF
        packet.append(checksum)
        self.ser.write(packet)
        time.sleep(0.01)
        recv = self.ser.read_all()
        data_len = (recv[3] & 0xFF) - 3
        return [recv[7 + i] for i in range(data_len)] if len(recv) else []

    def set_hand_pos(self, value):
        payload = []
        for item in value:
            item = int(np.clip(item, 0, 2000))
            payload.append(item & 0xFF)
            payload.append((item >> 8) & 0xFF)
        self._write_register(self.REGDICT["posSet"], 12, payload)

    def get_hand_pos(self) -> np.ndarray:
        raw = self._read_register(self.REGDICT["posAct"], 12)
        if len(raw) < 12:
            raise RuntimeError("No valid response while reading inspire hand state.")
        vals = [int((raw[2 * i] & 0xFF) + (raw[2 * i + 1] << 8)) for i in range(6)]
        return np.asarray(vals, dtype=np.float32)


class L515ColorCamera:
    def __init__(self):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def get_data(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if frame is None:
            raise RuntimeError("Failed to read L515 color frame.")
        return np.asanyarray(frame.get_data())

    def close(self):
        self.pipeline.stop()


class OrbbecFemtoBoltColorCamera:
    def __init__(self):
        import pyorbbecsdk as sdk

        self.sdk = sdk
        self.config = sdk.Config()
        self.pipeline = sdk.Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(1280, 720, sdk.OBFormat.BGR, 30)
        self.config.enable_stream(color_profile)
        self.pipeline.start(self.config)

    def get_data(self) -> np.ndarray:
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            frame = frames.get_color_frame()
            if frame is None:
                continue
            width = frame.get_width()
            height = frame.get_height()
            data = np.asanyarray(frame.get_data())
            return data.reshape((height, width, 3))

    def close(self):
        self.pipeline.stop()


@dataclass
class RobotConfig:
    robot_ip: str = "192.168.1.109"
    hand_port: str = "/dev/ttyUSB0"
    control_hz: float = 10.0
    arm_speed: float = 0.1
    arm_acceleration: float = 0.1
    convert_bgr_to_rgb: bool = True
    joint_delta_scale: float = 0.35
    hand_delta_scale: float = 0.35
    max_joint_delta: float = 0.03
    max_hand_delta: float = 120.0
    action_ema_alpha: float = 0.25
    record_video: bool = True
    record_dir: str = "real_robot_demos"
    record_fps: float = 10.0
    max_steps: int = 0


class PI05RealRobotRunner:
    def __init__(self, checkpoint_dir: str, train_config_name: str, prompt: str, robot_cfg: RobotConfig):
        self.logger = _build_logger(self.__class__.__name__)
        self.prompt = prompt
        self.robot_cfg = robot_cfg
        self.robot = UR5eRTDE(robot_cfg.robot_ip, acceleration=robot_cfg.arm_acceleration, speed=robot_cfg.arm_speed)
        self.hand = InspireHandSerial(robot_cfg.hand_port)
        self.hand.open()
        self.head_camera = OrbbecFemtoBoltColorCamera()
        self.wrist_camera = L515ColorCamera()

        train_cfg = train_config_lib._CONFIGS_DICT[train_config_name]
        self.policy = policy_config_lib.create_trained_policy(train_cfg, pathlib.Path(checkpoint_dir), default_prompt=prompt)
        self.dt = 1.0 / robot_cfg.control_hz
        self.prev_action: np.ndarray | None = None
        self.record_dir = pathlib.Path(robot_cfg.record_dir)
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.record_frame_idx = 0

    def _make_video_writer(self, path: pathlib.Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {path}")
        return writer

    def _ensure_video_writers(self, wrist_frame: np.ndarray, head_frame: np.ndarray):
        if self.video_writers or not self.robot_cfg.record_video:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.record_dir / timestamp
        self.video_writers["wrist"] = self._make_video_writer(run_dir / "wrist.mp4", wrist_frame.shape, self.robot_cfg.record_fps)
        self.video_writers["head"] = self._make_video_writer(run_dir / "head.mp4", head_frame.shape, self.robot_cfg.record_fps)
        combo = np.concatenate([head_frame, cv2.resize(wrist_frame, (head_frame.shape[1], head_frame.shape[0]))], axis=1)
        self.video_writers["combined"] = self._make_video_writer(run_dir / "combined.mp4", combo.shape, self.robot_cfg.record_fps)
        self.logger.info("Recording demo videos under %s", run_dir)

    def _write_video_frames(self, wrist_frame: np.ndarray, head_frame: np.ndarray):
        if not self.robot_cfg.record_video:
            return
        self._ensure_video_writers(wrist_frame, head_frame)
        combo = np.concatenate([head_frame, cv2.resize(wrist_frame, (head_frame.shape[1], head_frame.shape[0]))], axis=1)
        self.video_writers["wrist"].write(wrist_frame)
        self.video_writers["head"].write(head_frame)
        self.video_writers["combined"].write(combo)
        self.record_frame_idx += 1

    def _resize_rgb(self, image: np.ndarray, shape=(224, 224)) -> np.ndarray:
        image = np.asarray(image)
        if self.robot_cfg.convert_bgr_to_rgb:
            image = image[..., ::-1]
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def _get_obs(self):
        joints = self.robot.get_pos_j().astype(np.float32)
        eef = self.robot.get_pos_eef().astype(np.float32)
        hand = self.hand.get_hand_pos().astype(np.float32)
        wrist_frame = self.wrist_camera.get_data()
        head_frame = self.head_camera.get_data()
        self._write_video_frames(wrist_frame, head_frame)
        state = np.concatenate([joints, eef, hand], axis=0)
        return {
            "image": self._resize_rgb(wrist_frame),
            "wrist_image": self._resize_rgb(head_frame),
            "state": state,
            "prompt": self.prompt,
        }, joints, hand

    def _smooth_and_limit_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 12:
            raise ValueError(f"Expected at least 12-D action, got {action.shape}.")

        joint_delta = action[:6] * float(self.robot_cfg.joint_delta_scale)
        hand_delta = action[6:12] * float(self.robot_cfg.hand_delta_scale)

        joint_delta = np.clip(joint_delta, -self.robot_cfg.max_joint_delta, self.robot_cfg.max_joint_delta)
        hand_delta = np.clip(hand_delta, -self.robot_cfg.max_hand_delta, self.robot_cfg.max_hand_delta)

        filtered_action = np.concatenate([joint_delta, hand_delta], axis=0)
        if self.prev_action is None:
            self.prev_action = filtered_action
        else:
            alpha = float(np.clip(self.robot_cfg.action_ema_alpha, 0.0, 1.0))
            self.prev_action = alpha * filtered_action + (1.0 - alpha) * self.prev_action
        return self.prev_action.copy()

    def _apply_action(self, action: np.ndarray, current_joints: np.ndarray, current_hand: np.ndarray):
        filtered_action = self._smooth_and_limit_action(action)
        joint_delta = filtered_action[:6]
        hand_delta = filtered_action[6:12]
        target_joints = current_joints + joint_delta
        target_hand = current_hand + hand_delta
        self.logger.info(
            "joint_delta=%s hand_delta=%s",
            np.array2string(joint_delta, precision=4, floatmode="fixed"),
            np.array2string(hand_delta, precision=1, floatmode="fixed"),
        )
        self.robot.set_pos_j(target_joints, servo=True)
        self.hand.set_hand_pos(np.rint(np.clip(target_hand, 0, 2000)).astype(np.int32).tolist())

    def run(self):
        self.logger.info("Press Ctrl+C to stop. Starting real-robot rollout loop.")
        step_idx = 0
        try:
            while True:
                if self.robot_cfg.max_steps > 0 and step_idx >= self.robot_cfg.max_steps:
                    self.logger.info("Reached max_steps=%d, stopping rollout.", self.robot_cfg.max_steps)
                    break
                obs, joints, hand = self._get_obs()
                action = self.policy.infer(obs)["actions"]
                if action.ndim > 1:
                    action = action[0]
                self._apply_action(action, joints, hand)
                step_idx += 1
                time.sleep(self.dt)
        finally:
            try:
                self.robot.stop()
            except Exception:
                pass
            for dev in (self.hand, self.head_camera, self.wrist_camera):
                try:
                    dev.close()
                except Exception:
                    pass
            for writer in self.video_writers.values():
                try:
                    writer.release()
                except Exception:
                    pass
            if self.video_writers:
                self.logger.info("Saved %d recorded frames to %s", self.record_frame_idx, self.record_dir)


def main():
    parser = argparse.ArgumentParser(description="Run a trained PI05 checkpoint on the real robot.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-config", default="pi05_pickplace_full_pytorch")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--robot-ip", default="192.168.1.109")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--arm-speed", type=float, default=0.1)
    parser.add_argument("--arm-acceleration", type=float, default=0.1)
    parser.add_argument("--joint-delta-scale", type=float, default=0.35)
    parser.add_argument("--hand-delta-scale", type=float, default=0.35)
    parser.add_argument("--max-joint-delta", type=float, default=0.03)
    parser.add_argument("--max-hand-delta", type=float, default=120.0)
    parser.add_argument("--action-ema-alpha", type=float, default=0.25)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-dir", default="real_robot_demos")
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    cfg = RobotConfig(
        robot_ip=args.robot_ip,
        hand_port=args.hand_port,
        control_hz=args.control_hz,
        arm_speed=args.arm_speed,
        arm_acceleration=args.arm_acceleration,
        joint_delta_scale=args.joint_delta_scale,
        hand_delta_scale=args.hand_delta_scale,
        max_joint_delta=args.max_joint_delta,
        max_hand_delta=args.max_hand_delta,
        action_ema_alpha=args.action_ema_alpha,
        record_video=args.record_video,
        record_dir=args.record_dir,
        record_fps=args.record_fps,
        max_steps=args.max_steps,
    )
    runner = PI05RealRobotRunner(args.checkpoint_dir, args.train_config, args.prompt, cfg)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
