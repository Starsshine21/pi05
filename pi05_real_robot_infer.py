#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self._last_pos = np.zeros(6, dtype=np.float32)

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
        for _ in range(3):
            self.ser.write(packet)
            time.sleep(0.01)
            recv = self.ser.read_all()
            if len(recv) >= 7:
                data_len = max(0, (recv[3] & 0xFF) - 3)
                if len(recv) >= 7 + data_len:
                    return [recv[7 + i] for i in range(data_len)]
        return []

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
            return self._last_pos.copy()
        vals = np.asarray([int((raw[2 * i] & 0xFF) + (raw[2 * i + 1] << 8)) for i in range(6)], dtype=np.float32)
        self._last_pos = vals.copy()
        return vals


class L515ColorCamera:
    def __init__(self):
        try:
            import pyrealsense2.pyrealsense2 as rs
        except Exception:
            import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames()
            except Exception:
                pass

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


class MockColorCamera:
    def __init__(self, height: int = 480, width: int = 640, value: int = 0):
        self.height = int(height)
        self.width = int(width)
        self.value = int(value)

    def get_data(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), self.value, dtype=np.uint8)

    def close(self):
        pass




class BridgeCommandColorCamera:
    def __init__(self, command: list[str], image_path: str):
        self.command = [str(x) for x in command]
        self.image_path = pathlib.Path(image_path)
        self._last_image: np.ndarray | None = None

    def get_data(self) -> np.ndarray:
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(self.command + ["--output", str(self.image_path)], check=True)
            image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None and self.image_path.exists():
            image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None and self._last_image is not None:
            return self._last_image.copy()
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._last_image = image.copy()
        return image

    def close(self):
        pass


@dataclass
class RobotConfig:
    robot_ip: str = "192.168.1.109"
    hand_port: str = "/dev/ttyUSB0"
    control_hz: float = 10.0
    arm_speed: float = 0.1
    arm_acceleration: float = 0.1
    convert_bgr_to_rgb: bool = True
    mock_cameras: bool = False
    bridge_l515: bool = False
    joint_scale: float = 10.0
    hand_scale: float = 1.0
    max_eef_pos_delta: float = 0.01
    max_eef_rot_delta: float = 0.08
    max_hand_delta: float = 80.0
    disable_position_delta: bool = False
    disable_rotation_delta: bool = False
    disable_hand_delta: bool = False
    swap_cameras: bool = False
    log_action_details: bool = False
    state_mode: str = "joint_eef_hand"
    action_mode: str = "eef_delta"
    save_obs_dir: str | None = "./pi05_obs"
    save_obs_every: int = 1


class PI05RealRobotRunner:
    def __init__(self, checkpoint_dir: str, train_config_name: str, prompt: str, robot_cfg: RobotConfig):
        self.logger = _build_logger(self.__class__.__name__)
        self.prompt = prompt
        self.robot_cfg = robot_cfg
        self.obs_step = 0
        self.save_obs_dir = pathlib.Path(robot_cfg.save_obs_dir) if robot_cfg.save_obs_dir else None
        if self.save_obs_dir is not None:
            self.save_obs_dir.mkdir(parents=True, exist_ok=True)
        self.robot = UR5eRTDE(robot_cfg.robot_ip, acceleration=robot_cfg.arm_acceleration, speed=robot_cfg.arm_speed)
        self.hand = InspireHandSerial(robot_cfg.hand_port)
        self.hand.open()
        if robot_cfg.mock_cameras:
            self.logger.warning("Using mock cameras. Real image devices are disabled.")
            self.head_camera = MockColorCamera(value=64)
            self.wrist_camera = MockColorCamera(value=128)
        else:
            self.head_camera = OrbbecFemtoBoltColorCamera()
            try:
                self.wrist_camera = L515ColorCamera()
                if robot_cfg.bridge_l515:
                    self.logger.warning("`--bridge-l515` specified, but direct L515 succeeded; using direct L515 camera.")
            except Exception as exc:
                self.logger.warning("Direct L515 open failed: %s", exc)
                self.logger.warning("Falling back to bridged L515 capture command.")
                self.wrist_camera = BridgeCommandColorCamera(
                    [
                        "/home/iprc-dex/miniconda3/envs/dp3/bin/python",
                        str(pathlib.Path(__file__).resolve().parent / "grab_l515_frame_dp3.py"),
                    ],
                    "/tmp/pi05_l515_bridge.png",
                )

        train_cfg = train_config_lib._CONFIGS_DICT[train_config_name]
        self.policy = policy_config_lib.create_trained_policy(train_cfg, pathlib.Path(checkpoint_dir), default_prompt=prompt)
        self.dt = 1.0 / robot_cfg.control_hz

    def _resize_rgb(self, image: np.ndarray, shape=(224, 224)) -> np.ndarray:
        image = np.asarray(image)
        if self.robot_cfg.convert_bgr_to_rgb:
            image = image[..., ::-1]
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def _save_obs_images(self, head_bgr: np.ndarray, wrist_bgr: np.ndarray) -> None:
        if self.save_obs_dir is None:
            return
        if self.obs_step % max(1, self.robot_cfg.save_obs_every) != 0:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(self.save_obs_dir / f"step_{self.obs_step:06d}_{stamp}_headrgb.png"), head_bgr)
        cv2.imwrite(str(self.save_obs_dir / f"step_{self.obs_step:06d}_{stamp}_wristrgb.png"), wrist_bgr)

    def _get_obs(self):
        joints = self.robot.get_pos_j().astype(np.float32)
        eef = self.robot.get_pos_eef().astype(np.float32)
        hand = self.hand.get_hand_pos().astype(np.float32)
        head_bgr = self.head_camera.get_data()
        wrist_bgr = self.wrist_camera.get_data()
        self._save_obs_images(head_bgr, wrist_bgr)
        if self.robot_cfg.state_mode == "joint_hand":
            state = np.concatenate([joints, hand], axis=0)
        elif self.robot_cfg.state_mode == "eef_hand":
            state = np.concatenate([eef, hand], axis=0)
        else:
            state = np.concatenate([joints, eef, hand], axis=0)
        base_bgr = head_bgr if self.robot_cfg.swap_cameras else wrist_bgr
        wrist_obs_bgr = wrist_bgr if self.robot_cfg.swap_cameras else head_bgr
        if self.robot_cfg.log_action_details and self.obs_step <= 3:
            self.logger.info(
                "obs step=%d state[j=%s eef=%s hand=%s] image_src=%s wrist_image_src=%s head_shape=%s wrist_shape=%s",
                self.obs_step,
                np.array2string(joints, precision=4),
                np.array2string(eef, precision=4),
                np.array2string(hand, precision=1),
                "head_camera" if self.robot_cfg.swap_cameras else "wrist_camera",
                "wrist_camera" if self.robot_cfg.swap_cameras else "head_camera",
                tuple(head_bgr.shape),
                tuple(wrist_bgr.shape),
            )
        return {
            "observation/image": self._resize_rgb(base_bgr),
            "observation/wrist_image": self._resize_rgb(wrist_obs_bgr),
            "observation/state": state,
            "prompt": self.prompt,
        }, joints, hand

    def _apply_action(self, action: np.ndarray, current_joints: np.ndarray, current_hand: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 12:
            raise ValueError(f"Expected at least 12-D action, got {action.shape}.")
        if self.robot_cfg.action_mode == "joint_hand_target":
            target_joints = action[:6]
            target_hand = action[6:12]
            self.logger.info(
                "infer joint_hand_target target_joints=%s target_hand=%s",
                np.array2string(target_joints, precision=4),
                np.array2string(target_hand, precision=2),
            )
            self.robot.set_pos_j(np.asarray(target_joints, dtype=np.float32), servo=True)
            self.hand.set_hand_pos(np.rint(np.clip(target_hand, 0, 2000)).astype(np.int32).tolist())
            return
        if self.robot_cfg.action_mode == "eef_hand_target":
            target_eef = action[:6].astype(np.float32)
            target_hand = action[6:12].astype(np.float32)
            target_joints = self.robot.rtde_c.getInverseKinematics(target_eef.tolist(), current_joints.tolist())
            if target_joints is None:
                self.logger.warning("IK failed for target_eef=%s", np.array2string(target_eef, precision=4))
                return
            target_joints = np.asarray(target_joints, dtype=np.float32)
            self.logger.info(
                "infer eef_hand_target target_eef=%s target_joints=%s target_hand=%s",
                np.array2string(target_eef, precision=4),
                np.array2string(target_joints, precision=4),
                np.array2string(target_hand, precision=2),
            )
            self.robot.set_pos_j(target_joints, servo=True)
            self.hand.set_hand_pos(np.rint(np.clip(target_hand, 0, 2000)).astype(np.int32).tolist())
            return
        raw_eef_delta = action[:6]
        raw_hand_delta = action[6:12]
        eef_delta = raw_eef_delta * float(self.robot_cfg.joint_scale)
        hand_delta = raw_hand_delta * float(self.robot_cfg.hand_scale)
        if self.robot_cfg.disable_position_delta:
            eef_delta[:3] = 0.0
        else:
            max_pos = abs(float(self.robot_cfg.max_eef_pos_delta))
            eef_delta[:3] = np.clip(eef_delta[:3], -max_pos, max_pos)
        if self.robot_cfg.disable_rotation_delta:
            eef_delta[3:6] = 0.0
        else:
            max_rot = abs(float(self.robot_cfg.max_eef_rot_delta))
            eef_delta[3:6] = np.clip(eef_delta[3:6], -max_rot, max_rot)
        if self.robot_cfg.disable_hand_delta:
            hand_delta[:] = 0.0
        else:
            max_hand = abs(float(self.robot_cfg.max_hand_delta))
            hand_delta = np.clip(hand_delta, -max_hand, max_hand)
        current_eef = self.robot.get_pos_eef().astype(np.float32)
        target_eef = current_eef.copy()
        target_eef[:3] = current_eef[:3] + eef_delta[:3]
        current_rot = R.from_rotvec(current_eef[3:6]).as_matrix()
        delta_rot = R.from_rotvec(eef_delta[3:6]).as_matrix()
        target_rot = delta_rot @ current_rot
        target_eef[3:6] = R.from_matrix(target_rot).as_rotvec()
        target_joints = self.robot.rtde_c.getInverseKinematics(target_eef.tolist(), current_joints.tolist())
        if target_joints is None:
            self.logger.warning("IK failed for target_eef=%s", np.array2string(target_eef, precision=4))
            return
        target_joints = np.asarray(target_joints, dtype=np.float32)
        target_hand = current_hand + hand_delta
        if self.robot_cfg.log_action_details:
            self.logger.info(
                "infer raw_eef_delta=%s clipped_eef_delta=%s current_eef=%s target_eef=%s raw_hand_delta=%s hand_delta=%s current_joints=%s target_joints=%s current_hand=%s target_hand=%s",
                np.array2string(raw_eef_delta, precision=4),
                np.array2string(eef_delta, precision=4),
                np.array2string(current_eef, precision=4),
                np.array2string(target_eef, precision=4),
                np.array2string(raw_hand_delta, precision=2),
                np.array2string(hand_delta, precision=2),
                np.array2string(current_joints, precision=4),
                np.array2string(target_joints, precision=4),
                np.array2string(current_hand, precision=2),
                np.array2string(target_hand, precision=2),
            )
        else:
            self.logger.info(
                "infer eef_delta=%s target_eef=%s hand_delta=%s",
                np.array2string(eef_delta, precision=4),
                np.array2string(target_eef, precision=4),
                np.array2string(hand_delta, precision=2),
            )
        self.robot.set_pos_j(target_joints, servo=True)
        self.hand.set_hand_pos(np.rint(np.clip(target_hand, 0, 2000)).astype(np.int32).tolist())

    def run(self):
        self.logger.info("Press Ctrl+C to stop. Starting real-robot rollout loop.")
        try:
            while True:
                self.obs_step += 1
                obs, joints, hand = self._get_obs()
                outputs = self.policy.infer(obs)
                self.logger.info("policy_timing=%s", outputs.get("policy_timing"))
                action = outputs["actions"]
                if action.ndim > 1:
                    action = action[0]
                self._apply_action(action, joints, hand)
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
    parser.add_argument("--mock-cameras", action="store_true")
    parser.add_argument("--bridge-l515", action="store_true")
    parser.add_argument("--joint-scale", type=float, default=10.0)
    parser.add_argument("--hand-scale", type=float, default=1.0)
    parser.add_argument("--max-eef-pos-delta", type=float, default=0.01)
    parser.add_argument("--max-eef-rot-delta", type=float, default=0.08)
    parser.add_argument("--max-hand-delta", type=float, default=80.0)
    parser.add_argument("--disable-position-delta", action="store_true")
    parser.add_argument("--disable-rotation-delta", action="store_true")
    parser.add_argument("--disable-hand-delta", action="store_true")
    parser.add_argument("--swap-cameras", action="store_true")
    parser.add_argument("--log-action-details", action="store_true")
    parser.add_argument("--state-mode", choices=["joint_eef_hand", "joint_hand", "eef_hand"], default="joint_eef_hand")
    parser.add_argument("--action-mode", choices=["eef_delta", "eef_hand_target", "joint_hand_target"], default="eef_delta")
    parser.add_argument("--save-obs-dir", default="./pi05_obs")
    parser.add_argument("--save-obs-every", type=int, default=1)
    args = parser.parse_args()

    cfg = RobotConfig(
        robot_ip=args.robot_ip,
        hand_port=args.hand_port,
        control_hz=args.control_hz,
        arm_speed=args.arm_speed,
        arm_acceleration=args.arm_acceleration,
        mock_cameras=args.mock_cameras,
        bridge_l515=args.bridge_l515,
        joint_scale=args.joint_scale,
        hand_scale=args.hand_scale,
        max_eef_pos_delta=args.max_eef_pos_delta,
        max_eef_rot_delta=args.max_eef_rot_delta,
        max_hand_delta=args.max_hand_delta,
        disable_position_delta=args.disable_position_delta,
        disable_rotation_delta=args.disable_rotation_delta,
        disable_hand_delta=args.disable_hand_delta,
        swap_cameras=args.swap_cameras,
        log_action_details=args.log_action_details,
        state_mode=args.state_mode,
        action_mode=args.action_mode,
        save_obs_dir=args.save_obs_dir,
        save_obs_every=args.save_obs_every,
    )
    runner = PI05RealRobotRunner(args.checkpoint_dir, args.train_config, args.prompt, cfg)
    runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
