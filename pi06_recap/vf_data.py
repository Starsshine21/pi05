from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

from pi06_recap.vf_model import IMAGE_TOKEN


class VideoFrameCache:
    def __init__(self) -> None:
        try:
            from decord import VideoReader, cpu
        except ImportError as exc:
            raise ImportError("decord is required for video manifests: pip install decord") from exc
        self._video_reader_cls = VideoReader
        self._cpu = cpu
        self._cache: dict[str, Any] = {}

    def read(self, video_path: str | Path, frame_index: int) -> Image.Image:
        path = str(video_path)
        if path not in self._cache:
            self._cache[path] = self._video_reader_cls(path, ctx=self._cpu(0))
        vr = self._cache[path]
        idx = min(max(0, int(frame_index)), len(vr) - 1)
        return Image.fromarray(vr[idx].asnumpy()).convert("RGB")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RobotValueManifestDataset(Dataset):
    """Dataset backed by a JSONL manifest produced by make_value_manifest.py."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_processor: Any,
        tokenizer: Any,
        max_length: int = 160,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else load_jsonl(manifest_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._video_cache: VideoFrameCache | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, row: dict[str, Any]) -> Image.Image:
        if row.get("image_path"):
            return Image.open(row["image_path"]).convert("RGB")
        if row.get("video_path"):
            if self._video_cache is None:
                self._video_cache = VideoFrameCache()
            return self._video_cache.read(row["video_path"], int(row.get("video_frame", row["frame_index"])))
        raise KeyError("Manifest row must contain image_path or video_path")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = self._image(row)
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        prompt = row.get("prompt") or "complete the robot task"
        text = (
            f"{IMAGE_TOKEN}\n"
            f"Task: {prompt}\n"
            "Predict the normalized time-to-success value for this robot state."
        )
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["value_bin"]), dtype=torch.long),
            "episode_index": torch.tensor(int(row["episode_index"]), dtype=torch.long),
            "frame_index": torch.tensor(int(row["frame_index"]), dtype=torch.long),
        }


def split_rows_by_episode(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    episodes = sorted({int(row["episode_index"]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_train = int(len(episodes) * train_ratio)
    n_val = int(len(episodes) * val_ratio)
    train_eps = set(episodes[:n_train])
    val_eps = set(episodes[n_train : n_train + n_val])
    test_eps = set(episodes[n_train + n_val :])
    return (
        [row for row in rows if int(row["episode_index"]) in train_eps],
        [row for row in rows if int(row["episode_index"]) in val_eps],
        [row for row in rows if int(row["episode_index"]) in test_eps],
    )


def robot_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        ),
        "labels": torch.stack([item["labels"] for item in batch]),
        "episode_index": torch.stack([item["episode_index"] for item in batch]),
        "frame_index": torch.stack([item["frame_index"] for item in batch]),
    }


class VQAStreamDataset(IterableDataset):
    """Small streaming VQA/caption loader for projector alignment."""

    def __init__(
        self,
        *,
        image_processor: Any,
        tokenizer: Any,
        dataset_name: str = "HuggingFaceM4/the_cauldron",
        subset_name: str = "vqav2",
        split: str = "train",
        max_length: int = 512,
        max_samples: int | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        self.max_length = max_length
        self.max_samples = max_samples

    def _format(self, example: dict[str, Any]) -> tuple[Image.Image, str, str] | None:
        image = example.get("image") or example.get("images")
        if isinstance(image, list):
            image = image[0] if image else None
        if image is None or not hasattr(image, "convert"):
            return None
        image = image.convert("RGB")

        question = ""
        answer = ""
        texts = example.get("texts") or example.get("conversations") or []
        for turn in texts:
            if not isinstance(turn, dict):
                continue
            question = str(turn.get("user") or turn.get("question") or question).strip()
            answer = str(turn.get("assistant") or turn.get("answer") or answer).strip()
            if question and answer:
                break
        if not question:
            question = str(example.get("question") or "Describe the image.").strip()
        raw_answer = example.get("answer") or example.get("answers") or answer
        if isinstance(raw_answer, list):
            raw_answer = raw_answer[0].get("answer", raw_answer[0]) if raw_answer else ""
        answer = str(raw_answer or answer).strip()
        if not answer:
            return None
        return image, question, answer

    def _encode(self, image: Image.Image, question: str, answer: str) -> dict[str, torch.Tensor]:
        prompt = f"{IMAGE_TOKEN}\nQuestion: {question}\nAnswer:"
        prompt_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        answer_enc = self.tokenizer(
            " " + answer,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, self.max_length - prompt_enc["input_ids"].shape[1]),
            add_special_tokens=False,
        )
        input_ids = torch.cat([prompt_enc["input_ids"].squeeze(0), answer_enc["input_ids"].squeeze(0)], dim=0)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: prompt_enc["input_ids"].shape[1]] = -100
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self) -> Iterable[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        kwargs = {"split": self.split, "streaming": True}
        if self.subset_name:
            kwargs["name"] = self.subset_name
        stream = load_dataset(self.dataset_name, **kwargs)
        emitted = 0
        for example in stream:
            if self.max_samples is not None and emitted >= self.max_samples:
                break
            formatted = self._format(example)
            if formatted is None:
                continue
            yield self._encode(*formatted)
            emitted += 1


def alignment_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0),
        "labels": pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100),
    }

