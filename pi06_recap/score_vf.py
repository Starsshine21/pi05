from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi06_recap.labels import bin_to_value
from pi06_recap.vf_data import RobotValueManifestDataset, load_jsonl, robot_collate
from pi06_recap.vf_model import Pi06StitchedValueFunction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a LeRobot frame manifest with a trained VF.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="Directory containing projector.pt and value_head.pt.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vision-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--language-model", default="google/gemma-3-270m-it")
    parser.add_argument("--num-value-bins", type=int, default=201)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.manifest)
    model = Pi06StitchedValueFunction(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        num_value_bins=args.num_value_bins,
        cache_dir=args.cache_dir,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    model.load_parts(args.checkpoint)
    model.eval()
    dataset = RobotValueManifestDataset(
        args.manifest,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        rows=rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(robot_collate, pad_token_id=model.tokenizer.pad_token_id),
    )
    bin_values = torch.tensor(
        [bin_to_value(i, num_bins=args.num_value_bins) for i in range(args.num_value_bins)],
        device=args.device,
        dtype=torch.float32,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with output.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="score"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda")):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    task_type="robot",
                )
            probs = out["logits"].float().softmax(dim=-1)
            expected_value = (probs * bin_values[None, :]).sum(dim=-1).detach().cpu().tolist()
            top_bin = probs.argmax(dim=-1).detach().cpu().tolist()
            for local_idx, (score, pred_bin) in enumerate(zip(expected_value, top_bin, strict=True)):
                source = rows[offset + local_idx]
                record = {
                    "episode_index": int(source["episode_index"]),
                    "frame_index": int(source["frame_index"]),
                    "prompt": source.get("prompt", ""),
                    "vf_score": float(score),
                    "vf_pred_bin": int(pred_bin),
                    "target_value": source.get("value"),
                    "target_bin": source.get("value_bin"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            offset += len(expected_value)
    print(f"Wrote scores for {offset:,} rows to {output}")


if __name__ == "__main__":
    main()
