from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pi06_recap.vf_data import (
    RobotValueManifestDataset,
    VQAStreamDataset,
    alignment_collate,
    load_jsonl,
    robot_collate,
    split_rows_by_episode,
)
from pi06_recap.vf_model import Pi06StitchedValueFunction


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def set_alignment_trainable(model: Pi06StitchedValueFunction) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True


def set_robot_trainable(
    model: Pi06StitchedValueFunction,
    *,
    train_projector: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.value_head.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = train_projector
    if use_lora:
        model.apply_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)


def run_alignment(model: Pi06StitchedValueFunction, args: argparse.Namespace, output_dir: Path) -> None:
    set_alignment_trainable(model)
    loader = DataLoader(
        VQAStreamDataset(
            image_processor=model.image_processor,
            tokenizer=model.tokenizer,
            dataset_name=args.alignment_dataset,
            subset_name=args.alignment_subset,
            split=args.alignment_split,
            max_length=args.alignment_max_length,
            max_samples=args.alignment_max_samples,
        ),
        batch_size=args.alignment_batch_size,
        num_workers=0,
        collate_fn=partial(alignment_collate, pad_token_id=model.tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.alignment_lr, weight_decay=0.01)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, args.alignment_steps)
    model.train()
    iterator = iter(loader)
    pbar = tqdm(range(args.alignment_steps), desc="alignment")
    optimizer.zero_grad(set_to_none=True)
    for step in pbar:
        batch = next(iterator)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda")):
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                task_type="alignment",
                labels=batch["labels"].to(args.device),
            )
            loss = out["loss"] / args.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")
        if args.save_steps and (step + 1) % args.save_steps == 0:
            ckpt = output_dir / "alignment" / f"step-{step + 1}"
            model.save_parts(ckpt, save_lora=False)

    final_dir = output_dir / "alignment"
    model.save_parts(final_dir, save_lora=False)


@torch.no_grad()
def evaluate_robot(model: Pi06StitchedValueFunction, loader: DataLoader, args: argparse.Namespace) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total = 0
    for batch in tqdm(loader, desc="val", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda")):
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                task_type="robot",
                labels=batch["labels"].to(args.device),
            )
        labels = batch["labels"].to(out["logits"].device)
        logits = out["logits"].float()
        total_loss += out["loss"].item() * labels.shape[0]
        total_top1 += (logits.argmax(dim=-1) == labels).sum().item()
        total_top5 += (logits.topk(min(5, logits.shape[-1]), dim=-1).indices == labels[:, None]).any(dim=1).sum().item()
        total += labels.shape[0]
    model.train()
    total = max(1, total)
    return total_loss / total, total_top1 / total, total_top5 / total


def run_robot(model: Pi06StitchedValueFunction, args: argparse.Namespace, output_dir: Path) -> None:
    if args.projector_path:
        state = torch.load(args.projector_path, map_location="cpu", weights_only=True)
        model.projector.load_state_dict(state)
    set_robot_trainable(
        model,
        train_projector=args.train_projector,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    rows = load_jsonl(args.manifest)
    train_rows, val_rows, test_rows = split_rows_by_episode(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        with (split_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")

    train_ds = RobotValueManifestDataset(
        args.manifest,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        max_length=args.robot_max_length,
        rows=train_rows,
    )
    val_ds = RobotValueManifestDataset(
        args.manifest,
        image_processor=model.image_processor,
        tokenizer=model.tokenizer,
        max_length=args.robot_max_length,
        rows=val_rows,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.robot_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(robot_collate, pad_token_id=model.tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.robot_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(robot_collate, pad_token_id=model.tokenizer.pad_token_id),
    )

    steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = args.robot_epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.robot_lr, weight_decay=0.01)
    scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)

    best_val = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.robot_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"robot epoch {epoch + 1}/{args.robot_epochs}")
        for step, batch in enumerate(pbar):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda")):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    task_type="robot",
                    labels=batch["labels"].to(args.device),
                )
                loss = out["loss"] / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            pbar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")
            if args.save_steps and global_step and global_step % args.save_steps == 0:
                model.save_parts(output_dir / "robot" / f"step-{global_step}", save_lora=args.use_lora)

        val_loss, val_top1, val_top5 = evaluate_robot(model, val_loader, args)
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save_parts(output_dir / "robot" / "best", save_lora=args.use_lora)

    model.save_parts(output_dir / "robot" / "final", save_lora=args.use_lora)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stitched pi0.6-style value function.")
    parser.add_argument("--stage", choices=["alignment", "robot", "both"], default="robot")
    parser.add_argument("--output-dir", default="outputs/pi06_vf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--vision-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--language-model", default="google/gemma-3-270m-it")
    parser.add_argument("--num-value-bins", type=int, default=201)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1000)

    parser.add_argument("--alignment-dataset", default="HuggingFaceM4/the_cauldron")
    parser.add_argument("--alignment-subset", default="vqav2")
    parser.add_argument("--alignment-split", default="train")
    parser.add_argument("--alignment-steps", type=int, default=5000)
    parser.add_argument("--alignment-batch-size", type=int, default=8)
    parser.add_argument("--alignment-lr", type=float, default=2e-5)
    parser.add_argument("--alignment-max-length", type=int, default=512)
    parser.add_argument("--alignment-max-samples", type=int, default=None)

    parser.add_argument("--manifest", help="Robot VF manifest JSONL.")
    parser.add_argument("--projector-path", default=None, help="Optional projector.pt from alignment stage.")
    parser.add_argument("--robot-epochs", type=int, default=5)
    parser.add_argument("--robot-batch-size", type=int, default=8)
    parser.add_argument("--robot-lr", type=float, default=1e-4)
    parser.add_argument("--robot-max-length", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-projector", action="store_true", default=True)
    parser.add_argument("--freeze-projector", dest="train_projector", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"robot", "both"} and not args.manifest:
        raise ValueError("--manifest is required for robot or both stages")

    model = Pi06StitchedValueFunction(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        num_value_bins=args.num_value_bins,
        cache_dir=args.cache_dir,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    if args.stage in {"alignment", "both"}:
        run_alignment(model, args, output_dir)
    if args.stage in {"robot", "both"}:
        if args.stage == "both" and args.projector_path is None:
            args.projector_path = str(output_dir / "alignment" / "projector.pt")
        run_robot(model, args, output_dir)


if __name__ == "__main__":
    main()
