#!/usr/bin/env python3
"""Evaluate PI0.5 checkpoints: compute MSE between predicted and ground truth actions."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import safetensors.torch as st
import torch
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache


GEMMA_2B = dict(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
GEMMA_300M = dict(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)


def make_paligemma_config():
    cfg = CONFIG_MAPPING["paligemma"]()
    cfg._vocab_size = 257152
    cfg.image_token_index = 257152
    cfg.text_config.hidden_size = GEMMA_2B["width"]
    cfg.text_config.intermediate_size = GEMMA_2B["mlp_dim"]
    cfg.text_config.num_attention_heads = GEMMA_2B["num_heads"]
    cfg.text_config.head_dim = GEMMA_2B["head_dim"]
    cfg.text_config.num_hidden_layers = GEMMA_2B["depth"]
    cfg.text_config.num_key_value_heads = GEMMA_2B["num_kv_heads"]
    cfg.text_config.hidden_activation = "gelu_pytorch_tanh"
    cfg.text_config.torch_dtype = "float32"
    cfg.text_config.vocab_size = 257152
    cfg.text_config.use_adarms = False
    cfg.text_config.adarms_cond_dim = None
    cfg.vision_config.intermediate_size = 4304
    cfg.vision_config.projection_dim = 2048
    cfg.vision_config.projector_hidden_act = "gelu_fast"
    cfg.vision_config.torch_dtype = "float32"
    return cfg


def make_gemma_expert_config():
    return CONFIG_MAPPING["gemma"](
        head_dim=GEMMA_300M["head_dim"],
        hidden_size=GEMMA_300M["width"],
        intermediate_size=GEMMA_300M["mlp_dim"],
        num_attention_heads=GEMMA_300M["num_heads"],
        num_hidden_layers=GEMMA_300M["depth"],
        num_key_value_heads=GEMMA_300M["num_kv_heads"],
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        use_adarms=True,
        adarms_cond_dim=GEMMA_300M["width"],
    )


def load_parquet_data(parquet_dir: str, max_episodes: int = None):
    parquets = sorted(Path(parquet_dir).glob("data/chunk-*/*.parquet"))
    if not parquets:
        parquets = sorted(Path(parquet_dir).glob("*.parquet"))
    if max_episodes:
        parquets = parquets[:max_episodes]
    all_states, all_actions = [], []
    for p in parquets:
        table = pq.read_table(p)
        for row in table.to_pylist():
            all_states.append(np.array(row["state"], dtype=np.float32))
            all_actions.append(np.array(row["actions"], dtype=np.float32))
    return np.stack(all_states), np.stack(all_actions)


def unnormalize_quantile(x, q01, q99):
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def load_norm_stats(path: str):
    with open(path) as f:
        data = json.load(f)
    stats = {}
    for key, val in data["norm_stats"].items():
        stats[key] = {
            "mean": np.array(val["mean"], dtype=np.float32),
            "std": np.array(val["std"], dtype=np.float32),
            "q01": np.array(val["q01"], dtype=np.float32) if val.get("q01") is not None else None,
            "q99": np.array(val["q99"], dtype=np.float32) if val.get("q99") is not None else None,
        }
    return stats


def unnormalize_actions(x, norm_stats):
    s = norm_stats["actions"]
    if s["q01"] is not None and s["q99"] is not None:
        return unnormalize_quantile(x, s["q01"], s["q99"])
    return x * (s["std"] + 1e-6) + s["mean"]


def sinusoidal_pos_embedding(time_val, dim, device):
    """Returns [dim] tensor."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=device)
    min_period, max_period = 0.004, 4.0
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * torch.pi
    sin_input = scaling * time_val
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)]).to(torch.bfloat16)


class PI05Model(nn.Module):
    """Minimal PI0.5 model matching the openpi PaliGemmaWithExpertModel architecture."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.paligemma = PaliGemmaForConditionalGeneration(config=make_paligemma_config())
        self.gemma_expert = GemmaForCausalLM(config=make_gemma_expert_config())
        self.gemma_expert.model.embed_tokens = None

        self.action_in_proj = nn.Linear(32, GEMMA_300M["width"])
        self.action_out_proj = nn.Linear(GEMMA_300M["width"], 32)
        self.time_mlp_in = nn.Linear(GEMMA_300M["width"], GEMMA_300M["width"])
        self.time_mlp_out = nn.Linear(GEMMA_300M["width"], GEMMA_300M["width"])

        self.to(dtype=torch.bfloat16)
        self.to(device)
        self.eval()

    def _compute_prefix_kv(self, images, device):
        """Run paligemma to compute prefix KV cache from images."""
        bsize = images[0].shape[0]
        prefix_embeds = self.paligemma.model.get_image_features(images[0])
        prefix_embeds = prefix_embeds.to(torch.bfloat16)
        if prefix_embeds.ndim == 2:
            prefix_embeds = prefix_embeds.unsqueeze(0)
        seq_len = prefix_embeds.shape[1]
        image_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        position_ids = torch.cumsum(image_masks, dim=1) - 1
        attn_mask_4d = torch.where(image_masks[:, None, :, None], 0.0, -2.3819763e38)

        self.paligemma.language_model.config._attn_implementation = "eager"

        outputs = self.paligemma.language_model.forward(
            inputs_embeds=prefix_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            use_cache=True,
        )
        return outputs.past_key_values

    def _denoise_step(self, state, prefix_kv, x_t, time_val, device):
        """One denoising step: embed suffix, run expert, project output."""
        # x_t: [B, action_horizon, 32]
        action_embed = self.action_in_proj(x_t.to(torch.bfloat16))
        time_sin = sinusoidal_pos_embedding(time_val, action_embed.shape[-1], device)
        # time_sin is [width] - expand to [B, action_horizon, width]
        time_sin = time_sin[None, None, :].expand_as(action_embed)
        action_time_emb = action_embed + time_sin

        # adaRMS conditioning: time_emb through MLP
        def time_mlp(x):
            x = self.time_mlp_in(x)
            x = torch.nn.functional.silu(x)
            x = self.time_mlp_out(x)
            return torch.nn.functional.silu(x)

        adarms_cond = time_mlp(time_sin[:, 0, :])  # [B, width]
        suffix_input = action_time_emb

        action_horizon = suffix_input.shape[1]
        bsize = state.shape[0]
        prefix_len = prefix_kv.key_cache[0].shape[2]

        # Build attention mask
        prefix_pad_2d = torch.ones(bsize, action_horizon, prefix_len, dtype=torch.bool, device=device)
        suffix_causal = torch.tril(torch.ones(bsize, action_horizon, action_horizon, dtype=torch.bool, device=device))
        prefix_pad_2d[:, 1:, :] = False

        full_att_2d = torch.cat([prefix_pad_2d, suffix_causal], dim=2)
        full_att_4d = torch.where(full_att_2d[:, None, :, :], 0.0, -2.3819763e38)

        prefix_offsets = torch.full((bsize, 1), prefix_len, dtype=torch.long, device=device)
        position_ids = prefix_offsets + torch.cumsum(torch.ones(bsize, action_horizon, dtype=torch.long, device=device), dim=1) - 1

        self.gemma_expert.model.config._attn_implementation = "eager"

        expert_out = self.gemma_expert.model.forward(
            inputs_embeds=suffix_input,
            attention_mask=full_att_4d,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = expert_out.last_hidden_state.to(dtype=torch.bfloat16)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(self, device, state, images, num_steps=10):
        bsize = state.shape[0]
        noise = torch.normal(0.0, 1.0, size=(bsize, 10, 32), dtype=torch.float32, device=device)

        prefix_kv = self._compute_prefix_kv(images, device)

        dt = -1.0 / num_steps
        x_t = noise
        time_val = 1.0

        while time_val >= -dt / 2:
            v_t = self._denoise_step(state, prefix_kv, x_t, time_val, device)
            x_t = x_t + dt * v_t
            time_val += dt

        return x_t


def load_model(ckpt_path: str, device: str = "cpu"):
    model = PI05Model(device=device)
    state_dict = st.load_file(str(Path(ckpt_path) / "model.safetensors"))

    model_dict = model.state_dict()
    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("paligemma_with_expert.paligemma."):
            new_key = k.replace("paligemma_with_expert.paligemma.", "paligemma.")
        elif k.startswith("paligemma_with_expert.gemma_expert."):
            new_key = k.replace("paligemma_with_expert.gemma_expert.", "gemma_expert.")
        elif k in model_dict:
            new_key = k
        else:
            continue
        if new_key in model_dict and model_dict[new_key].shape == v.shape:
            new_dict[new_key] = v

    model.load_state_dict(new_dict, strict=False)
    print(f"  Loaded {len(new_dict)}/{len(model_dict)} params")
    missing = [k for k in model_dict if k not in new_dict]
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dirs", nargs="+", required=True)
    parser.add_argument("--data-dir", default="/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place")
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir} (max_episodes={args.max_episodes})", flush=True)
    states, gt_actions = load_parquet_data(args.data_dir, max_episodes=args.max_episodes)
    print(f"Loaded {len(states)} frames, action shape: {gt_actions.shape}")
    gt_12 = gt_actions[:, :12]

    results = {}

    for ckpt_dir in args.ckpt_dirs:
        ckpt_name = Path(ckpt_dir).parent.name + "/" + Path(ckpt_dir).name
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}", flush=True)
        print(f"{'='*60}")

        norm_stats_path = Path(ckpt_dir) / "assets" / "local" / "pi05-pickplace-il" / "norm_stats.json"
        if not norm_stats_path.exists():
            print(f"  [SKIP] norm_stats.json not found")
            continue

        norm_stats = load_norm_stats(str(norm_stats_path))

        try:
            model = load_model(ckpt_dir, device=args.device)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Running inference on {len(states)} frames...", flush=True)
        all_preds = []
        with torch.no_grad():
            for i in range(len(states)):
                state = torch.tensor(states[i:i+1], dtype=torch.float32, device=args.device)
                dummy_img = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=args.device)
                pred = model.sample_actions(args.device, state, [dummy_img], num_steps=args.num_steps)
                all_preds.append(pred.squeeze(0).cpu().numpy())
                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(states)}]", flush=True)

        pred_all = np.stack(all_preds)
        if pred_all.ndim != 3 or pred_all.shape[1] < 1 or pred_all.shape[2] < 12:
            raise ValueError(f'Unexpected prediction shape: {pred_all.shape}')
        pred_first = pred_all[:, 0, :]
        pred_12 = pred_first[:, :12]

        pred_phys = unnormalize_actions(pred_12, norm_stats)
        gt_phys = unnormalize_actions(gt_12, norm_stats)

        mse_per_dim = np.mean((pred_phys - gt_phys) ** 2, axis=0)
        mse_total = np.mean(mse_per_dim)

        print(f"\n  Results:")
        print(f"    MSE (total, 12D): {mse_total:.10f}")
        print(f"    MSE per dim:")
        print(f"      eef pos(x,y,z):  {mse_per_dim[0]:.10f}, {mse_per_dim[1]:.10f}, {mse_per_dim[2]:.10f}")
        print(f"      eef rot(r,p,y):  {mse_per_dim[3]:.10f}, {mse_per_dim[4]:.10f}, {mse_per_dim[5]:.10f}")
        print(f"      hand (0-5):      {mse_per_dim[6]:.10f}, {mse_per_dim[7]:.10f}, {mse_per_dim[8]:.10f}, {mse_per_dim[9]:.10f}, {mse_per_dim[10]:.10f}, {mse_per_dim[11]:.10f}")

        print(f"\n  Sample (first frame, eef pos delta in meters):")
        print(f"    Pred: {pred_phys[0, :3]}")
        print(f"    GT:   {gt_phys[0, :3]}")

        results[ckpt_name] = {"mse_total": float(mse_total)}
        del model

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  {name}: MSE = {res['mse_total']:.10f}")


if __name__ == "__main__":
    main()
