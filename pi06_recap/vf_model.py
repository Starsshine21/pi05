from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer


IMAGE_TOKEN = "<image>"


def _config_hidden_size(config: Any, default: int | None = None) -> int:
    for attr in ("hidden_size", "projection_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    for nested in ("vision_config", "text_config"):
        nested_cfg = getattr(config, nested, None)
        if nested_cfg is not None:
            try:
                return _config_hidden_size(nested_cfg)
            except ValueError:
                pass
    if default is not None:
        return default
    raise ValueError(f"Cannot infer hidden size from config: {config}")


class Pi06StitchedValueFunction(nn.Module):
    """SigLIP + projector + small Gemma + 201-bin value head.

    This intentionally mirrors the open reproduction in PI-0.6-reproduction,
    with one important fix: value prediction uses the last valid token in the
    stitched image+text sequence, not the original text attention length.
    """

    def __init__(
        self,
        *,
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        language_model_name: str = "google/gemma-3-270m-it",
        num_value_bins: int = 201,
        cache_dir: str | None = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.num_value_bins = num_value_bins
        self.device_name = device
        dtype = torch_dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)

        vision = AutoModel.from_pretrained(
            vision_model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.vision_tower = getattr(vision, "vision_model", vision)
        self.vision_tower.to(device)
        self.vision_tower.eval()
        for param in self.vision_tower.parameters():
            param.requires_grad = False

        vision_dim = _config_hidden_size(getattr(self.vision_tower, "config", vision.config))

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                cache_dir=cache_dir,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name, cache_dir=cache_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN))
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name, cache_dir=cache_dir)

        language_dim = _config_hidden_size(self.language_model.config)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, language_dim),
        ).to(device)
        self.value_head = nn.Linear(language_dim, num_value_bins).to(device)

        for param in self.language_model.parameters():
            param.requires_grad = False

    def freeze_language_model(self) -> None:
        for param in self.language_model.parameters():
            param.requires_grad = False

    def freeze_projector(self) -> None:
        for param in self.projector.parameters():
            param.requires_grad = False

    def unfreeze_projector(self) -> None:
        for param in self.projector.parameters():
            param.requires_grad = True

    def apply_lora(
        self,
        *,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.language_model = get_peft_model(self.language_model, config)

    def _vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.vision_tower(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output[:, None, :]
        return outputs[0]

    def _stitch(
        self,
        image_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_embeds = image_embeds.to(dtype=text_embeds.dtype, device=text_embeds.device)
        input_ids = input_ids.to(text_embeds.device)
        attention_mask = attention_mask.to(text_embeds.device)
        labels = labels.to(text_embeds.device) if labels is not None else None

        stitched: list[torch.Tensor] = []
        stitched_masks: list[torch.Tensor] = []
        stitched_labels: list[torch.Tensor] | None = [] if labels is not None else None
        num_image_tokens = image_embeds.shape[1]

        for i in range(input_ids.shape[0]):
            valid_len = int(attention_mask[i].sum().item())
            ids = input_ids[i, :valid_len]
            txt = text_embeds[i, :valid_len]
            mask = attention_mask[i, :valid_len]
            img_positions = torch.where(ids == self.image_token_id)[0]
            img_pos = int(img_positions[0].item()) if len(img_positions) else 0

            if len(img_positions):
                combined = torch.cat([txt[:img_pos], image_embeds[i], txt[img_pos + 1 :]], dim=0)
                combined_mask = torch.cat(
                    [
                        mask[:img_pos],
                        torch.ones(num_image_tokens, dtype=mask.dtype, device=mask.device),
                        mask[img_pos + 1 :],
                    ],
                    dim=0,
                )
                if labels is not None and stitched_labels is not None:
                    lab = labels[i, :valid_len]
                    combined_labels = torch.cat(
                        [
                            lab[:img_pos],
                            torch.full((num_image_tokens,), -100, dtype=lab.dtype, device=lab.device),
                            lab[img_pos + 1 :],
                        ],
                        dim=0,
                    )
            else:
                combined = torch.cat([image_embeds[i], txt], dim=0)
                combined_mask = torch.cat(
                    [torch.ones(num_image_tokens, dtype=mask.dtype, device=mask.device), mask],
                    dim=0,
                )
                if labels is not None and stitched_labels is not None:
                    lab = labels[i, :valid_len]
                    combined_labels = torch.cat(
                        [torch.full((num_image_tokens,), -100, dtype=lab.dtype, device=lab.device), lab],
                        dim=0,
                    )

            stitched.append(combined)
            stitched_masks.append(combined_mask)
            if labels is not None and stitched_labels is not None:
                stitched_labels.append(combined_labels)

        max_len = max(x.shape[0] for x in stitched)
        padded_embeds = []
        padded_masks = []
        padded_labels = [] if stitched_labels is not None else None
        for idx, (emb, mask) in enumerate(zip(stitched, stitched_masks, strict=True)):
            pad = max_len - emb.shape[0]
            if pad:
                emb = torch.cat([emb, torch.zeros(pad, emb.shape[-1], dtype=emb.dtype, device=emb.device)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype, device=mask.device)], dim=0)
            padded_embeds.append(emb)
            padded_masks.append(mask)
            if stitched_labels is not None and padded_labels is not None:
                lab = stitched_labels[idx]
                if pad:
                    lab = torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype, device=lab.device)], dim=0)
                padded_labels.append(lab)

        labels_out = torch.stack(padded_labels, dim=0) if padded_labels is not None else None
        return torch.stack(padded_embeds, dim=0), torch.stack(padded_masks, dim=0), labels_out

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: str,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device_name)
        vision_features = self._vision_features(pixel_values)
        image_embeds = self.projector(vision_features)
        lm_labels = labels if task_type in {"alignment", "vqa"} else None
        stitched_embeds, stitched_mask, stitched_labels = self._stitch(
            image_embeds=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

        outputs = self.language_model(
            inputs_embeds=stitched_embeds,
            attention_mask=stitched_mask,
            labels=stitched_labels,
            output_hidden_states=True,
            return_dict=True,
        )

        if task_type in {"alignment", "vqa"}:
            return {"loss": outputs.loss, "logits": outputs.logits}

        hidden = outputs.hidden_states[-1]
        final_pos = stitched_mask.sum(dim=1).to(torch.long) - 1
        final_pos = final_pos.clamp(min=0, max=hidden.shape[1] - 1)
        final_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), final_pos]
        value_logits = self.value_head(final_hidden)
        result = {"logits": value_logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(value_logits, labels.to(value_logits.device))
        return result

    def save_parts(self, output_dir: str | Path, *, save_lora: bool = True) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.projector.state_dict(), output / "projector.pt")
        torch.save(self.value_head.state_dict(), output / "value_head.pt")
        if save_lora and hasattr(self.language_model, "save_pretrained"):
            self.language_model.save_pretrained(output / "lora_adapter")

    def load_parts(self, checkpoint_dir: str | Path, *, load_lora: bool = True) -> None:
        checkpoint = Path(checkpoint_dir)
        self.projector.load_state_dict(torch.load(checkpoint / "projector.pt", map_location="cpu", weights_only=True))
        self.value_head.load_state_dict(torch.load(checkpoint / "value_head.pt", map_location="cpu", weights_only=True))
        lora_dir = checkpoint / "lora_adapter"
        if load_lora and lora_dir.exists():
            from peft import PeftModel

            self.language_model = PeftModel.from_pretrained(self.language_model, str(lora_dir))

