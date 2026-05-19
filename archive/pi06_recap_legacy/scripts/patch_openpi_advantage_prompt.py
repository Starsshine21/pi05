#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


HELPER = r'''
# --- pi06-recap advantage prompt patch ---
_PI06_ADVANTAGE_ENV = "PI06_ADVANTAGE_JSONL"
_PI06_FORCE_ADVANTAGE_ENV = "PI06_FORCE_ADVANTAGE"


def _pi06_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=4)
def _pi06_load_advantage_lookup(path: str):
    lookup = {}
    if not path:
        return lookup
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["episode_index"]), int(row["frame_index"]))
            label = row.get("advantage") or row.get("advantage_label") or row.get("label")
            if label:
                lookup[key] = str(label).lower()
    return lookup


def _pi06_append_advantage(prompt: str, data: DataDict) -> str:
    forced = os.environ.get(_PI06_FORCE_ADVANTAGE_ENV)
    if forced:
        return f"{prompt}\nAdvantage: {forced.lower()}"

    path = os.environ.get(_PI06_ADVANTAGE_ENV)
    if not path:
        return prompt
    if "episode_index" not in data or "frame_index" not in data:
        return prompt
    key = (int(_pi06_scalar(data["episode_index"])), int(_pi06_scalar(data["frame_index"])))
    label = _pi06_load_advantage_lookup(path).get(key)
    if not label:
        return prompt
    return f"{prompt}\nAdvantage: {label}"
# --- end pi06-recap advantage prompt patch ---
'''


def patch_transforms(path: Path) -> None:
    text = path.read_text()
    if "pi06-recap advantage prompt patch" in text:
        print(f"Already patched: {path}")
        return

    import_block = "from collections.abc import Callable, Mapping, Sequence\nimport dataclasses\nimport re\n"
    if import_block not in text:
        raise RuntimeError("Could not find expected import block in transforms.py")
    text = text.replace(
        import_block,
        "from collections.abc import Callable, Mapping, Sequence\n"
        "import dataclasses\n"
        "from functools import lru_cache\n"
        "import json\n"
        "import os\n"
        "import re\n",
        1,
    )

    marker = "@dataclasses.dataclass(frozen=True)\nclass PromptFromLeRobotTask"
    if marker not in text:
        raise RuntimeError("Could not find PromptFromLeRobotTask in transforms.py")
    text = text.replace(marker, HELPER + "\n" + marker, 1)

    old_return = '        return {**data, "prompt": prompt}\n'
    new_return = '        return {**data, "prompt": _pi06_append_advantage(prompt, data)}\n'
    if old_return not in text:
        raise RuntimeError("Could not find PromptFromLeRobotTask return statement")
    text = text.replace(old_return, new_return, 1)

    backup = path.with_suffix(path.suffix + ".pi06_recap_backup")
    backup.write_text(path.read_text())
    path.write_text(text)
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch openpi to append Advantage labels to prompts.")
    parser.add_argument("openpi_root", help="Path to a cloned Physical-Intelligence/openpi repository.")
    args = parser.parse_args()
    root = Path(args.openpi_root).resolve()
    target = root / "src" / "openpi" / "transforms.py"
    if not target.exists():
        raise FileNotFoundError(target)
    patch_transforms(target)


if __name__ == "__main__":
    main()

