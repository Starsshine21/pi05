# RECAP workspace

This directory is the single entrypoint for the current RECAP experiment.

## What lives here

- `scripts/` - slurm launchers and helper scripts
- `configs/` - local RECAP configs used for this machine
- `docs/` - short notes for running/debugging
- `logs/` - slurm logs
- `models/` - local model entrypoints / symlinks
- `external/rlinf-recap` - upstream RLinf RECAP code tree
- `external/venv` - symlink to the Python environment actually used

## Open-source note

This repository does not include local datasets, private model checkpoints, or machine-specific absolute paths.

Before running the provided scripts, replace local paths in your configs and Slurm launchers with paths valid for your own environment.
