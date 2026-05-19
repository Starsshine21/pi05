# pi0.6 RECAP 复现包

这个目录不是官方 π0.6。官方 π0.6 没有开源训练代码和权重；这里按论文/参考仓库能落地的部分，整理成一条 openpi 兼容复现线：

1. 用 SigLIP + Gemma-270M + projector + 201-bin value head 训练 value function。
2. 用 value function 给离线 LeRobot 轨迹打分。
3. 把高分帧/轨迹阈值化成 `Advantage: positive/negative`。
4. patch openpi，让 π0.5 policy fine-tuning 时把 advantage label 注入 prompt。
5. 推理时固定使用 `Advantage: positive`。

我看过的几个参考仓库结论：

- `Physical-Intelligence/openpi`：可靠的官方基座，包含 π0/π0.5、pi05_base、LeRobot 训练框架；不包含 π0.6。
- `Beiyu-kk/openpi-zh`：基本是 openpi 的中文/派生版本，不是独立 π0.6 实现。
- `Alegunm/PI-0.6-reproduction`：主要复现 value function，方向有用，但只覆盖 VF，不含 policy 复现。
- `hzm8341/pi0.6`：基于 openpi 加了 G1 pick-apple 配置和部分 MEM 草稿；可参考 G1 配置，但 MEM 数据包装没有完整接到训练主路径。
- `MINT-SJTU/Evo-RL`：LeRobot 体系下更完整的 real-world RL/ACP 流程；我把它的 n-step advantage + task 内 top-ratio 二值化思路合进了 `make_advantage_labels.py`。

完整步骤写在 [docs/REPRODUCE_PI06_RECAP.md](docs/REPRODUCE_PI06_RECAP.md)，参考仓库逐项判断见 [docs/SOURCE_NOTES.md](docs/SOURCE_NOTES.md)。

关键入口：

```bash
python scripts/make_value_manifest.py --help
python scripts/train_vf.py --help
python scripts/score_vf.py --help
python scripts/make_advantage_labels.py --help
python scripts/patch_openpi_advantage_prompt.py --help
```

## PI05 训练怎么开始

如果你现在关注的是**已经打通的 pi05 训练链路**，直接看这份文档：

- `docs/PI05_TRAINING_GUIDE.md`

推荐启动顺序：

```bash
bash scripts/check_local_openpi_env.sh
sbatch scripts/pi05_smoke.slurm
sbatch scripts/pi05_train_diag.slurm
sbatch --export=ALL,PI05_EXPERIMENT_NAME=pi05_run_001 scripts/pi05_train.slurm
```

常用速查命令见：

- `docs/PI05_TRAINING_QUICKREF.md`
