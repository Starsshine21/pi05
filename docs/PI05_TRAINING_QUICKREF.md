# PI05 训练命令速查

## 1. 环境检查

```bash
cd /nfs_global/S/yangrongzheng/pi05
bash scripts/check_local_openpi_env.sh
```

## 2. 基础 smoke

```bash
sbatch scripts/pi05_smoke.slurm
```

## 3. 训练诊断

```bash
sbatch scripts/pi05_train_diag.slurm
```

## 4. 正式训练

```bash
sbatch scripts/pi05_train.slurm
```

## 5. 自定义实验名

```bash
sbatch --export=ALL,PI05_EXPERIMENT_NAME=my_pi05_run scripts/pi05_train.slurm
```

## 6. 切换精度

```bash
sbatch --export=ALL,PI05_OPENPI_DTYPE=float32 scripts/pi05_train.slurm
```

默认是：

```bash
PI05_OPENPI_DTYPE=bfloat16
```

## 7. 查看任务

```bash
squeue -u $USER
```

```bash
squeue -j <jobid>
```

## 8. 看日志

```bash
tail -f logs/pi05-train-<jobid>.out
```

```bash
tail -f logs/pi05-train-<jobid>.err
```

## 9. 看 GPU

```bash
nvidia-smi
```

## 10. 关键文件

- `docs/PI05_TRAINING_GUIDE.md`
- `scripts/pi05_train.slurm`
- `scripts/pi05_train_diag.slurm`
- `scripts/pi05_train_smoke.slurm`
- `scripts/pi05_smoke.slurm`
- `scripts/use_local_openpi_env.sh`
