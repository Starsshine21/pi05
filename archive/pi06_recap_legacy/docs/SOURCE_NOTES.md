# 参考仓库判断

分析时间：2026-05-06。

| 仓库 | 看到的 commit | 结论 |
| --- | --- | --- |
| `Physical-Intelligence/openpi` | `c23745b5ad24e98f66967ea795a07b2588ed6c79` | 官方 π0/π0.5 基座和训练框架；没有 π0.6。适合作为 policy 微调底座。 |
| `Beiyu-kk/openpi-zh` | `41751a6ce0047c8c16915e43f3721a5ad220b885` | openpi 的中文/派生版本；没有提供独立完整 π0.6 管线。 |
| `Alegunm/PI-0.6-reproduction` | `e2bb171a43133ebb7f614cd0c5855bd396b027e5` | 主要复现 value function：SigLIP + Gemma + projector + 201-bin value head。本文档的 VF 部分吸收这条路线，并修正了 stitched sequence 最后 token 位置的问题。 |
| `hzm8341/pi0.6` | `a14a0a65144b5be95b4303173b958f1b0b26836b` | openpi fork。实际可用部分是 G1 pick-apple π0.5 fine-tuning 配置；MEM/π0.6 相关代码多为草稿，`MEMLeRobotDataset` 未接入主训练入口。 |
| `MINT-SJTU/Evo-RL` | `4425ccd672340c2f61b646b6706b4ab9ca308a9e` | LeRobot fork，提供 `pistar06` value model、`lerobot-value-train`、`lerobot-value-infer` 和 ACP policy training。最值得吸收的是 n-step advantage：用 value target 构造 dense reward，计算 `A_t = sum r + V_{t+n} - V_t`，再按 task 内 top ratio 二值化。 |

因此本复现包选择的主线是：官方 openpi π0.5 policy + VF 打分 + Evo-RL 风格 ACP 二值化 + advantage-conditioned prompt 微调。MEM 分支不作为默认复现路径。
