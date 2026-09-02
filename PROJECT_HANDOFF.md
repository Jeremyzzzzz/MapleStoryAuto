# MapleStoryAutoLevelUp 项目交接

更新时间：2026-08-11（Asia/Shanghai）

## 1. 项目位置

仓库根目录：

```text
C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp
```

当前工作区存在未提交改动和未跟踪文件。接手前先执行：

```powershell
git status --short
```

不要清理、重置或覆盖现有工作区；部分改动来自此前工作，尚未逐项归属或提交。

## 2. 当前能力边界

当前支持两条彼此隔离的链路：

| 链路 | 输入 | 输出 | 是否向游戏客户端发按键 |
| --- | --- | --- | --- |
| 实时感知查看器 | 游戏窗口的只读画面 | 角色、怪物、运动、HP/MP/EXP、攻击/躲避建议 | 否 |
| 离线战斗仿真器 | 本地一维战斗状态 | 接近、攻击、冷却拉开、躲避、击杀和奖励统计 | 否，不连接客户端 |

实时查看器窗口标题固定为：

```text
MapleStory Perception - OBSERVE ONLY
```

`tools/run_bounded_combat_smoke.py` 会导入键盘控制代码并可能发送输入，不属于本交接中验证和支持的只读/离线工作流，不要在实际客户端上运行。

## 3. 主要文件

| 路径 | 说明 |
| --- | --- |
| `tools/live_perception_viewer.py` | 实时只读感知、YOLO 检测、运动候选、状态条读取和战斗建议 |
| `config/config_shanda_legacy.yaml` | 当前窗口、UI 区域、按键字段、识别和建议阈值 |
| `nametag/shanda_legacy_player.png` | 当前角色名牌模板 |
| `tools/prepare_small_yolo_dataset.py` | 三分类数据集构建和只读帧采集 |
| `tools/train_small_yolo.py` | YOLO11n 训练与验证入口 |
| `training_data/maple_three_class_v1/` | 当前训练/验证数据及元数据 |
| `training_runs/maple_three_class_v1_balanced/` | 当前训练结果、图表和权重 |
| `tools/offline_combat_simulator.py` | 与客户端隔离的闭环攻击/躲避仿真器 |
| `tests/` | 感知、运动、建议、离线战斗和键盘后端测试 |
| `probe_output/` | 最新截图、JSON 摘要和离线仿真结果 |

## 4. Python 与硬件环境

训练和实时查看器解释器：

```text
C:\quant_lab\lstm_gpu_venv\Scripts\python.exe
```

2026-08-11 实测：

```text
torch=2.12.0+cu130
cuda=True
gpu=NVIDIA GeForce RTX 5080
```

仓库测试使用：

```text
C:\Users\Administrator\Documents\ChatGPT\冒险岛\.venv-msal\Scripts\python.exe
```

查看器会在启动时把仓库内 `.yolo_runtime` 加入模块路径，因此直接在解释器顶层执行 `import cv2` 不一定成功；应通过查看器或训练脚本入口运行。

## 5. 启动实时只读查看器

在仓库根目录执行：

```powershell
& 'C:\quant_lab\lstm_gpu_venv\Scripts\pythonw.exe' tools\live_perception_viewer.py `
  --cfg shanda_legacy `
  --monster-backend yolo `
  --yolo-confidence 0.10 `
  --motion-detection `
  --motion-candidate-score 0.70 `
  --combat-advisory
```

交接时查看器仍在运行：

```text
launcher PID: 16732
window PID:   24044
```

PID 会随重启改变，确认方式：

```powershell
Get-Process | Where-Object { $_.MainWindowTitle -like '*MapleStory Perception*' } |
  Select-Object Id, ProcessName, MainWindowTitle
```

有限帧、只读验证命令：

```powershell
& 'C:\quant_lab\lstm_gpu_venv\Scripts\python.exe' tools\live_perception_viewer.py `
  --cfg shanda_legacy `
  --monster-backend yolo `
  --yolo-confidence 0.10 `
  --motion-detection `
  --motion-candidate-score 0.70 `
  --combat-advisory `
  --headless `
  --max-frames 12 `
  --snapshot probe_output\combat_advisory_preview_final.png `
  --summary probe_output\combat_advisory_summary_final.json
```

最近一次有限帧结果：

- 12 帧，约 4.22 FPS。
- 检测到角色和 7 个怪物。
- HP 100%，MP 98.98%，EXP 73.15%。
- 最近目标为木妖，水平差 4.5 px、垂直差 240.5 px。
- 建议状态为 `TRACKING`，未触发 `ATTACK READY` 或 `DODGE RISK`。
- 角色模板置信度只有 0.3053，接近当前阈值 0.30，需要提高模板稳定性。

结果文件：

```text
probe_output/combat_advisory_preview_final.png
probe_output/combat_advisory_summary_final.json
```

## 6. 检测模型

类别：

```text
red_snail
blue_snail
stump
```

最佳权重：

```text
training_runs/maple_three_class_v1_balanced/weights/best.pt
```

文件大小：5,517,265 bytes

SHA256：

```text
9619F4633F2397D0DB40D609E3E53E080FA1AC0192B082FE8FED5E8B75551B21
```

当前数据集：

- 65 个真实来源帧，其中 60 个为新采集帧。
- 105 张训练图、13 张验证图。
- 24 张合成木妖训练图。
- 29 张蓝蜗牛增强训练图。
- 训练实例：红蜗牛 213、蓝蜗牛 36、木妖 230。
- 验证实例：红蜗牛 24、蓝蜗牛 2、木妖 12。

同地图/同会话族验证指标：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.9387 |
| Recall | 0.9946 |
| mAP50 | 0.9950 |
| mAP50-95 | 0.9285 |

这些指标不是独立跨地图、跨会话 OOS 结果。蓝蜗牛验证集仅有 2 个实例，不能据此判断泛化能力。实时查看器使用 `0.10` 的低置信度阈值，可能产生低置信度木妖误检。

训练配置：YOLO11n、40 epochs、`imgsz=960`、batch 4、seed 17、CUDA device 0。

重新训练命令：

```powershell
& 'C:\quant_lab\lstm_gpu_venv\Scripts\python.exe' tools\train_small_yolo.py `
  --data training_data\maple_three_class_v1\data.yaml `
  --project training_runs `
  --name maple_three_class_v1_balanced `
  --epochs 40 `
  --imgsz 960 `
  --batch 4 `
  --device 0 `
  --seed 17
```

## 7. 离线闭环战斗仿真

执行：

```powershell
& '..\.venv-msal\Scripts\python.exe' tools\offline_combat_simulator.py `
  --seed 7 `
  --max-steps 900 `
  --output probe_output\offline_combat_result.json
```

2026-08-11 实测结果：

```text
completed=true
steps=43
elapsed_seconds=4.3
kills=3
player_hp=100
attack decisions=6
dodge decisions=1
total_reward=36.43
client_connected=false
input_devices_used=false
```

该仿真器当前是确定性规则基线，不是强化学习模型。它用于验证状态、动作、冷却、奖励和终止条件；若要测试学习能力，应在此接口上新增训练策略，并与规则基线做固定种子评估。

## 8. 测试

完整测试命令：

```powershell
& '..\.venv-msal\Scripts\python.exe' -m unittest discover -s tests -v
```

2026-08-11 当前结果：

```text
20 tests passed
```

覆盖范围：

- 角色名牌首次匹配/丢失。
- HP/MP/EXP 配置区域读取。
- 本地运动检测、角色区域屏蔽和镜头运动暂停。
- 攻击范围、目标接近速度、躲避方向和最近目标选择。
- 离线攻击伤害、躲避动作和完整回合。
- Windows 键盘后端单元测试；这不代表允许或需要在实际客户端运行输入链路。

## 9. 已知问题

1. 验证数据与训练数据来自相同地图/会话族，泛化指标偏乐观。
2. 蓝蜗牛验证样本严重不足。
3. 当前只支持红蜗牛、蓝蜗牛、木妖三个类别。
4. 角色检测依赖单个名牌模板，当前实时置信度接近阈值。
5. `--yolo-confidence 0.10` 适合观察召回，但会增加误检；部署阈值尚未校准。
6. 运动检测只能作为 YOLO 的辅助证据，背景动画仍可能产生候选。
7. 当前战斗建议使用像素距离和接近速度，没有平台拓扑、碰撞体或技能范围模型。
8. 工作区未提交，训练数据、权重、输出和代码的版本边界尚未固化。

## 10. 建议的接手顺序

1. 保存当前工作树差异并建立独立分支，不要覆盖已有改动。
2. 固化模型、数据集元数据、权重哈希和验证报告。
3. 补采跨地图、跨角色、跨分辨率和跨会话标注数据，优先补蓝蜗牛。
4. 将怪物检测置信度按类别校准，并报告独立测试集 Precision/Recall。
5. 增加平台拓扑和角色脚底坐标，避免仅凭矩形中心判断攻击距离。
6. 将只读感知结果按帧写入 JSONL，并与人工 QA 或内部服务端动作日志按时间戳对齐。
7. 若内部环境提供正式的 `reset/observe/step/reward` 测试 API，可把离线策略接到该接口；不要通过操作系统键盘注入连接实际客户端。
8. 在固定种子和冻结场景上增加学习策略，与当前规则策略比较击杀率、受伤、动作数和奖励。

## 11. 交接验收清单

- [ ] 能运行完整 20 项测试。
- [ ] 能加载指定 SHA256 的 `best.pt`。
- [ ] 能启动标题为 `OBSERVE ONLY` 的查看器。
- [ ] 能输出有限帧截图和 JSON 摘要。
- [ ] 能在离线仿真器完成 3 次击杀并触发攻击、躲避分支。
- [ ] 已确认工作区未提交改动的归属。
- [ ] 已建立独立跨地图测试集，或明确标记尚未完成。
