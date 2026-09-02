# YOLO 感知识别交接文档（2026-08-14）

本文只交接画面识别部分：七类怪物检测、检测框跟踪，以及当前地图的梯子、绳子、平台三分类检测。所有新工具均为观察模式，不发送键盘或鼠标输入，也不会切换游戏窗口焦点。

## 1. 当前状态

- 项目目录：`C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp`
- Python：`C:\quant_lab\lstm_gpu_venv\Scripts\python.exe`
- GPU 训练/推理设备：`0`
- 当前怪物实时查看器使用本文 10 节的新身份确认和尺寸一致性逻辑运行。
- `yolo_terrain_viewer.py` 和地形训练进程未启动。
- 工作区本来就有大量未提交文件和用户修改，接手时不要执行 `git reset --hard`、`git clean` 或覆盖式恢复。

## 2. 安全和功能边界

本交接中的可运行入口只有画面捕获、YOLO 推理、框绘制、结果跟踪、静态图片/JSON 输出。

- 可以运行：`tools/yolo_monster_viewer.py`、`tools/yolo_terrain_viewer.py`、`tools/infer_terrain_yolo.py`。
- 不要把识别结果连接到攻击、走路、回血、自动寻路或其他游戏控制逻辑。
- 不要使用 `tools/auto_combat.py` 验证本交接内容；它不是只读感知入口。
- 两个实时查看器的窗口标题都包含 `OBSERVE ONLY`。

## 3. 七类怪物检测

### 3.1 类别

模型类别 ID 顺序如下：

| ID | 类别 |
|---:|---|
| 0 | `slime`（绿水灵） |
| 1 | `red_snail`（红蜗牛） |
| 2 | `green_mushroom`（绿蘑菇） |
| 3 | `stump`（树妖） |
| 4 | `flower_mushroom`（花蘑菇） |
| 5 | `zombie_mushroom`（僵尸蘑菇） |
| 6 | `thorn_mushroom`（刺蘑菇） |

### 3.2 关键文件

| 路径 | 用途 |
|---|---|
| `tools/build_seven_class_increment_dataset.py` | 在原五类数据上加入僵尸蘑菇和刺蘑菇 |
| `tools/yolo_monster_viewer.py` | 七类只读实时查看器、玩家定位、实体坐标和检测框跟踪 |
| `tests/test_build_seven_class_increment_dataset.py` | 新增类别与标注测试 |
| `tests/test_yolo_monster_viewer.py` | 类别校验、去重和跟踪测试 |
| `training_data/seven_class_increment_v1/` | 七类训练数据与元信息 |
| `training_runs/seven_class_increment_v1/weights/best.pt` | 当前七类模型 |
| `probe_output/new_mushroom_live_1280_preview.png` | 僵尸/刺蘑菇实时检测预览 |
| `probe_output/new_mushroom_live_1280_summary.json` | 对应检测框和计数 |

模型 SHA256：

```text
0F2C43DCFE02593ADB21C0F717AF9045863D9E9DC06C00B2913072F54906CBC1
```

### 3.3 数据和结果

- 新类别源图包含 `3` 个僵尸蘑菇、`2` 个刺蘑菇人工框。
- 新增训练图 `280` 张，验证图 `32` 张；它们都是同一源场景的增强样本。
- 训练配置：18 epochs、`imgsz=640`、batch 8、seed 41。
- 模型由 `training_runs/five_class_hard_v2/weights/best.pt` 继续训练。
- 汇总指标：precision `0.9855`、recall `0.9798`、mAP50 `0.995`、mAP50-95 `0.9938`。
- 上述验证分数不能视为跨地图验证，尤其是新增两类来自同一张源场景。

### 3.4 只查看僵尸蘑菇和刺蘑菇

```powershell
cd "C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp"
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\yolo_monster_viewer.py `
  --cfg shanda_legacy `
  --model training_runs\two_class_real_v2_1280\weights\best.pt `
  --monster-labels zombie_mushroom thorn_mushroom `
  --player-name 麻超圆 `
  --confidence 0.75 `
  --image-size 1280 `
  --fps-limit 12 `
  --device 0
```

`yolo_monster_viewer.py` 当前默认值已经与上面模型、两类、`0.75` 和
`1280` 一致。保留完整参数是为了让实验记录自描述。

在检测窗口内按 `Q`、`Esc` 或点击右上角 `X` 退出。

传入 `--player-name 麻超圆` 时，查看器默认在后台用 RapidOCR 精确读取姓名并
作为身份锚点；主线程仍用轻量模板和轨迹逐帧定位。启动后约需 1-3 秒取得首次
姓名锚点，在此之前宁可显示 `player MISSED`，不会用相似名牌冒充玩家。

### 3.5 玩家与怪物坐标

查看器会为玩家和怪物建立统一的屏幕坐标记录：

- 玩家实体号固定为 `P1`；后台 OCR 精确确认“麻超圆”，随后通过
  `nametag/麻超圆_player.png` 和局部轨迹定位。
- 怪物实体号为 `M1/M2/...`，跨帧匹配成功时保持不变。
- 框内圆点是中心坐标，箭头表示当前估计速度方向。
- 右侧 `ENTITY COORDINATES` 面板显示 `xy`、速度 `v`、运动状态和怪物相对玩家的 `rel/distance`。
- `UP/DOWN` 表示明显的屏幕纵向运动，`MOVE` 表示其他明显位移，`STILL` 表示低速；这不是动作语义分类器。
- `PRED` 表示当前帧漏检后由轨迹预测保留，主要用于动作换帧、特效遮挡或跳跃时减少框闪烁。

坐标系为捕获窗口的画面像素：左上角是 `(0, 0)`，x 向右增加，y 向下增加。JSON 同时写入 `[0,1]` 归一化坐标。它不是地图世界坐标；摄像机滚动时所有屏幕坐标都会变化。

坐标验证产物：

```text
probe_output/entity_coordinates_probe.png
probe_output/entity_coordinates_probe_raw.png
probe_output/entity_coordinates_probe_summary.json
```

最后一次无窗口验证定位到 `P1`，玩家模板置信度约 `0.71`，整体约 `11.8 FPS`。JSON 中每个实体包含：

```text
entity_id, box, center_px, center_norm, velocity_px_s, speed_px_s,
motion_state, tracking_state, missed_frames, relative_to_player
```

## 4. 当前地图三类地形检测

### 4.1 类别和标注定义

| ID | 类别 | 当前源图标注数 | 定义 |
|---:|---|---:|---|
| 0 | `ladder` | 2 | 可攀爬梯子整体 |
| 1 | `rope` | 1 | 可攀爬绳子整体 |
| 2 | `platform` | 5 | 一段连续、可站立、当前可见的平台/台阶地形 |

源截图是 `probe_output/new_mushroom_live_raw.png`，训练时使用的游戏画面区域为 `1370 x 687`。人工框位于 `tools/build_terrain_yolo_dataset.py` 的 `SOURCE_BOXES`，审核预览为：

```text
training_data/terrain_three_class_v1/source_annotations.png
```

### 4.2 关键文件

| 路径 | 用途 |
|---|---|
| `tools/build_terrain_yolo_dataset.py` | 从已审核当前地图截图生成三类训练/验证数据 |
| `tools/train_small_yolo.py` | 通用小型 YOLO 训练和验证入口 |
| `tools/infer_terrain_yolo.py` | 只读取已保存图片的离线推理工具 |
| `tools/yolo_terrain_viewer.py` | 只读实时窗口捕获、三类推理、跟踪和画框 |
| `tests/test_build_terrain_yolo_dataset.py` | 类别、标注数和裁剪保留测试 |
| `tests/test_infer_terrain_yolo.py` | 静态图片读写和绘制测试 |
| `tests/test_yolo_terrain_viewer.py` | 模型类别校验和实时绘制计数测试 |
| `training_data/terrain_three_class_v1/` | 480 张训练图、54 张验证图和元信息 |
| `training_runs/terrain_three_class_v1/weights/best.pt` | 当前地图三类模型 |

模型 SHA256：

```text
74AEF61A41A2D6FCFC50F858E3315E0E4960BAECD1BA8BC9C7C0A08AD63B8A33
```

### 4.3 训练复现

重新生成数据：

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\build_terrain_yolo_dataset.py `
  --source probe_output\new_mushroom_live_raw.png `
  --output training_data\terrain_three_class_v1 `
  --gameplay-height 687 `
  --train-per-class 160 `
  --val-per-class 18 `
  --seed 53
```

训练：

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\train_small_yolo.py `
  --data training_data\terrain_three_class_v1\data.yaml `
  --model .yolo_runtime\yolo11n.pt `
  --project training_runs `
  --name terrain_three_class_v1 `
  --epochs 20 `
  --imgsz 960 `
  --batch 4 `
  --device 0 `
  --seed 53
```

训练汇总位于 `training_runs/terrain_three_class_v1/evaluation_summary.json`。同源验证指标是 precision `0.9993`、recall `1.0`、mAP50 `0.995`、mAP50-95 `0.9925`。

### 4.4 静态推理

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\infer_terrain_yolo.py `
  --source probe_output\new_mushroom_live_raw.png `
  --model training_runs\terrain_three_class_v1\weights\best.pt `
  --output probe_output\terrain_three_class_preview.png `
  --summary probe_output\terrain_three_class_summary.json `
  --gameplay-height 687 `
  --imgsz 960 `
  --conf 0.25 `
  --iou 0.45 `
  --device 0
```

已验证静态结果：`2 ladder / 1 rope / 5 platform`。

### 4.5 实时只读查看器

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\yolo_terrain_viewer.py `
  --cfg shanda_legacy `
  --model training_runs\terrain_three_class_v1\weights\best.pt `
  --confidence 0.25 `
  --image-size 960 `
  --fps-limit 12 `
  --device 0
```

窗口标题：`YOLO Terrain Detector - OBSERVE ONLY`。按 `Q`、`Esc` 或点击 `X` 退出。

最后一次 3 秒实时探测：22 帧、约 `11.61 FPS`，当前视野识别到 `2 ladder / 1 rope / 4 platform`。预览和结构化结果：

```text
probe_output/terrain_live_probe.png
probe_output/terrain_live_probe_summary.json
```

实时结果少一个平台是当前视野/遮挡状态与源标注帧不同，不应通过降低阈值直接补数量；应先采集独立实时帧并人工复核。

## 5. 跟踪逻辑

`tools/yolo_monster_viewer.py` 的 `DetectionTracker` 被两个实时查看器复用：

- 以同类别 IoU 或归一化中心距离进行匹配。
- 怪物匹配还要求相对宽度、高度和面积变化不超过 `1.70 / 2.00 / 2.50`，
  尺寸变化同时计入匹配代价；这是逐轨迹相对约束，不是绝对高度过滤。
- 指数平滑检测框和置信度。
- 使用速度预测下一帧位置。
- 短暂漏检时保留轨迹，减少检测框闪烁。
- 怪物默认最多保留 4 个漏检帧；地形默认最多保留 6 个漏检帧。

地形是静态目标，若后续进一步增强稳定性，应优先采集多帧并做固定地图锚点融合，而不是无限延长漏检保留时间。

## 6. 测试

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" -m unittest `
  tests.test_build_seven_class_increment_dataset `
  tests.test_yolo_monster_viewer `
  tests.test_build_terrain_yolo_dataset `
  tests.test_infer_terrain_yolo `
  tests.test_yolo_terrain_viewer `
  -v
```

还应做静态安全检查，确认只读工具没有输入控制依赖：

```powershell
rg -n "pyautogui|pynput|keybd_event|SendInput|SetForegroundWindow" `
  tools\yolo_monster_viewer.py `
  tools\yolo_terrain_viewer.py `
  tools\infer_terrain_yolo.py
```

注：源码注释会出现 `keyboard`/`mouse` 字样，用于说明“不发送输入”；应检查实际 import 和调用，而不是只按关键词计数。

## 7. 已知限制和下一步

1. 地形数据的训练集和验证集都来自同一张当前地图截图的增强裁剪，只能说明当前地图效果。
2. 僵尸蘑菇和刺蘑菇的新增数据也来自单一源场景，七类模型的高分不能证明跨地图能力。
3. 新 agent 若要评估泛化，应先采集不同时间、不同角色位置、不同遮挡、不同窗口大小和至少一张不同地图的独立测试帧；这些帧不能回流到训练后再继续称为测试集。
4. `platform` 目前按“连续可站立地形段”标注，不是按单个石块、贴图或碰撞多边形标注。更改定义前应新建数据版本，不能原地混标。
5. `training_data/`、`training_runs/`、`.yolo_runtime/` 被 `.gitignore` 忽略；模型和数据只存在于本机工作区。若迁移到另一台机器，需要单独复制并核对上面的 SHA256。
6. 中文路径下 OpenCV 文件输出应继续使用 `cv2.imencode(...).tofile(...)`，不要直接依赖 `cv2.imwrite`。

## 8. 进程检查和停止

检查查看器：

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -in @('python.exe', 'pythonw.exe') -and
  $_.CommandLine -match 'yolo_(monster|terrain)_viewer\.py'
} | Select-Object ProcessId, Name, CommandLine
```

仅在窗口无法正常退出时，精确停止这两个查看器：

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -in @('python.exe', 'pythonw.exe') -and
  $_.CommandLine -match 'yolo_(monster|terrain)_viewer\.py'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

正常情况下优先在查看器窗口按 `Q`、`Esc` 或点击 `X`，这样可以执行清理逻辑并在指定参数存在时保存最后一帧和 JSON 汇总。

## 9. 两类精度改进（2026-08-14 第二版）

### 9.1 根因和代码修正

`tools/codex_monster_detector.py` 的精度下降不是七类权重损坏。离线同帧
A/B 证明主要有三个原因：

1. `imgsz=640` 会漏掉当前画面中宽度只有约 50-70 px 的怪物；当前配置改为
   `1280`。
2. 旧适配器用玩家中心 `+/-60 px` 直接删除上下层怪物和跳跃怪物；现在默认
   保留全图，并输出 `same_level` 标记。只有显式传入
   `same_level_only=True` 才过滤。
3. 旧跟踪器会把一次低置信度误检预测 4 帧。现在低置信度新轨迹必须连续命中
   2 帧；置信度达到 `0.75` 时允许首帧确认；未确认轨迹永远不输出 `PREDICTED`
   框。

`YoloMonsterDetector` 的类别校验也已改为校验当前选中的类别，因此既能加载
七类模型，也能加载只有以下两个 class ID 的专用模型：

| class ID | class name |
|---:|---|
| 0 | `zombie_mushroom` |
| 1 | `thorn_mushroom` |

不要把专用模型继续按旧七类模型的 ID `5/6` 解释。

### 9.2 数据集隔离

新增文件：

```text
training_data/two_class_real_v2_manifest.json
tools/build_two_class_monster_dataset.py
tools/evaluate_monster_yolo.py
```

生成的数据位于 `training_data/two_class_real_v2/`。原始帧分组固定如下：

- train：3 个源组、22 个源标注，生成 179 张全图/裁剪图。
- val：1 个独立源帧、2 个刺蘑菇，包含强攻击特效和边缘残缺目标。
- test：1 个独立源帧、7 个目标，包含重叠、跳跃、受击数字和遮挡。

同一原始帧不能跨 split；只有 train 源会生成裁剪。构建器会拒绝检测到的源帧
泄漏。复现命令：

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" `
  tools\build_two_class_monster_dataset.py `
  --manifest training_data\two_class_real_v2_manifest.json `
  --output training_data\two_class_real_v2 `
  --crops-per-box 8 `
  --seed 73
```

### 9.3 训练和阈值选择

训练从七类基线权重继续，配置为 `imgsz=1280`、batch 4、seed 73、最多
30 epochs；验证早停后实际完成 13 epochs。输出：

```text
training_runs/two_class_real_v2_1280/weights/best.pt
SHA256 3E2A166A0F728980358B7BC8D6D1AABB0FF6FBB6266A95D0103CBA3E2EA719B2
```

阈值只在 val 上扫描 `0.25 / 0.50 / 0.75 / 0.85`，没有用 test 选择。
`0.75` 的 val 结果为 `TP=2 / FP=0 / FN=0`，因此锁定为推荐阈值。
扫描记录：

```text
probe_output/two_class_threshold_scan_val.json
```

### 9.4 固定测试 A/B

评估条件：`imgsz=1280`、NMS IoU `0.45`、匹配 IoU `0.50`。旧七类基线使用
原阈值 `0.10`，专用模型使用只在 val 选出的 `0.75`。

| model | threshold | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 七类基线 | 0.10 | 4 | 0 | 3 | 1.0000 | 0.5714 | 0.7273 |
| 两类专用 | 0.75 | 6 | 0 | 1 | 1.0000 | 0.8571 | 0.9231 |

分类别结果：僵尸蘑菇仍为 `3/3`；刺蘑菇由 `1/4` 提升到 `3/4`。剩余漏检
是左上两个高度重叠刺蘑菇中的一个。完整记录和画框预览：

```text
probe_output/two_class_final_comparison_test.json
probe_output/two_class_final_previews/test/
```

复现：

```powershell
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" `
  tools\evaluate_monster_yolo.py `
  --data training_data\two_class_real_v2\data.yaml `
  --split test `
  --model baseline=training_runs\seven_class_increment_v1\weights\best.pt `
  --model candidate=training_runs\two_class_real_v2_1280\weights\best.pt `
  --model-confidence baseline=0.10 `
  --model-confidence candidate=0.75 `
  --image-size 1280 `
  --device 0 `
  --output probe_output\two_class_final_comparison_test.json
```

### 9.5 集成要求和限制

推荐的观察型集成参数：

```text
model=training_runs/two_class_real_v2_1280/weights/best.pt
labels=zombie_mushroom,thorn_mushroom
imgsz=1280
confidence=0.75
track_min_hits=2
track_high_confidence=0.75
same_level_only=false
```

不要启动七类标签但加载两类权重；不要在模型输出后按玩家 y 坐标覆盖全图检测；
不要用 test 继续调整阈值。当前 val 和 test 各只有一个源帧，且都来自同一张
洞穴地图，因此结果只证明当前地图的小样本回归改进，不是跨地图准确率。今后模型
有任何训练、阈值或后处理变更，必须采集新的未见源帧作为 test；当前 test 已经被
本轮评估消耗，只能继续作为回归集。

本轮没有启动实时查看器。新增/修改感知模块的定向测试共 32 项通过，Python 编译
检查和 `git diff --check` 通过。全仓 `132` 项测试中另有 5 项既有自动战斗/控制
测试失败、1 项跳过；失败涉及未在本轮修改的战斗建议、暂停、绘图和路线跟随代码，
没有为追求全绿而改动这些控制链。

## 10. 玩家身份与框尺寸一致性更新（2026-08-14 第三版）

### 10.1 玩家身份链路

旧实现只对 `麻超圆_player.png` 做一次全图模板匹配并取最高分。名牌背景、聊天文字
和其他玩家名字具有大量相似灰度结构，在多人地图会选错人。新链路为：

```text
后台 RapidOCR 精确读取“麻超圆”
  -> 姓名坐标成为身份锚点
  -> 锚点附近多候选模板匹配
  -> 名字字形相似度重排
  -> DetectionTracker / P1 坐标
```

- OCR 在独立后台线程运行，2 个 ONNX CPU 线程，每次完成后间隔 3 秒；不会阻塞
  YOLO 主循环。
- 未取得精确姓名锚点时不输出玩家框；不会退化为共享称号或最近的其他玩家。
- 获得锚点后只在 120 px 半径内局部重找，短时遮挡由玩家轨迹保留 8 帧。
- `--no-player-ocr` 可显式关闭身份 OCR，但多人场景不建议使用。
- 玩家 JSON 新增 `template_score`、`glyph_score`、`identity_score` 和
  `identity_mode`；汇总新增 `player_identity.ocr_enabled/error/identity_seeded`。

四张历史原始帧的独立全局定位和连续锁定均命中人工确认的“麻超圆”坐标。最新
多人实时探测也由 OCR 以约 `0.998` 读到“麻超圆”，最终玩家名牌框为
`[685, 466, 41, 23]`，约 `11.88 FPS`。验证文件：

```text
probe_output/identity_size_gate_live_probe_v3.png
probe_output/identity_size_gate_live_probe_v3_raw.png
probe_output/identity_size_gate_live_probe_v3_summary.json
```

### 10.2 怪物轨迹尺寸门槛

`DetectionTracker` 现在按同一轨迹上一帧的框计算对称尺寸比。默认门槛：

```text
max_width_ratio=1.70
max_height_ratio=2.00
max_area_ratio=2.50
size_cost_weight=0.25
```

合理的跳跃/受击姿态变化仍可沿用原 `track_id`；同类别候选若突然变成明显不同尺寸，
不会强行关联，而会等待确认或建立新轨迹。`CodexMonsterDetector` 已透传
`track_max_*` 参数。`level_band`、`same_level_only` 和配置中的同层/高度参数没有
修改。

尺寸门槛不是分类器，不能消除跨地图外观相似的稳定误检。最新城镇探测中，洞穴
专用两类模型会把大型蘑菇房屋误判为刺蘑菇；这再次说明该模型只应在训练/测试覆盖
的洞穴地图使用，若要支持城镇必须新增独立标注和负样本后重新评估。

### 10.3 验证状态

- 感知相关定向测试：`30/30` 通过。
- Python 编译：`tools/yolo_monster_viewer.py`、
  `tools/codex_monster_detector.py` 通过。
- 全仓发现 `137` 项测试，其中本次相关项通过；另有 `6` 个既存失败、`1` 个跳过，
  位于未修改的自动战斗/旧 advisory 行为，未为本次感知改动调整。

## 11. 勇士部落木妖单类检测（2026-08-15）

### 11.1 数据与模型

本轮只做只读截图、YOLO 检测、轨迹和坐标，不连接 `auto_combat.py`，也不发送
键盘或鼠标输入。木妖单类数据及构建器：

```text
training_data/warrior_stump_v1_manifest.json
training_data/warrior_stump_v1/
tools/build_warrior_stump_dataset.py
```

原始帧按来源隔离：train 3 个正样本源帧加 1 个城镇负样本，32 个训练源标注，
生成 164 张全图/裁剪图；val 1 帧 10 只；test 1 帧 9 只。只有 train 源会生成
派生裁剪，val/test 不参与训练。当前数据来自同一张勇士部落地图、同一短会话。

训练从七类模型继续，配置为 30 epochs、`imgsz=1280`、batch 4、seed 89：

```text
training_runs/warrior_stump_v1_1280/weights/best.pt
SHA256 D5EAFA92CD086F558FDE320C7E39984C94B8C861B67B6C34CF5D86512B8C8CAC
```

Ultralytics val 指标为 precision `1.0000`、recall `0.9932`、mAP50 `0.9950`、
mAP50-95 `0.7623`。完整摘要：

```text
training_runs/warrior_stump_v1_1280/evaluation_summary.json
```

### 11.2 阈值与固定测试

候选模型只在 val 扫描 `0.10 / 0.25 / 0.50 / 0.75`。`0.25` 得到
`TP=10 / FP=0 / FN=0`，因此锁定为实时推荐阈值。各次扫描记录在：

```text
training_runs/warrior_stump_candidate_val_conf_*.json
```

固定 test 使用 `imgsz=1280`、NMS IoU `0.45`、匹配 IoU `0.50`；旧七类基线
采用其 val 最佳阈值 `0.05`，新单类模型采用锁定阈值 `0.25`：

| model | threshold | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 七类基线 | 0.05 | 6 | 2 | 3 | 0.7500 | 0.6667 | 0.7059 |
| 木妖单类 | 0.25 | 8 | 1 | 1 | 0.8889 | 0.8889 | 0.8889 |

剩余错误位于测试帧顶部四只高度重叠的木妖：模型重复预测最左目标一次，并漏掉其
右侧被严重遮挡的目标。实时查看器会按同类框 IoU 去重，因此不会显示重复框，但
该帧去重后仍只有 8 只。报告和预览：

```text
training_runs/warrior_stump_v1_1280/fixed_test_report.json
training_runs/warrior_stump_v1_1280/fixed_test_previews/
```

`tools/evaluate_monster_yolo.py` 已从写死的洞穴两类改为读取 `data.yaml` 中的类别，
因此可正确评估单类木妖及其他数据集。

### 11.3 只读实时启动

```powershell
cd "C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp"
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" `
  tools\yolo_monster_viewer.py `
  --cfg shanda_legacy `
  --model training_runs\warrior_stump_v1_1280\weights\best.pt `
  --monster-labels 木妖 `
  --confidence 0.25 `
  --iou 0.45 `
  --image-size 1280 `
  --fps-limit 12 `
  --track-min-hits 2 `
  --track-high-confidence 0.75 `
  --player-name 麻超圆
```

按查看窗口中的 `q` 或 `Esc` 退出。`木妖` 已映射到模型类别 `stump`。配置中的
`ui_y_start=687` 以及既有 `level_band`、`same_level_only`、尺寸门槛均未修改。

本轮实时无界面探测运行 6 秒、67 帧、约 `11.9 FPS`，最后一帧维护 13 条木妖
轨迹；客户端捕获尺寸为 `1370x687`，模型输入为 `1280`，查看器自动保持纵横比。
验证文件：

```text
probe_output/warrior_stump_live_v1.png
probe_output/warrior_stump_live_v1_raw.png
probe_output/warrior_stump_live_v1_summary.json
```

该模型只验证了当前勇士部落场景，不应宣称跨地图泛化。若要继续处理极端重叠，
应采集新的训练/验证源帧，并保留当前 test 只作为已消耗的回归集；不得用当前 test
反复选择阈值或训练配置。
