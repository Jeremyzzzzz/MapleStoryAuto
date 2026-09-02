# MapleStoryAutoLevelUp 项目完整交接文档

> 交接日期：2026-08-26
> 交接人：WorkBuddy（**交接后不再由 WorkBuddy 维护**，后续由 codex 接手）
> 项目路径：`C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp`
> 运行环境：Windows + Python venv `C:\quant_lab\lstm_gpu_venv\`（注意：该 venv 的 `python.exe` 是启动器，实际真身是 `codex-primary-runtime` 的 python）

---

## 一、项目定位

冒险岛怀旧服（盛大 shanda_legacy）**自动打怪/自动升级**脚本。技术路线：
- **玩家定位**：color_anchor（蓝色称号条"中级冒险家勋章" + 红色身体锚点），no-ocr 模式长期固定
- **怪物识别**：YOLO 双模型并行（野猪 wild_boar + 树妖 stump/木妖）
- **移动巡航**：小地图坐标航点录制/回放（minimap_patrol），18 段路线
- **其他玩家检测**：小地图红点（R1/R2），检测到即挂机（只喝药不攻击），红点消失 5 分钟或按 F8 恢复
- **反检测**：30 分钟战斗 + 5 分钟休息（minimap_patrol 模式跳过）、单实例文件锁、12fps 帧率上限、拟人化参数

---

## 二、启动方式（二选一）

### 方式 A：桌面启动脚本（推荐）
双击 `C:\Users\Administrator\Desktop\自动打猪.bat`（v8，单斜杠路径，完整参数已固化）。

### 方式 B：命令行
```powershell
Set-Location "C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp"
& "C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\auto_combat.py `
  --cfg shanda_legacy `
  --monster-backend yolo `
  --yolo-model training_runs\wild_boar_real_hardneg_v4_960\weights\best.pt `
  --yolo-confidence 0.10 --yolo-iou 0.70 --yolo-image-size 960 `
  --no-color-verify --show-viz `
  --player-name 麻超圆 --fps-limit 12 `
  --no-ocr --monster-labels wild_boar --no-terrain `
  --mode minimap_patrol --map-name "野猪的领土！！"
```

**启动后步骤**：等 1~2 分钟（双 YOLO 模型加载慢）→ 出现 `[status]` 日志 = 就绪 → 点游戏窗口聚焦（`游戏聚焦=Y`）→ F8 恢复 → F4 巡航。

---

## 三、模块结构

### 核心入口
| 文件 | 作用 |
|---|---|
| `tools/auto_combat.py` | **主脚本**（约 6800 行）：玩家定位 wrapper、怪物检测、攻击决策、巡航回放、喝药、可视化、热键 |
| `tools/yolo_monster_viewer.py` | 检测引擎库：ReadOnlyPlayerDetector(color_anchor)、YoloMonsterDetector、小地图玩家/红点检测、MinimapRedMarkerTracker |

### 检测工具（独立观察器）
| 文件 | 命令要点 | 用途 |
|---|---|---|
| `tools/minimap_viewer.py` | `python -m tools.minimap_viewer --cfg shanda_legacy --window-title-token '冒险岛怀旧服'` | 只读观察：显示黄点(P1)/红点(R1/R2)坐标，不操作 |
| `tools/yolo_monster_viewer.py`（独立跑） | `python tools\yolo_monster_viewer.py --cfg shanda_legacy --model <模型> --monster-labels <标签> --no-player-ocr` | 单模型观察器（树妖模型验证用） |
| `tools/live_perception_viewer.py` | - | 实时感知观察 |
| `tools/debug_player_detect.py` / `debug_live_player.py` | - | 玩家定位调试 |

### 数据构建工具（训练用，均 `tools/build_*.py`）
`build_wild_boar_real_hardneg_dataset.py`、`build_warrior_stump_dataset.py`、`build_pig_yolo_dataset.py`、`build_player_identity_yolo_dataset.py`、`build_terrain_yolo_dataset.py` 等——从游戏帧截图构建 YOLO 训练集。

### 引擎/状态机（早期版本，auto_combat 已替代主入口）
`src/engine/MapleStoryAutoLevelUp.py`（旧主逻辑）、`src/states/*.py`（状态机）、`src/input/GameWindowCapturor.py`（窗口捕获，auto_combat 复用）、`src/input/KeyBoardController.py`（按键模拟）。

### 资源目录
| 目录 | 内容 |
|---|---|
| `nametag/` | 玩家名字模板（`麻超圆_player.png` 等） |
| `monster/` | 怪物 sprite 图（模板匹配用，按中文名分目录） |
| `minimaps/野猪的领土！！/routes/*.json` | **巡航路线文件**（录制的小地图航点序列） |
| `training_runs/` | YOLO 模型权重 |
| `log/` | 运行日志 `MSBot_*.log`、锁文件相关 |
| `config/` | 配置 yaml |

---

## 四、关键配置（config/config_shanda_legacy.yaml）

| 段 | 关键内容 |
|---|---|
| `bot` | 角色、HP/MP 药键、攻击键 |
| `key` | 按键绑定（directional_attack=d、hp_potion=c、mp_potion=v、feed=N） |
| `health_monitor` | 血量监控、OCR 区域 |
| `minimap` | 小地图标定(canvas_region)、玩家黄点、**其他玩家红点参数**（other_player_*） |
| `minimap_waypoint` | 航点巡航参数（掉层恢复容差等） |
| `auto_combat` | yolo_confidence、**stump_*（树妖模型参数）**、sprite 阈值 |
| `character_profiles` | 角色档案（麻超圆：攻击范围 100x120、保持距离 20） |
| `combat_advisory` | 攻击目标选择 |

**树妖配置段**（stump_model/stump_confidence=0.70/stump_iou=0.45/stump_image_size=1280）——第二 YOLO 模型参数，可在此调。

---

## 五、关键模型

| 模型 | 类别 | 用途 |
|---|---|---|
| `training_runs/wild_boar_real_hardneg_v4_960/weights/best.pt` | wild_boar | 野猪检测（主，conf 0.10 960px） |
| `training_runs/warrior_stump_hardneg_v2_1280/weights/best.pt` | stump | 树妖/木妖检测（辅，conf 0.70 1280px） |

---

## 六、核心机制速览

1. **color_anchor 玩家定位**：蓝色称号条(HSV 90-135, 80+, 50+) + 几何过滤(宽80~180高8~28, 宽高比≥4, 面积≥300) + 红分验证(红分≥0.02) + 同Y相邻条合并(抗宠物遮挡) + 颜色参照(自己称号条平均色滤地形)。黄框位置 = 称号条中心x / 称号条y-24-30。
2. **EMA 平滑**：wrapper 层 `_smoothed_box`，静止 alpha=0.35、移动 0.85，只平滑位置（尺寸直取检测值）。
3. **丢失保持**：`color_anchor_hold`（keep_color_anchor_misses=6 帧保持上帧位置）+ 小地图运动估计（mini_norm_x→屏幕x 线性映射，丢失时用 mini 估算）。
4. **巡航掉层保护**（8-25 修复，保留）：move 段仅当**目标在上方**（dy<-0.08）才触发同Y恢复；目标在下方（走下坡/掉下去）是正常路径不触发。
5. **其他玩家红点**：`locate_minimap_players` → HSV 红 + 通道主导过滤 → `MinimapRedMarkerTracker`(2帧确认) → 挂机(只喝药 suppress_feed)。
6. **单实例锁**：`%TEMP%\auto_combat_ms.lock`（msvcrt 文件锁），进程退出自动释放。

---

## 七、已知问题（交接时未解决）

1. **黄框飘**（核心遗留问题）：角色不动时黄框坐标/尺寸跳变（16:05 日志实测 266→279 跳、框宽 41~78）。直接原因：color_anchor 频繁丢失（跳跃时红分 0.19 接近阈值 0.02）→ 回退 local 模板匹配（不稳定）。
   - **用户诉求**：黄框永远钉在红色勋章框上方固定位置（公式：玩家中心x=称号条中心x，玩家中心y=称号条y-54）。
   - 排查方向见 `HANDOFF_2026-08-26.md` 第五节（红分低是核心矛盾：丢失时没有 anchor_box 可锚定）。
2. 其余问题（树妖识别精度、红点误检边界）待实测。

---

## 八、常见运维操作

| 操作 | 方法 |
|---|---|
| 重启 | 杀进程（**杀内存大的 codex 真身**，壳进程只有 8MB）→ 删 `%TEMP%\auto_combat_ms.lock` → 双击 bat |
| 看日志 | `log/MSBot_*.log`（滚动）或项目根 `bot_run.log` |
| 录新路线 | F1 开始录制→手动走→F2 结束→保存到 `minimaps/<地图>/routes/` |
| 换地图 | 改 `--map-name`（必须精确匹配路线目录名） |
| 调红点灵敏度 | config minimap 段 `other_player_*` 参数 |
| 验证树妖模型 | `python tools\yolo_monster_viewer.py --cfg shanda_legacy --model training_runs\warrior_stump_hardneg_v2_1280\weights\best.pt --monster-labels 木妖 --no-player-ocr` |

---

## 九、交接给后续维护者

- 用户对"黄框飘"问题有明确预期（见第七节），接手后**先复现再动手**，每次改动经真实游戏帧验证。
- 用户偏好：中文注释/界面、参数写入 config 可调、保留红框可视化排查误检、不靠上下文拼启动命令（用 bat）。
- 相关文档：`STARTUP.md`（启动说明）、`HANDOFF_2026-08-26.md`（回退点+黄框问题详析）、`PROJECT_HANDOFF.md`（早期交接）。
- **本交接后 WorkBuddy 不再维护本项目。**
