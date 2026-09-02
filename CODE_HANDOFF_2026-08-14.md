# 冒险岛自动打怪系统 — 代码交接文档

> 整理时间：2026-08-14
> 项目目录：`C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp`
> Python：`C:\quant_lab\lstm_gpu_venv\Scripts\python.exe`（系统 python 无 cv2，必须用这个）

---

## 1. 系统概述

这是一个《冒险岛怀旧服》的自动打怪（挂机）脚本。核心能力：

1. **感知**：用 YOLO 检测怪物（僵尸蘑菇、刺蘑菇等），用名字框模板匹配定位玩家
2. **决策**：定向清怪（沿一个方向推进打怪）、被攻击反击
3. **执行**：模拟键盘输入（移动、攻击、喝药、跳跃）

当前正处于"检测效果调试"阶段，怪物检测已由 codex 训练了专用模型。

---

## 2. 运行方式（当前命令）

```bash
cd "C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp"

"C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools/auto_combat.py \
  --cfg shanda_legacy \
  --monster-backend yolo \
  --yolo-model training_runs/two_class_real_v2_1280/weights/best.pt \
  --yolo-confidence 0.75 \
  --yolo-image-size 1280 \
  --no-color-verify \
  --show-viz \
  --player-name 麻超圆 \
  --fps-limit 12 \
  --no-ocr \
  --monster-labels 僵尸蘑菇 刺蘑菇
```

**热键**：
| 按键 | 功能 |
|---|---|
| F8 | 启动/暂停打怪（启动默认暂停） |
| F1 | 开始/停止录制路线 |
| F2 | 清空重录路线 |
| F3 | 保存路线 |
| F9 | 退出 |

---

## 3. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  tools/auto_combat.py  (主程序，约 4800 行)                  │
│                                                             │
│  捕获层  GameWindowCapturor (WindowsCapture 抓游戏窗口帧)     │
│      ↓                                                      │
│  感知层  ┌─ 玩家检测: ReadOnlyPlayerDetector (codex)         │
│          └─ 怪物检测: CodexMonsterDetector (codex 封装)       │
│      ↓                                                      │
│  决策层  ┌─ AdvisoryEvaluator: 选目标(前方过滤+锁定)          │
│          └─ CombatPolicy: 决策(攻击/移动/转身/反击)           │
│      ↓                                                      │
│  执行层  CombatExecutor: 转成键盘输入(移动/攻击/喝药)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 数据流

```
抓帧(原始帧 1926×1130, DPI 1.5x)
  → 怪物检测 CodexMonsterDetector.detect(frame, player)
      → YoloMonsterDetector.detect (YOLO 推理, imgsz=1280 letterbox)
      → DetectionTracker.update (平滑 + 漏检桥接 + 确认)
      → EntityCoordinateTracker.update (中心坐标/速度/运动状态)
      → 同层过滤(可选) + 相对玩家坐标 + 字段转换
  → 玩家检测 ReadOnlyPlayerDetector (名字框模板匹配)
  → AdvisoryEvaluator.evaluate(player, monsters, facing) → 选目标(前方)
  → CombatPolicy.decide(player, advisory, hp, mp) → command
  → CombatExecutor.execute(command) → 键盘输入
```

---

## 5. 关键文件清单

| 文件 | 用途 | 说明 |
|---|---|---|
| `tools/auto_combat.py` | 主程序 | 集成调用 + 决策 + 执行 + 可视化 |
| `tools/codex_monster_detector.py` | **怪物检测插件** | 封装 codex 检测 + 同层过滤 + 字段转换 |
| `tools/yolo_monster_viewer.py` | codex 检测原版 | YoloMonsterDetector / DetectionTracker / EntityCoordinateTracker / ReadOnlyPlayerDetector（**原封不动用 codex 的，不要改**） |
| `tools/live_perception_viewer.py` | 感知库 | AdvisoryEvaluator(选目标) / read_vitals / 可视化 |
| `src/input/GameWindowCapturor.py` | 抓帧 | WindowsCapture 封装 |
| `src/engine/HealthMonitor.py` | 血条检测 | HP/MP/EXP 读取 |
| `config/config_shanda_legacy.yaml` | 配置 | 攻击范围/阈值等 |

---

## 6. 感知层（检测）详解

### 6.1 怪物检测 —— `CodexMonsterDetector`

文件：`tools/codex_monster_detector.py`

```python
class CodexMonsterDetector:
    def __init__(self, model_path, confidence, iou=0.45, device="0",
                 image_size=1280, labels=None, level_band=60.0,
                 same_level_only=False, min_confirmed_hits=2,
                 high_confidence_confirm=0.75, ui_y_start=687):
        # codex 原版三件套（import 自 yolo_monster_viewer.py）
        self.core = YoloMonsterDetector(...)      # YOLO 推理
        self.tracker = DetectionTracker(max_missed=4, ...)  # 平滑+漏检桥接
        self.coord = EntityCoordinateTracker()    # 坐标/速度

    def detect(self, frame, player=None):
        dets = self.core.detect(frame, self.ui_y_start)   # YOLO 检测
        dets = self.tracker.update(dets)                   # 跟踪
        dets = self.coord.update(...)                      # 坐标
        # 同层过滤(same_level_only=True 时): 只留玩家水平带 ±level_band
        # 字段转换: confidence→score, box list→tuple
        return out  # [{label, score, box, entity_id, center_px, ...}]
```

**检测核心 100% 是 codex 的类**（YoloMonsterDetector/DetectionTracker/EntityCoordinateTracker），本插件只加了：同层过滤 + 字段转换。

### 6.2 玩家检测 —— `ReadOnlyPlayerDetector`

文件：`tools/yolo_monster_viewer.py`（codex 原版）

- 用 `nametag/麻超圆_player.png` 名字框模板，`cv2.matchTemplate(TM_CCOEFF_NORMED)` 全图匹配
- 阈值 0.30
- 集成方式：`auto_combat.py` 里包了一层 `_CodexPlayerWrapper`（加 P1 实体坐标 + 简单 sticky）

---

## 7. 决策层（打怪逻辑）详解

文件：`tools/auto_combat.py` 的 `CombatPolicy`，`tools/live_perception_viewer.py` 的 `AdvisoryEvaluator`

### 7.1 定向清怪（核心策略）

战士沿一个方向推进清怪，不检测背后：

1. `CombatPolicy.facing_direction` 记录面朝方向（"left"/"right"）
2. `AdvisoryEvaluator.evaluate(..., facing=facing)` 只在前方（facing 方向）选目标
3. 前方有怪 → 攻击/走向前方最近怪
4. 前方没怪（且背后有怪）→ **连续 1.2 秒确认**后才转身（`_last_front_seen` 延迟，避免短暂漏检误转身）

### 7.2 被攻击反击

利用"被攻击时血条闪烁 → HP 检测读成 0%/None"作为被攻击信号：

```python
# CombatPolicy.decide 开头
if hp_percent is not None and hp_percent > 0.05:
    self._last_hp = hp_percent          # 记录正常 HP
hp_now_missing = (hp_percent is None or hp_percent <= 0.001)
if (hp_now_missing and self._last_hp is not None and self._last_hp > 0.05
        and now - self._last_counter_at >= 0.8):
    self._last_counter_at = now
    direction = self.facing_direction or "left"
    return f"attack_{direction}", "counter_attack"   # 立即反击，无视冷却
```

### 7.3 目标锁定

- 优先用 codex 的稳定 `entity_id`（M1/M2）锁定目标，解决"两只同名怪来回翻转"
- 锁定后坚持锁定坐标（不跟检测框漂）

---

## 8. 关键参数配置

文件：`config/config_shanda_legacy.yaml`

| 参数 | 值 | 说明 |
|---|---|---|
| `attack_horizontal_px` | 90 | 战士攻击横向范围 |
| `attack_vertical_px` | 55 | 战士攻击纵向范围 |
| `keep_distance` | 20 | 近战保持距离（≤30 判定为近战） |
| `tag_threshold` | 0.3 | 玩家模板匹配置信度 |
| `monster_min_interval` | 0.08 | 怪物检测最小间隔（≈12fps，匹配 codex） |
| `monster_max_age` | 2.0 | 检测结果有效期 |
| `codex_level_band` | 60 | 同层过滤的 y 差阈值 |

---

## 9. 上一个 agent 的改动清单

### 已修改的文件

1. **`tools/codex_monster_detector.py`**（新建插件）
   - 封装 codex 检测 + 同层过滤 + 字段转换
   - `max_missed=4`（漏检桥接帧数）

2. **`tools/auto_combat.py`**（主程序，大量改动）
   - yolo backend 改为 `from tools.codex_monster_detector import CodexMonsterDetector`
   - 玩家检测改为 codex 的 `ReadOnlyPlayerDetector`
   - `CombatPolicy.decide`：加了定向清怪（前方过滤+转身）、被攻击反击、转身确认延迟
   - `PauseController`：启动默认暂停（`paused=True`）
   - `RouteRecorderCore`：启动不录制（`is_recording=False`）
   - 目标锁定改用 entity_id

3. **`tools/live_perception_viewer.py`**
   - `AdvisoryEvaluator.evaluate` 加 `facing` 参数（前方过滤）
   - 加 `front_count`/`back_count` 返回

4. **`src/input/GameWindowCapturor.py`**
   - **注释掉启动时的 `resize_window(1296,759)`**（关键！见注意事项）

5. **`config/config_shanda_legacy.yaml`**
   - `tag_threshold` 0.3、`monster_min_interval` 0.08

6. **`src/engine/MapleStoryAutoLevelUp.py`**
   - resize 插值 INTER_NEAREST → INTER_LINEAR

---

## 10. 已知问题与注意事项（重要）

### ⚠️ 1. 窗口尺寸是检测质量的关键（刚踩的坑）

- **auto_combat 启动时如果强制 `resize_window(1296,759)`，会触发游戏窗口锁定，把窗口弹回 864×506** → 游戏渲染分辨率变低 → 怪物像素变小 → YOLO 检测变差
- **codex 的 viewer 不 resize 窗口**，所以保持 1296×759，检测正常
- **修复**：已注释掉 `GameWindowCapturor.py` 的 resize_window，窗口由用户手动保持 1296×759
- **规律**：
  - 窗口 864×506 → DDA 帧 1278×750
  - 窗口 1296×759 → DDA 帧 1926×1130（DPI 1.5x 缩放）
- **不要手动缩小窗口**，否则检测变差

### ⚠️ 2. DPI 缩放

系统 DPI 150%，WindowsCapture 抓的帧是物理像素 × 1.5。窗口 1296×759 时实际抓 1926×1130。

### ⚠️ 3. 模型选择

- 旧七类模型 `seven_class_increment_v1`：对刺蘑菇召回差（1/4）
- **新两类模型 `two_class_real_v2_1280`**：僵尸蘑菇+刺蘑菇，召回好（3/4），当前使用

### ⚠️ 4. 中文路径

读图必须用 `imread_cn`（`src.utils.common`），`cv2.imread` 读不了中文路径。

### ⚠️ 5. 依赖注入

所有脚本顶部要照抄：
```python
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / ".yolo_runtime"))
import os
os.add_dll_directory(str(REPO_ROOT / ".yolo_runtime/pywin32_system32"))
```

---

## 11. 当前运行状态

- 窗口：1296×759（稳定保持）
- 模型：two_class_real_v2_1280（僵尸蘑菇+刺蘑菇）
- 置信度：0.75
- 打怪：定向清怪 + 被攻击反击 + 转身确认延迟
- 启动：默认暂停（F8 启动）
