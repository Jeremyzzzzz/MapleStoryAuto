# MapleStoryAutoLevelUp 项目交接（Agent 版）

> 更新时间：2026-08-13 · 供接手 agent 阅读。本文件替代 `PROJECT_HANDOFF.md`（8-11 旧版）中已过时部分，旧版仍保留在仓库。

## 0. 一页速览

| 项 | 值 |
|---|---|
| 项目 | 冒险岛怀旧服 自动打怪挂机（闭环：识别→寻怪→移动→攻击） |
| 仓库 | `C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp` |
| Python | `C:\quant_lab\lstm_gpu_venv\Scripts\python.exe`（torch 2.12+cu130 / RTX 5080 / **opencv 4.11.0**） |
| 游戏 | 冒险岛怀旧服 · 角色「麻超圆」（战士） · 配置 `shanda_legacy` |
| 当前主线 | 怪物识别方案重构（YOLO 失败 → sprite 贴图模板匹配） |
| 待办主线 | **录制打怪流程**（用户已确认方向，未实现） |

**接手第一步**：`git status --short`，工作区有未提交改动，勿覆盖勿清理。

---

## 1. 已完成且稳定 ✅

### 1.1 玩家定位（`tools/auto_combat.py` 内 PlayerLocator）
- 名字框 41×23 模板 SQDIFF + 绿幕 mask + 上帧局部搜索（主锚点）
- 勋章 96×21 模板（二级锚点，名字被遮挡时兜底）
- 整体上移 35px（dy=35）让框对准角色本体
- 按键感知追踪：记录移动命令 + 130px/s 预测位置 + 220px 容差，防框瞬移
- 启动必须带 `--player-name 麻超圆`，否则走旧 PlayerDetector 接口不兼容主循环

### 1.2 战斗逻辑（CombatPolicy + AdvisoryEvaluator）
- 近战站桩：`keep_distance <= 30` 不躲避
- 朝向限制：攻击/移动更新朝向，目标在背后不算攻击范围
- 目标锁定：`lock_hold_seconds=3.0`，HP 条怪优先，不频繁换目标
- 攻击范围：麻超圆 90×55，保持距离 20，攻击键 d

### 1.3 怪物检测：sprite 贴图模板匹配（本次核心产出）
- 模板库 `monster_templates_final/`：red_snail 79 / blue_snail 62 / stump 79 张（真实游戏 sprite，绿幕背景）
- 检测器 `tools/sprite_monster_detector.py`：
  - `FastSpriteDetectorV8`：粗查（0.5x 图 + 均值填充模板，**无 mask**）记录"哪个模板/尺度命中"→ 精查仅验证命中模板（原图 mask 匹配）→ 类色验证（HSV 占比）
  - `SpriteMonsterDetector`：包装类，`detect(frame, player)` 接口，按玩家位置生成 ROI
  - `TemplateCollector`：战斗锁定目标时自动抠图入库（主色连通域抠图 + 绿幕化 + 去重阈值 0.93）
- 主程序接线：`--monster-backend sprite` + `--collect-templates`（`tools/auto_combat.py`）

---

## 2. 关键实验结论（接手必读，避免重蹈覆辙）⚠️

| # | 实验 | 结论 |
|---|---|---|
| 1 | 游戏内同类 sprite 互配 | **中位数 1.000，94% 配对 >0.8** → 冒险岛怪是像素级固定贴图，用户直觉正确 |
| 2 | 跨类 sprite 互配 | 仅 0.11–0.65 → 不同怪模板匹配可完全区分 |
| 3 | 白底官网立绘匹配实景 | 仅 0.45–0.58 全误报 → **官网立绘 ≠ 游戏内 sprite，禁止当模板** |
| 4 | YOLO v5 "0.995" 分数 | 验证集是白底图，假分数；实景基本全错 → YOLO 数据构成错误 |
| 5 | 旧标注质量 | 标错（blue_snail 框内实为橙色怪）、漏标普遍 → **评估基准不可信** |
| 6 | OpenCV mask 匹配 | 5.0 连续帧随机崩溃；4.11 下大量调用仍偶发 → 粗查用均值填充绕开 mask |

---

## 3. 当前缺陷（按优先级）❌

1. **实景识别准确率不足**：用户实测 sprite 后端"检测到的怪物全是错的"，检测框常套在背景上（最高优先级）
2. **新图（射手训练场III）4 怪模板缺失**：花蘑菇/绿水灵/绿蘑菇无实景模板；红蜗牛模板来自旧图
3. **检测速度慢**：全图 1.7s/帧、玩家 ROI 1.1s/帧（异步线程跑，依赖 3s sticky 窗口兜底）
4. **旧标注质量差**：导致评估失真 + 模板库继承错误
5. **地图 OCR 错误**："射手训练场III" → "射手训练场川"，无路线 → 只能巡逻模式
6. **模板采集依赖"锁定目标"信号**：锁定本身依赖检测 → 采集内容可能学错

---

## 4. 下一步：录制打怪流程（用户已确认，待实现）🎯

**核心思路**：用户手动打怪时，**每次按下攻击键（d）的瞬间，角色正前方攻击范围内的目标就是怪物**——这是最可靠的标注信号，比自动检测强得多。

### 4.1 功能需求
- 监听攻击键（pynput，参考 `PauseController` 写法）
- 攻击瞬间截图 → 定位玩家 + 朝向 → 正前方搜索区域找怪 → 抠图入库
- 朝向判定：监听左右方向键 + 攻击时目标相对玩家方位双信号互证
- 同时记录行走路线（玩家坐标序列 + 跳跃/爬绳标记）
- 复用：`GameWindowCapturor`（截图）、`PlayerLocator`（定位玩家）、`TemplateCollector._build_green_screen`（抠图）

### 4.2 落地建议
- 新文件 `tools/record_combat.py`（录制模式），产出：
  - 模板 → `monster_templates_final/<label>/`（供 sprite 后端加载）
  - 路线 → JSON（供寻路）
- 录制 5-10 分钟可覆盖训练场III 全部怪种，解决缺陷 #1/#2/#6

### 4.3 已就绪的可复用资产
- `TemplateCollector._build_green_screen(patch)`：主色连通域抠图 → 绿幕
- `TemplateCollector.collect(frame, detection)`：去重入库
- `sprite_monster_detector.CLASS_HSV`：各类颜色签名
- 玩家朝向逻辑在 `CombatPolicy.facing_direction`

---

## 5. 启动命令（当前状态）

```powershell
# 实景挂机（sprite 后端 + 自动学习）
& 'C:\quant_lab\lstm_gpu_venv\Scripts\python.exe' tools\auto_combat.py `
  --cfg shanda_legacy --player-name 麻超圆 --monster-backend sprite `
  --collect-templates --show-viz

# 冒烟测试（已验证通过）
& 'C:\quant_lab\lstm_gpu_venv\Scripts\python.exe' tools\auto_combat.py `
  --cfg shanda_legacy --player-name 麻超圆 --monster-backend sprite `
  --dry-run --max-frames 5 --headless
```

快捷键：**F8** 暂停/继续，**F9** 退出。

---

## 6. 关键文件索引

| 路径 | 说明 |
|---|---|
| `tools/auto_combat.py` | 主程序（152KB）：玩家定位、战斗策略、检测接线、--monster-backend/--collect-templates |
| `tools/sprite_monster_detector.py` | sprite 检测器（v8）+ SpriteMonsterDetector + TemplateCollector |
| `tools/live_perception_viewer.py` | 感知查看器：AdvisoryEvaluator（目标锁定）、YOLO 检测、MotionDetector |
| `monster_templates_final/` | 真实 sprite 模板库（red_snail/blue_snail/stump） |
| `training_data/maple_six_class_v5/` | YOLO 6 类训练数据（**白底图为主，实景表现差**） |
| `training_runs/maple_six_class_v5/weights/best.pt` | YOLO 模型（实景基本不可用） |
| `probe_output/_test_*.py` | 本次全部实验脚本（贴图可行性/多尺度/OpenCV 崩溃定位等） |
| `config/config_shanda_legacy.yaml` | 窗口/UI 区域/按键/阈值配置 |
| `nametag/麻超圆_player.png`、`nametag/麻超圆_勋章.png` | 玩家离线模板（名字框 + 勋章） |

---

## 7. 已知坑（接手别踩）

1. **Windows 中文路径**：`cv2.imwrite` 对含中文路径静默失败 → 用 `cv2.imencode` + `Path.write_bytes`
2. **OpenCV 版本**：已降到 4.11.0；带 mask 的 `matchTemplate` 大量连续调用仍可能 C 层崩溃 → 粗查用均值填充模板（无 mask），精查保持 mask（次数少）
3. **`cv2.imdecode` 参数**：正确写法 `cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)`
4. **颜色验证单位**：`inRange` 返回 0/255，颜色占比要 `(mask>0).sum()/N`，不能 `mask.sum()/N`
5. **旧数据标注不可信**：不要用 `maple_six_class_v5` 的 val 分数做评估；评估以人工看检测标注图为准
6. **player 参数**：主循环调用 `player_detector.detect(frame, now)` 必须 3 参 → 必须带 `--player-name`
7. **沙箱**：脚本里别用 `shutil.rmtree` 清理训练目录（Windows 沙箱拦截）→ 换新目录名或逐个 unlink

---

## 8. 交接验收清单

- [ ] 能跑冒烟测试（dry-run 5 帧无崩溃）
- [ ] 能启动 sprite 后端挂机 + 自动学习入库
- [ ] 玩家定位稳定（框不瞬移、对准角色）
- [ ] 录制模式 `tools/record_combat.py` 已实现（**待办**）
- [ ] 训练场III 4 怪模板已采集（**待办**）
- [ ] 地图 OCR 修正"射手训练场川"（**待办**）
