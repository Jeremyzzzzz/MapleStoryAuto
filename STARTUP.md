# 冒险岛自动打猪 - 启动说明

> **统一启动方式：双击桌面 `自动打猪.bat`**（v7 最新参数，无需手动输命令）。
> 不要凭记忆/上下文拼命令，bat 已固化全部参数。

## 一、启动命令（bat 内已固化，供参考）

```
"C:\quant_lab\lstm_gpu_venv\Scripts\python.exe" tools\auto_combat.py ^
  --cfg shanda_legacy ^
  --monster-backend yolo ^
  --yolo-model training_runs\wild_boar_real_hardneg_v4_960\weights\best.pt ^
  --yolo-confidence 0.10 ^
  --yolo-iou 0.70 ^
  --yolo-image-size 960 ^
  --no-color-verify ^
  --show-viz ^
  --player-name 麻超圆 ^
  --fps-limit 12 ^
  --no-ocr ^
  --monster-labels wild_boar ^
  --no-terrain ^
  --mode minimap_patrol ^
  --map-name "野猪的领土！！"
```

## 二、参数说明

| 参数 | 值 | 含义 |
|---|---|---|
| `--cfg` | shanda_legacy | 盛大怀旧服配置 |
| `--monster-backend` | yolo | 怪物检测用 YOLO |
| `--yolo-model` | training_runs/wild_boar_real_hardneg_v4_960/weights/best.pt | 野猪专用模型 |
| `--yolo-confidence` | 0.10 | YOLO 置信度 |
| `--yolo-iou` | 0.70 | NMS IoU |
| `--yolo-image-size` | 960 | 模型输入尺寸 |
| `--no-color-verify` | - | 关闭颜色验证 |
| `--show-viz` | - | 显示可视化窗口 |
| `--player-name` | 麻超圆 | 角色名 |
| `--fps-limit` | 12 | 帧率上限 |
| `--no-ocr` | - | 关闭 OCR（color_anchor 定位） |
| `--monster-labels` | wild_boar | 只打野猪 |
| `--no-terrain` | - | 不检测地形 |
| `--mode` | minimap_patrol | 小地图航点巡逻 |
| `--map-name` | 野猪的领土！！ | 地图名（加载录制路线） |

## 三、重要参数（勿随意改）

- **`--player-name 麻超圆`**：角色名，改了认不出人。
- **`--no-ocr`**：长期固定，贴图/色块匹配比 OCR 稳定。
- **`--map-name "野猪的领土！！"`**：必须精确匹配路线文件名，否则不加载路线只巡游。

## 四、启动后步骤

1. 双击 `自动打猪.bat`，等 1~2 分钟（YOLO 加载慢是正常的，日志停在"加载路线"处是正常的）；
2. 出现 `[status]` 日志 = 启动完成；
3. 点游戏窗口聚焦（日志显示 `游戏聚焦=Y`）；
4. F8 暂停/恢复、F4 开始巡航。

## 五、其他玩家检测（小地图红点）

- 其他玩家在小地图显示为**红色点**，检测到即挂机（保持喝药、不攻击/不移动）；
- 红点消失 **5 分钟**后自动恢复，或按 **F8** 立即恢复；
- 可视化窗口会显示 `其他玩家红点 N个: R1(x,y) R2(x,y)` 和 `■ 挂机中`。

## 六、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 日志一直停在"加载路线" | YOLO 模型加载慢(60-90s) | 正常，等 |
| 黄框飘 | 红分低回退模板匹配 | 检查红分(>0.05 正常)，重新聚焦 |
| 不加载路线 | map-name 不匹配 | 精确写地图名 |
| "已有实例在运行" | 上次进程没退出 | 任务管理器结束 python 或重启电脑 |
