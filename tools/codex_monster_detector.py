"""怪物检测插件 —— 封装 codex 检测链路 + 空间标记 + 字段转换。

这是 auto_combat 的怪物检测"眼睛"，解耦成独立模块，方便单独调试、替换，
或交给 codex 排查问题。

检测链路（全部原封不动用 codex 的类，见 tools/yolo_monster_viewer.py）：
    YoloMonsterDetector（七类 YOLO 推理）
    -> DetectionTracker（检测框平滑 + 漏检桥接 + 速度预测）
    -> EntityCoordinateTracker（中心坐标 / 速度 / 运动状态）
    -> attach_player_relative_coordinates（相对玩家的距离）

本插件在此基础上额外做两件事：
    1. 同层标记：计算怪物是否与玩家处于同一水平带。默认保留全图检测，
       避免玩家定位漂移或怪物跳跃时把正确框直接删除。
    2. 字段转换：把 codex 的字段（class/confidence/box list）转成
       auto_combat 攻击逻辑需要的字段（label/score/box tuple 等）。
"""
import time

from tools.yolo_monster_viewer import (
    YoloMonsterDetector,
    DetectionTracker,
    EntityCoordinateTracker,
    attach_player_relative_coordinates,
    normalize_label,
)

# 当前地图默认只启用两类，其他地图仍可通过 labels 显式传入七类。
DEFAULT_LABELS = ["僵尸蘑菇", "刺蘑菇"]


class CodexMonsterDetector:
    """怪物检测插件：封装 codex 检测 + 同层标记 + 字段转换。"""

    def __init__(self, model_path, confidence, iou=0.45, device="0",
                 image_size=1280, labels=None, level_band=60.0,
                 same_level_only=False, min_confirmed_hits=2,
                 high_confidence_confirm=0.75, ui_y_start=687,
                 track_max_width_ratio=1.70,
                 track_max_height_ratio=2.00,
                 track_max_area_ratio=2.50,
                 track_size_cost_weight=0.25):
        self.labels = [normalize_label(x) for x in (labels or DEFAULT_LABELS)]
        # codex 原版 YOLO 检测核心
        self.core = YoloMonsterDetector(
            model_path, confidence, iou, device, image_size, labels=self.labels)
        # codex 原版跟踪 + 坐标
        self.tracker = DetectionTracker(
            max_missed=4,  # 用户要求改回4帧: 漏检靠降置信度解决, 不靠延长桥接
            smoothing=0.65,
            min_confirmed_hits=min_confirmed_hits,
            high_confidence_confirm=high_confidence_confirm,
            max_width_ratio=track_max_width_ratio,
            max_height_ratio=track_max_height_ratio,
            max_area_ratio=track_max_area_ratio,
            size_cost_weight=track_size_cost_weight,
        )
        self.coord = EntityCoordinateTracker()
        self.level_band = float(level_band)
        self.same_level_only = bool(same_level_only)
        self.ui_y_start = int(ui_y_start)

    def detect(self, frame, player=None):
        """返回 [{label, score, box, ...}] 供 auto_combat 攻击逻辑使用。"""
        dets = self.core.detect(frame, self.ui_y_start)
        dets = self.tracker.update(dets)
        dets = self.coord.update(
            dets, time.time(), frame.shape[1], self.ui_y_start, prefix="M")

        # 先标记同层关系；观察模式默认保留全图检测。
        player_y = None
        if player is not None and player.get("center"):
            player_y = player["center"][1]
        for detection in dets:
            center = detection.get("center_px")
            detection["same_level"] = (
                None
                if player_y is None or center is None
                else abs(center[1] - player_y) <= self.level_band
            )
        if self.same_level_only and player_y is not None:
            dets = [d for d in dets if d["same_level"]]

        # 相对玩家坐标（攻击距离判断用）
        if player is not None and player.get("center"):
            dets = attach_player_relative_coordinates(
                dets, {"center_px": player["center"]})

        # 字段转换：codex(class/confidence/box list) -> auto_combat(label/score/box tuple)
        out = []
        for d in dets:
            out.append({
                "label": d["label"],
                "score": d["confidence"],
                "box": tuple(d["box"]),
                "color": d["color"],
                "method": "codex",
                "entity_id": d.get("entity_id"),
                "center_px": d.get("center_px"),
                "velocity_px_s": d.get("velocity_px_s"),
                "speed_px_s": d.get("speed_px_s"),
                "motion_state": d.get("motion_state"),
                "tracking_state": d.get("tracking_state"),
                "confirmation_hits": d.get("confirmation_hits"),
                "confirmed": d.get("confirmed"),
                "same_level": d.get("same_level"),
                "relative_to_player": d.get("relative_to_player"),
                "has_hp_bar": False,
            })
        return out
