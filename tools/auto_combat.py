import argparse
import datetime
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
for module_path in (
    REPO_ROOT,
    LOCAL_RUNTIME,
    LOCAL_RUNTIME / "win32",
    LOCAL_RUNTIME / "win32" / "lib",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
WIN32_DLL_HANDLE = None
if os.name == "nt" and (LOCAL_RUNTIME / "pywin32_system32").exists():
    WIN32_DLL_HANDLE = os.add_dll_directory(str(LOCAL_RUNTIME / "pywin32_system32"))

import cv2
import numpy as np

from src.engine.HealthMonitor import HealthMonitor
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardController import key_down, key_up, press_key
from src.utils.common import (
    activate_game_window,
    find_pattern_sqdiff,
    get_player_location_on_minimap,
    imread_cn,
    load_yaml,
    override_cfg,
)
from src.utils.text_cn import put_text_cn
from src.utils.logger import logger
from tools.live_perception_viewer import (
    AdvisoryEvaluator,
    MotionDetector,
    MonsterDetector,
    PlayerDetector,
    YoloMonsterDetector,
    read_vitals,
)
from tools.yolo_monster_viewer import (
    locate_minimap_player,
    locate_minimap_players,
    MinimapRedMarkerTracker,
)
from tools.sprite_monster_detector import (
    SpriteMonsterDetector,
    TemplateCollector,
)

WINDOW_TITLE = "MapleStory Auto Combat"


class HpMpOcrReader:
    """后台 OCR 识别 HP/MP/EXP 具体数值(格式 "HP[当前/总]" / "MP[当前/总]" /
    "EXP <当前经验值>[<百分比>%]")。

    用于反击检测: 怪物一次只扣一滴血, 百分比几乎不变, 具体数(1836->1835)可稳定识别。
    EXP 用于经验统计: 记录当前经验值数字(如 324625, 不含方括号内的百分比)。
    独立 RapidOCR 后台线程, 不阻塞主循环。
    """
    REFERENCE_WIDTH = 1278  # config bar_regions 的参考宽度

    def __init__(self, hp_region, mp_region, exp_region=None, submit_interval=0.25, ocr_threads=1):
        self.hp_region = tuple(hp_region)  # [x, y, w, h] 参考1278宽坐标
        self.mp_region = tuple(mp_region)
        self.exp_region = tuple(exp_region) if exp_region else None
        self.submit_interval = float(submit_interval)
        self.ocr_threads = int(ocr_threads)
        self.condition = threading.Condition()
        self.pending_frame = None
        self.pending_id = 0
        self.consumed_id = 0
        self.latest_result = None
        self.last_submit = 0.0
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        now = time.time()
        with self.condition:
            if self.stopped or now - self.last_submit < self.submit_interval:
                return
            self.pending_frame = frame.copy()
            self.pending_id += 1
            self.last_submit = now
            self.condition.notify()

    def latest(self, max_age=2.0):
        with self.condition:
            r = self.latest_result
        if r is None or time.time() - r["timestamp"] > max_age:
            return None
        return r

    @staticmethod
    def _parse(text):
        import re
        m = re.search(r'(\d+)\s*/\s*(\d+)', text or '')
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    @staticmethod
    def _parse_exp(text):
        """从 EXP 行取当前经验值数字。
        格式如 "EXP 324625[90.95%]" 或 "324625[90.95%]":
        - 先剥掉方括号内的百分比(含小数点, 避免粘连);
        - 去掉字母后取数字段, 选最长数字段(经验值通常 5-7 位);
        - 太短(<4 位)视为噪声丢弃。
        """
        import re
        text = (text or "").replace(" ", "")
        text = re.sub(r'\[.*?\]', '', text)            # 剥掉 [百分比]
        nums = re.findall(r'\d+', text)                 # 只留数字段
        if not nums:
            return None
        best = max(nums, key=lambda s: (len(s), int(s)))
        if len(best) < 4:                               # 太短 = 噪声
            return None
        return int(best)

    def _run(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR(
                intra_op_num_threads=self.ocr_threads,
                inter_op_num_threads=1,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stopped or self.pending_id > self.consumed_id)
                if self.stopped:
                    return
                frame = self.pending_frame.copy()
                self.consumed_id = self.pending_id
            try:
                scale = frame.shape[1] / float(self.REFERENCE_WIDTH)
                result = {}
                regions = [("hp", self.hp_region), ("mp", self.mp_region)]
                if self.exp_region is not None:
                    regions.append(("exp", self.exp_region))
                for key, region in regions:
                    x, y, w, h = region
                    x0 = int(x * scale)
                    y0 = int(y * scale)
                    x1 = int((x + w) * scale)
                    y1 = int((y + h) * scale)
                    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
                        continue
                    crop = frame[y0:y1, x0:x1]
                    res, _ = ocr(crop)
                    for entry in (res or []):
                        if len(entry) < 2:
                            continue
                        text = str(entry[1])
                        if key == "exp":
                            # EXP 行格式: "EXP 324625[90.95%]" -> 取 EXP 后面的
                            # 纯数字(当前经验值), 丢弃方括号里的百分比。
                            parsed = self._parse_exp(text)
                        else:
                            parsed = self._parse(text)
                        if parsed is not None and key not in result:
                            result[key] = parsed
                if result:
                    result["timestamp"] = time.time()
                    with self.condition:
                        self.latest_result = result
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Combat policy: pure decision logic (no input devices used), testable.
# --------------------------------------------------------------------------
class CombatPolicy:
    """Map advisory status + vitals to discrete combat commands.

    Commands are strings consumed by CombatExecutor:
      "attack_left" / "attack_right" : directional attack (turns first)
      "move_left"   / "move_right"   : walk toward the target
      "dodge_left"  / "dodge_right"  : step away from a danger
      "jump"                          : vertical disengage
      "add_hp" / "add_mp"             : use potion
      "none"                          : no input
    """

    def __init__(self, cfg, player_name=None, rng=None, mode="normal"):
        self.mode = mode  # "normal" | "stationary"(站桩打怪, 不移动)
        advisory_cfg = cfg["combat_advisory"]
        # Per-character attack range: profiles keyed by character name override
        # the advisory defaults (which apply when no profile matches).
        profiles = cfg.get("attack_profiles", {})
        profile = profiles.get(player_name) or profiles.get("default") or {}
        self.attack_horizontal_px = float(
            profile.get("horizontal", advisory_cfg["attack_horizontal_px"])
        )
        self.attack_vertical_px = float(
            profile.get("vertical", advisory_cfg["attack_vertical_px"])
        )
        self.dodge_horizontal_px = float(advisory_cfg["dodge_horizontal_px"])
        self.dodge_vertical_px = float(advisory_cfg["dodge_vertical_px"])
        self.immediate_danger_px = float(advisory_cfg["immediate_danger_px"])
        # Mage / ranged classes should keep a minimum engagement distance so
        # they never end up meleeing a monster.
        self.min_engage_px = float(profile.get("keep_distance", 80))
        # Melee classes (warrior: keep_distance <= 30) fight face-to-face and
        # must NOT dodge/back off when the monster is close -- the original
        # dodge logic is tuned for ranged characters and made the warrior
        # permanently hop away from monsters it is supposed to hit.
        self.is_melee = self.min_engage_px <= 30

        auto_cfg = cfg.get("auto_combat", {})
        # Randomized attack interval: never a fixed cadence (anti-detection).
        self.attack_cooldown_min = float(
            auto_cfg.get("attack_cooldown_min", auto_cfg.get("attack_cooldown", 0.3))
        )
        self.attack_cooldown_max = float(
            auto_cfg.get("attack_cooldown_max", auto_cfg.get("attack_cooldown", 0.6))
        )
        if self.attack_cooldown_min > self.attack_cooldown_max:
            self.attack_cooldown_min, self.attack_cooldown_max = (
                self.attack_cooldown_max,
                self.attack_cooldown_min,
            )
        self.attack_hold_seconds = float(auto_cfg.get("attack_hold_seconds", 0.06))
        self.dodge_hold_seconds = float(auto_cfg.get("dodge_hold_seconds", 0.15))
        self.step_hold_seconds = float(auto_cfg.get("step_hold_seconds", 0.06))
        self.turn_hold_seconds = float(auto_cfg.get("turn_hold_seconds", 0.08))
        self.turn_pause_seconds = float(auto_cfg.get("turn_pause_seconds", 0.05))
        self.move_hold_seconds = float(auto_cfg.get("move_hold_seconds", 0.12))
        # 【玩家置信度门槛】(跳跃防掉格): 玩家检测框飘到偏位时置信度低,
        # 低于此值不触发主动攻击(受伤反击不受限)。
        self.attack_min_player_confidence = float(
            auto_cfg.get("attack_min_player_confidence", 0.9))
        # 【巡游打怪范围】(绿色框): 以玩家为中心, 横向 ±此值内检测到怪物,
        # 角色离开航点路线去追怪消灭; 范围内无怪才继续走录制的点位。
        self.patrol_hunt_range_px = float(
            auto_cfg.get("patrol_hunt_range_px", 300))
        # Patrol: when no target, wander so the character keeps hunting.
        self.patrol_enabled = bool(auto_cfg.get("patrol_enabled", True))
        self.patrol_min_seconds = float(auto_cfg.get("patrol_min_seconds", 1.5))
        self.patrol_max_seconds = float(auto_cfg.get("patrol_max_seconds", 4.0))
        # Anti-stuck: if a target is attacked several times but never dies it is
        # probably a drop item or detection artifact; ignore it for a while.
        # The counter resets if the target keeps moving, so healthy monsters
        # that take a few hits are never misclassified as drops.
        self.attack_retry_limit = int(auto_cfg.get("attack_retry_limit", 4))
        self.attack_ignore_seconds = float(auto_cfg.get("attack_ignore_seconds", 2.0))
        # Jump disengage cooldown: don't spam jumps at a close target.
        self.jump_cooldown = float(auto_cfg.get("jump_cooldown", 1.5))
        # 跳跃攻击: 跳跃间隔小范围随机(跳跃后攻击是固定动作, 不受攻击冷却影响)
        self.jump_cooldown_min = float(auto_cfg.get("jump_cooldown_min", 1.2))
        self.jump_cooldown_max = float(auto_cfg.get("jump_cooldown_max", 1.8))
        self.rng = rng if rng is not None else random.Random()
        self.facing_direction = None  # "left" | "right"
        self.next_attack_time = float("-inf")
        # 被攻击反击: HP 从正常突变成 0%/None(血条被攻击闪烁导致检测不到)时反击一次
        self._last_hp = None           # 上一次正常的 HP(>5%)
        self._last_hp_cur = None       # 上一次 HP 具体数(OCR, 扣一滴血检测)
        self._hp_readings = []         # 最近 HP 读数(中位数滤波, 减少OCR个位数噪声)
        self._last_counter_at = float("-inf")  # 上次反击时间(防抖)
        self._last_front_seen = float("-inf")  # 上次前方有怪的时间(转身确认延迟)
        self._last_move_dir = None  # 最近一次移动方向(用于"被击退"检测)
        self._last_vx = None        # 上一次速度x分量(方向突变检测)
        # 攻击方向优化: 记录最近攻击/反击时怪物在玩家的哪一侧。
        # 被撞退/异步YOLO延迟导致 target_box 过期时, 用缓存方向兜底,
        # 保证"怪物在右侧撞到主角 → 主角向右攻击"。
        self._last_target_dir = None   # 最近一次目标的相对方向 "left"/"right"
        self._last_target_time = float("-inf")  # 上次记录时间(过期失效)
        # 巡游打怪方向防抖: 选定追怪方向后短时间锁定, 防怪框左右抖动导致
        # 角色来回走(每帧重算最近怪, 怪在两侧交替出现时方向会翻转)。
        self._hunt_dir_lock = None        # "left"/"right" 或 None
        self._hunt_dir_locked_until = 0.0 # 方向锁定截止时间
        self._hunt_dir_lock_seconds = float(
            auto_cfg.get("hunt_dir_lock_seconds", 0.8))  # 锁定时长(秒)
        # 【巡游打怪目标跟踪锁定】: 锁定绿色范围内最近的怪, 漏检(树妖漏检高/
        # 单帧抖动)时用最后位置继续追击 hunt_hold_seconds 秒, 防"走过去→漏检
        # →回点位→又检测到→再走过去"来回走。
        self._hunt_locked = None          # {"center": (x,y), "last_seen": ts}
        self._hunt_hold_seconds = float(
            auto_cfg.get("hunt_hold_seconds", 3.0))  # 漏检后保持追击时长(秒)
        # 【新巡游策略-怪出现确认帧】: 未追怪时, 怪需连续确认 hunt_confirm_frames
        # 次出现在绿框才追(单帧误检/漏检不打断巡航), 防"一帧有怪就追/一帧没有
        # 就回"来回走。追怪中已有 lock, 不受此限制。
        self.hunt_confirm_frames = int(auto_cfg.get("hunt_confirm_frames", 3))
        self._hunt_confirm_count = 0      # 连续确认计数
        self._session_started = time.time()  # 挂机开始时间(防封号休息计时)
        self.patrol_direction = None  # "left" | "right"
        # 小地图稳定坐标(每帧由主循环更新 self._mini): 用于巡逻边缘判断, 不误检蘑菇
        self._mini = None
        mm_patrol = cfg.get("patrol_minimap", {})
        self._mini_right_norm = float(mm_patrol.get("right_norm", 0.82))  # map_norm_x>此值(玩家接近地图右端)往左走
        self._mini_left_norm = float(mm_patrol.get("left_norm", 0.18))    # map_norm_x<此值(玩家接近地图左端)往右走
        self._waypoint_patrol = None  # MinimapWaypointPatrol 实例(主循环注入), 用于小地图坐标航点巡逻
        # 【安全点定时进商城】(测谎仪规避): 主循环状态机控制; 激活时
        # decide 不再打怪/追怪, 只沿 _safe_patrol 的 one-shot 序列走向安全点。
        self._safe_active = False
        self._safe_patrol = None
        # 【恢复路线】(安全点退出商城后/跌落底层时走回巡游线): 同上, 独立
        # one-shot 导航, 走完恢复路线后从当前位置继续主航线。
        self._recall_active = False
        self._recall_patrol = None
        self.patrol_deadline = float("-inf")
        self.last_jump_time = float("-inf")
        # 【跳跃后禁攻击】: 记录最近一次跳跃键发出时间, 跳跃后 2 秒内不攻击
        # (防在跳台上因攻击移动掉下来)。由主循环 set_jump_at 更新。
        self._last_jump_at = float("-inf")
        self.jump_lock_seconds = 2.0   # 跳跃后禁攻击时长(秒)
        self.next_jump_time = float("-inf")  # 下次跳跃攻击时间(随机间隔)
        # Climbing state: when > 0, the policy is climbing a rope.
        self._climb_started_at = 0.0
        self.climb_timeout = float(auto_cfg.get("climb_timeout", 4.0))
        # Rope grab state: some ropes hang with their bottom ABOVE the
        # player's reach, so the character must jump once and then press up
        # to grab the rope. _climb_jump_at records when the jump was issued.
        self._climb_jump_at = 0.0
        # Stuck detection: if the player has not moved for this long, try a jump.
        self.stuck_threshold = float(auto_cfg.get("stuck_threshold", 2.5))
        self.last_stuck_at = float("-inf")
        self._last_player_y = None
        # 巡游卡住检测: 移动中 x 坐标几乎不变(碰到坑/障碍)持续1秒 -> 跳跃跳过
        self._last_patrol_x = None
        self._stuck_start = None
        self._frame_width = None  # 帧宽度(地图边缘判断), 由主循环每帧更新
        self._last_edge_turn = float("-inf")  # 上次边缘换方向时间(反击过滤用)
        self._edge_start = None           # 持续处于边缘的起始时间(持续1秒才转向)
        self._jump_stuck_count = 0        # 连续跳坑次数(跳不动就换方向)
        self._force_dir = None          # 边缘转向后的强制方向(强制走一段再恢复正常)
        self._force_until = float("-inf")
        # target center grid -> (attack_count, ignore_until, last_center)
        self.target_attack_stats = {}

    @staticmethod
    def _center(detection):
        x, y, width, height = detection["box"]
        return (x + width / 2.0, y + height / 2.0)

    def _next_attack_delay(self):
        return self.rng.uniform(self.attack_cooldown_min, self.attack_cooldown_max)

    def _target_key(self, center, cell=40):
        return (int(center[0] // cell), int(center[1] // cell))

    def _is_target_ignored(self, center, now):
        count, ignore_until, _ = self.target_attack_stats.get(
            self._target_key(center), (0, 0.0, None)
        )
        return ignore_until > now

    def _record_attack_attempt(self, center, now):
        key = self._target_key(center)
        count, ignore_until, last_center = self.target_attack_stats.get(
            key, (0, 0.0, None)
        )
        if last_center is not None:
            moved = (
                abs(center[0] - last_center[0]) + abs(center[1] - last_center[1])
                >= 12.0
            )
            if moved:
                # A moving target is a real monster: reset the counter so it is
                # never mistaken for a stationary drop.
                count = 0
        count += 1
        if count >= self.attack_retry_limit:
            # Target stayed in place and survived too many attacks: treat it as
            # a drop/artifact.
            self.target_attack_stats[key] = (
                0,
                now + self.attack_ignore_seconds,
                center,
            )
        else:
            self.target_attack_stats[key] = (count, ignore_until, center)

    def _patrol_decision(self, now, player=None):
        """Wander in a random direction for a random duration when no target.

        巡游卡住检测: 角色朝一个方向走, 但 x 坐标几乎不变(碰到坑/障碍)持续约 1 秒,
        就朝该方向跳跃跳过障碍; 走到地图边缘(不是卡住)则换方向走。
        """
        direction = self.patrol_direction
        # 小地图稳定坐标(优先级最高, 不误检蘑菇): map_norm_x 反映玩家在地图里的左右绝对位置
        mini_nx = None
        if self._mini is not None:
            _mn = self._mini.get("map_norm")
            if _mn:
                mini_nx = _mn[0]
        if player is not None and direction is not None:
            px, py = self._center(player)
            fw = self._frame_width or 1370
            # --- 边缘转向: 方向性判断 + 持续1秒确认 ---
            # 往左走碰到左边缘才转向右, 往右走碰到右边缘才转向左。
            # 转向后方向变了, 即使坐标还在边缘区也不会重复触发(方向性)。
            left_edge_px = 340   # 左边缘: x<340 转向右(集中中间刷怪)
            right_edge_px = 815  # 右边缘: x>815 转向左(识别框候选已排除x>830, 用815留15px缓冲确保转向时识别框还在)
            # 小地图驱动(稳定): 玩家在地图右端(map_norm_x>阈值)往左, 左端(<阈值)往右
            mini_edge = False
            if mini_nx is not None:
                if direction == "right" and mini_nx > self._mini_right_norm:
                    mini_edge = True
                elif direction == "left" and mini_nx < self._mini_left_norm:
                    mini_edge = True
            hit_edge = (
                (direction == "left" and px < left_edge_px)
                or (direction == "right" and px > right_edge_px)
                or mini_edge
            )
            if hit_edge:
                if self._edge_start is None:
                    self._edge_start = now
                elif (now - self._edge_start >= 1.0
                        and now - self._last_edge_turn >= 2.0):
                    # 持续1秒在边缘 -> 换方向
                    new_dir = "right" if direction == "left" else "left"
                    self.patrol_direction = new_dir
                    self.facing_direction = new_dir
                    self._edge_start = None
                    self._last_edge_turn = now  # 记录(反击过滤+转向冷却)
                    self._last_patrol_x = None
                    self._stuck_start = None
                    self._jump_stuck_count = 0
                    return f"move_{new_dir}", "patrol_edge_turn"
            else:
                self._edge_start = None
            # (已停用) 老版 x 卡住跳坑检测: 平地上被怪挡/攻击停顿 x 不变 0.35 秒会误跳。
            # 掉坑跳跃已改由 y 坐标检测(y>475)负责, 这里不再发跳跃命令。
        # 方向保持: 只有碰到边缘才换方向; 玩家框丢失也保持方向继续按键盘走
        if self.patrol_direction is None:
            self.patrol_direction = self.rng.choice(("left", "right"))
        self.facing_direction = self.patrol_direction  # 移动方向即面朝方向
        return f"move_{self.patrol_direction}", "patrol"

    def _maybe_jump_out_of_pit(self, player, now, pit_y=475, stuck_px=5,
                               stuck_seconds=1.0, max_jumps=3):
        """掉坑后卡墙才跳; 连续跳几次仍在坑里则换向, 避免同一面墙空跳。

        Returns (command, reason) or None to continue the caller's logic.
        Always updates `_last_patrol_x` so the next frame can detect a stall.
        """
        if player is None:
            return None
        if self.patrol_direction is None:
            self.patrol_direction = "right"
        player_center = self._center(player)
        in_pit = player_center[1] > pit_y
        if in_pit and now - self.last_jump_time >= 1.0:
            stalled = (
                self._last_patrol_x is not None
                and abs(player_center[0] - self._last_patrol_x) < stuck_px
            )
            if stalled:
                if self._stuck_start is None:
                    self._stuck_start = now
                elif now - self._stuck_start >= stuck_seconds:
                    self._stuck_start = None
                    self._last_patrol_x = None
                    self.last_jump_time = now
                    self._jump_stuck_count += 1
                    if self._jump_stuck_count >= max_jumps:
                        new_dir = "left" if self.patrol_direction == "right" else "right"
                        self.patrol_direction = new_dir
                        self._jump_stuck_count = 0
                        self.facing_direction = new_dir
                        return f"jump_{new_dir}", "patrol_stuck_turn"
                    self.facing_direction = self.patrol_direction
                    return f"jump_{self.patrol_direction}", "jump_out_of_pit"
            else:
                self._stuck_start = None
        else:
            self._stuck_start = None
            if not in_pit:
                self._jump_stuck_count = 0
        self._last_patrol_x = player_center[0]
        return None

    def set_move_dir(self, command):
        """主循环调用: 记录本帧移动命令方向, 供下一帧 decide 判断被击退。"""
        if command in ("move_left", "attack_left", "jump_left", "dodge_left"):
            self._last_move_dir = "left"
        elif command in ("move_right", "attack_right", "jump_right", "dodge_right"):
            self._last_move_dir = "right"
        elif command in ("jump",):
            pass  # 原地跳, 保持原移动意图
        else:
            self._last_move_dir = None

    def set_jump_at(self, now):
        """主循环调用: 记录跳跃键发出时刻, 跳跃后 jump_lock_seconds 秒内禁主动攻击。"""
        self._last_jump_at = now

    def terrain_decision(self, player, terrain):
        """If a rope/platform is nearby and no monster is in the way, act on it.

        Returns (command, reason) or None if terrain is not actionable.
        The caller combines this with the combat decision.
        """
        if player is None or terrain is None:
            return None
        px, py = self._center(player)
        # Use player width/height from the box to estimate tile size.
        box = player.get("box") or (0, 0, 60, 80)
        tile_w = max(40, box[2])
        tile_h = max(40, box[3])

        # Already climbing: keep climbing upward until the player reaches the
        # top of the rope (no more rope pixels above) or we time out.
        if self._climb_started_at > 0:
            elapsed = time.time() - self._climb_started_at
            # If no rope is visible above the player anymore, we made it.
            ropes_above = [
                r for r in terrain["ropes"]
                if abs(r[3] - px) <= tile_w // 2 and r[0] < py
            ]
            if not ropes_above or elapsed > self.climb_timeout:
                self._climb_started_at = 0.0
                return "none", "climb_done"
            return "climb_up", "climb_in_progress"

        # Try to find a rope directly above (or nearly above) the player.
        ropes_above = [
            r for r in terrain["ropes"]
            if abs(r[3] - px) <= tile_w // 2 and r[0] < py - 10
        ]
        if ropes_above:
            if not self.patrol_enabled:
                return None
            # Some ropes hang with their bottom above the player's reach: the
            # character must JUMP first and then press up to grab the rope.
            # r = (x, y_top, y_bot, x_center); r[2] is the rope bottom.
            rope = min(ropes_above, key=lambda r: r[0])
            y_bot = rope[2]
            if y_bot < py - 25:
                # Rope bottom is out of reach -> jump, then climb for ~0.3s
                # (during the jump arc). After that, if still not grabbed,
                # jump again.
                now_t = time.time()
                if self._climb_jump_at > 0 and now_t - self._climb_jump_at < 0.3:
                    return "climb_up", "climb_after_jump"
                self._climb_jump_at = now_t
                return "jump", "jump_to_rope"
            # Rope bottom is within reach: climb directly.
            self._climb_jump_at = 0.0
            self._climb_started_at = time.time()
            return "climb_up", "reach_rope"

        # Otherwise: walk toward the nearest rope's center x.
        if not self.patrol_enabled:
            return None
        # Pick a rope whose bottom anchor is in the player's vertical
        # neighbourhood (roughly within a tile or so). Ropes dangling far
        # above / below are handled by the climb-vs-patrol logic in the caller.
        candidate = None
        candidate_dx = 10**9
        for r in terrain["ropes"]:
            if abs(r[2] - py) > 100:
                continue
            dx = abs(r[3] - px)
            if dx < candidate_dx:
                candidate = r
                candidate_dx = dx
        if candidate is None:
            return None
        # Don't chase a rope that's barely closer than the tile width: too far
        # to bother, prefer normal patrol.
        if candidate_dx > tile_w * 1.5:
            return None
        if candidate[3] < px - 4:
            return "move_left", "approach_rope"
        if candidate[3] > px + 4:
            return "move_right", "approach_rope"
        return None

    def stop_climbing(self):
        """Cancel an in-progress climb (used by the main loop when monsters appear)."""
        self._climb_started_at = 0.0

    def decide(self, player, advisory, hp_percent, mp_percent, now, hp_cur=None, hp_max=None, mp_cur=None, mp_max=None, monsters=None):
        """Return (command, reason)."""
        # 【定时休息已移除】(2026-08-27): 用户反馈30分钟休息+5分钟停打仍不能
        # 躲过测谎仪, 暂时去掉。红点挂机(检测到其他玩家只喝药)属独立机制, 保留。

        # 【跳跃后禁攻击(2秒)】: 人物启动跳跃键后 2 秒内不攻击——防在跳台上
        # 因攻击移动掉下台。反击(被怪打/击退)不受限(必须还手); 喝药/走位不受限。
        # 由主循环 set_jump_at 记录跳跃时刻; 冷却中这里直接标记跳过主动攻击。
        _jump_locked = (now - self._last_jump_at) < self.jump_lock_seconds
        if _jump_locked:
            self._jump_attack_locked = True
        else:
            self._jump_attack_locked = False

        # 被攻击反击: 优先用 HP 具体数下降判断(血条减少就马上攻击)。
        # 2次读数取最小: 比3窗口中位数更快反映下降(延迟约1个OCR周期), 仍能滤单帧上跳噪声。
        # 血条被攻击闪烁时 OCR 会误读成个位数(如 HP=9/1836), 需过滤异常低读数。
        # HP 检测不到(None)时, 方向突变(被击退)作为兜底, 并过滤边缘转向和跳坑。
        hp_dropped = False
        if hp_cur is not None:
            # 过滤异常低读数: 比上次稳定值低500+点视为血条闪烁误读, 拒绝进入窗口
            if self._last_hp_cur is not None and hp_cur < self._last_hp_cur - 500:
                hp_cur = None
        if hp_cur is not None:
            self._hp_readings.append(hp_cur)
            if len(self._hp_readings) > 2:
                self._hp_readings.pop(0)
            if len(self._hp_readings) >= 2:
                stable_hp = min(self._hp_readings)  # 取最小(任一帧读到低值即算下降)
                if self._last_hp_cur is not None and stable_hp < self._last_hp_cur:
                    hp_dropped = True  # 血条减少了, 马上反击
                self._last_hp_cur = stable_hp
        knocked_back = False
        if player is not None and player.get("velocity_px_s"):
            vx = player["velocity_px_s"][0]
            if self._last_vx is not None:
                # 方向突变: 速度方向突然反向(从正变负 或 从负变正), 幅度大
                reversed_dir = (
                    (self._last_vx > 40 and vx < -40)
                    or (self._last_vx < -40 and vx > 40)
                )
                if reversed_dir:
                    # 过滤边缘转向和跳坑: 这两种情况也有方向突变, 但不是被击退
                    if (now - self._last_edge_turn >= 0.6
                            and now - self.last_jump_time >= 0.6):
                        knocked_back = True
            self._last_vx = vx

        # 【扣血反击已禁用】(2026-08-30): 用户要求先禁用"人物扣血反击"——
        # HP 掉血不再触发反击, 只保留方向突变(knocked_back)反击(且有玩家
        # 置信度门槛兜底)。hp_dropped 计算保留但不再作为触发条件。
        # 【安全点/恢复路线行程中禁反击】: 走路期间被怪撞也不反击(反击会
        # 转向/位移, 打断行程导致走不到安全点/恢复点)。
        if (knocked_back
                and not (self._safe_active or self._recall_active)
                and now - self._last_counter_at >= 0.15):
            # 反击(所有模式含纯点位巡航): 被怪打/击退立即反击(现仅方向突变触发)。
            # 方向优化: 优先用【当前可见怪物位置】(advisory target_box);
            # 若 target_box 过期/丢失(异步延迟), 用缓存的最近目标方向;
            # 都没有才保持当前朝向。保证"怪物在右侧撞到主角 → 主角向右反击"。
            # 【玩家置信度门槛(跳跃防掉格)】: 跳跃格子时玩家框会飘到偏位
            # (置信度约0.55), 若仍按飘偏位置计算方向反击, 会把角色打到
            # 错误方向掉下格子。因此【方向突变触发的反击】在玩家置信度
            # < attack_min_player_confidence 时禁止(此时框不可信)。
            _counter_conf = float(player.get("score", 0.0) or 0.0) if player else 0.0
            if _counter_conf < self.attack_min_player_confidence:
                # 跳跃/下落时玩家框飘偏(置信度低)的"假击退": 不反击,
                # 也不记录方向, 直接跳过本次反击(继续正常巡航/跳跃)。
                self._last_vx = vx  # 保持速度基准, 防下一帧再触发
                pass
            else:
                self._last_counter_at = now
                direction = None
                if advisory is not None and advisory.get("target_box") is not None:
                    _tb = advisory["target_box"]
                    _tc = self._center({"box": tuple(_tb)})
                    _pc = self._center(player)
                    if _pc is not None:
                        direction = "left" if _tc[0] < _pc[0] else "right"
                # 方向缓存兜底: target_box 过期(1.5s内)用最近记录的目标方向
                if direction is None and now - self._last_target_time < 1.5:
                    direction = self._last_target_dir
                direction = direction or self.facing_direction or "left"
                self.facing_direction = direction
                # 记录本次反击方向(供后续被击退兜底)
                if advisory is not None and advisory.get("target_box") is not None:
                    _tb = advisory["target_box"]
                    _tc = self._center({"box": tuple(_tb)})
                    _pc = self._center(player)
                    if _pc is not None:
                        self._last_target_dir = "left" if _tc[0] < _pc[0] else "right"
                        self._last_target_time = now
                self.next_attack_time = now  # 立即攻击(无视冷却)
                return f"attack_{direction}", "counter_attack"
        # 站桩模式: 人物不动, 无目标/玩家丢失时绝不巡逻/巡游(避免移动)
        if self.mode == "stationary":
            if not player:
                return "none", "player_missed"
            if advisory is None or advisory.get("target_box") is None:
                return "none", "no_target"
        # 巡游打怪模式: 保持巡游方向, 攻击黄圈内有怪才攻击,
        # 不在攻击范围就继续按巡游方向走(不追远处怪)。
        # 方向硬约束: patrol_direction 是唯一移动方向来源, facing_direction
        # 始终钳制为 patrol_direction, 攻击/反击/跳坑方向全部=它, 绝不反向。
        if self.mode == "patrol_hunt":
            if self.patrol_direction is not None:
                self.facing_direction = self.patrol_direction  # 方向钳制
            if not player:
                return self._patrol_decision(now, player)
            pit_cmd = self._maybe_jump_out_of_pit(player, now)
            if pit_cmd is not None:
                return pit_cmd
            if advisory is None or advisory.get("target_box") is None:
                return self._patrol_decision(now, player)
            _tb = advisory["target_box"]
            _tc = self._center({"box": tuple(_tb)})
            _pc = self._center(player)
            _hd = abs(_tc[0] - _pc[0])
            _vd = abs(_tc[1] - _pc[1])
            if (_hd <= self.attack_horizontal_px
                    and _vd <= self.attack_vertical_px):
                # 方向硬约束: 只打巡逻方向(前方)的怪, 攻击方向永远=巡逻大方向,
                # 绝不朝侧后方变向(那会导致左右互搏)。evaluate 已按 facing=patrol_direction
                # 只选前方怪, 这里再兜底钳制一次。
                _dir = self.patrol_direction or ("left" if _tc[0] < _pc[0] else "right")
                self.facing_direction = _dir  # = patrol_direction, 不朝怪临时变向
                # 跳跃后 2 秒禁攻击(防跳台掉落): 保持巡游移动不攻击
                if self._jump_attack_locked:
                    return self._patrol_decision(now, player)[0], "patrol_jump_lock"
                if now >= self.next_attack_time:
                    self.next_attack_time = now + self._next_attack_delay()
                    return f"attack_{_dir}", "attack"
                # 冷却中继续巡游, 不要 none 松键(否则边走边打会变成走-停-走)
                return self._patrol_decision(now, player)[0], "attack_cooldown_patrol"
            # 不在攻击范围 -> 继续巡游(保持方向)
            return self._patrol_decision(now, player)
        if self.mode == "minimap_patrol":
            # 【纯点位巡航 + 顺路打怪】: 攻击只有两个触发源——
            #   ① 怪物进入黄色攻击范围(advisory.attack_ready)且玩家空闲 -> 转向攻击;
            #   ② 被怪撞到/击退 -> 上方反击逻辑(hp_dropped/knocked_back)。
            # 绝不为了范围外的怪移动/转向; 跳跃爬绳中(_wp_busy)不攻击。
            _wp_busy = (self._waypoint_patrol is not None
                        and bool(self._waypoint_patrol._action_state))
            # 【安全点定时进商城】: 激活时不打怪不追怪, 只导航安全点 one-shot 序列
            if self._safe_active and self._safe_patrol is not None:
                _wpd = self._safe_patrol.decide(
                    self._mini["map_norm"] if self._mini else None, now)
                if _wpd is not None:
                    return _wpd
                return "none", "safe_wait"
            # 【恢复路线】: 安全点退出商城后/跌落地底时走回巡游路线(同上机制)
            if self._recall_active and self._recall_patrol is not None:
                _wpd = self._recall_patrol.decide(
                    self._mini["map_norm"] if self._mini else None, now)
                if _wpd is not None:
                    return _wpd
                return "none", "recall_wait"
            # 【巡游打怪(绿色范围)】: 以玩家为中心横向 ±patrol_hunt_range_px(默认300)
            # 范围内检测到怪物 -> 角色离开航点路线去追怪消灭(向怪走/攻击);
            # 范围内没有怪 -> 继续走录制的点位巡航。优先级高于巡航但低于跳跃爬绳
            # (_wp_busy 时不抢)。
            if player and not _wp_busy:
                _pc = self._center(player)
                # 【已取消置信度低不攻击限制】: 玩家识别已稳定(蓝条color_anchor),
                # 不再因置信度门槛漏攻击。跳跃后 2 秒禁攻击由 _last_jump_at 控制。
                _hunt_ok = True
                _hunt = None   # 追击目标 {"center", "hd", "hdv"} (当前帧怪或锁定)
                _hunt_from_frame = False   # True=当前帧真实检测的怪(可攻击)
                # ---- ① 当前帧范围内选最近的怪(更新锁定, 带确认帧防摇摆) ----
                # 【新巡游策略-防来回走】: 怪检测不稳定(树妖丢帧), 若"一帧看到怪
                # 就追/一帧丢就去"会在 追怪↔回点位 之间摇摆。给"怪出现"加确认:
                # 无锁定目标时, 怪需连续 count 次出现在绿框才追(单帧误检不打断);
                # 追怪中已有锁定, 保持 hold_seconds 漏检容忍。回点位只在锁定清空后。
                _frame_hunt = None
                if _hunt_ok and monsters:
                    for _m in monsters:
                        if not _m.get("box"):
                            continue
                        _mc = self._center(_m)
                        _hd = abs(_mc[0] - _pc[0])
                        _hdv = abs(_mc[1] - _pc[1])
                        if (_hd <= self.patrol_hunt_range_px
                                and _hdv <= self.attack_vertical_px):
                            if _frame_hunt is None or _hd < _frame_hunt["hd"]:
                                _frame_hunt = {"center": _mc, "hd": _hd, "hdv": _hdv}
                if _frame_hunt is not None:
                    # 命中(当前帧绿框内有怪)
                    if self._hunt_locked is None:
                        # 未在追怪: 连续确认帧累计(防单帧误检打断巡航)
                        self._hunt_confirm_count += 1
                        if self._hunt_confirm_count >= self.hunt_confirm_frames:
                            self._hunt_locked = {
                                "center": tuple(_frame_hunt["center"]),
                                "last_seen": now,
                            }
                            self._hunt_confirm_count = 0
                            _hunt = _frame_hunt
                            _hunt_from_frame = True
                        else:
                            self._hunt_locked = None  # 确认中: 继续巡航, 不追
                    else:
                        # 已在追怪: 更新锁定(最新位置+时间)
                        self._hunt_locked = {
                            "center": tuple(_frame_hunt["center"]),
                            "last_seen": now,
                        }
                        self._hunt_confirm_count = 0
                        _hunt = _frame_hunt
                        _hunt_from_frame = True
                elif (self._hunt_locked is not None
                        and now - self._hunt_locked["last_seen"] <= self._hunt_hold_seconds):
                    # ---- ② 追怪中漏检短时保持: 用最后位置继续追 ----
                    _lc = self._hunt_locked["center"]
                    _hd = abs(_lc[0] - _pc[0])
                    _hdv = abs(_lc[1] - _pc[1])
                    _hunt = {"center": _lc, "hd": _hd, "hdv": _hdv}
                    _hunt_from_frame = False   # 锁定保持: 只追不攻击(防对空气挥拳)
                else:
                    # ---- ③ 无怪且锁定过期: 清除锁定, 回巡航点位 ----
                    self._hunt_locked = None
                    self._hunt_confirm_count = 0
                if _hunt is not None:
                    _hdir = "left" if _hunt["center"][0] < _pc[0] else "right"
                    # 【方向防抖】: 怪框在玩家左右两侧抖动会导致 _hdir 每帧翻转
                    # -> 角色来回走。策略: 选定方向后锁定 _hunt_dir_lock_seconds
                    # (0.8s); 锁定期内即使怪跳到另一侧也不翻转方向(除非怪横穿
                    # 远离超过半个范围, 即真的跑到对面去了)。锁定期后才允许按
                    # 当前怪位置重新选方向。
                    _locked = self._hunt_dir_lock
                    if (_locked is not None and now < self._hunt_dir_locked_until
                            and _locked == _hdir):
                        pass  # 方向已锁定且怪仍在锁定侧: 保持
                    elif (_locked is not None and now < self._hunt_dir_locked_until
                          and _locked != _hdir
                          and _hunt["hd"] <= self.patrol_hunt_range_px * 0.5):
                        pass  # 怪短暂跳到另一侧(小抖动): 保持锁定方向
                    else:
                        # 锁定期已过 或 怪真正跑到对面: 更新方向并重新上锁
                        self._hunt_dir_lock = _hdir
                        self._hunt_dir_locked_until = now + self._hunt_dir_lock_seconds
                    _hdir = self._hunt_dir_lock or _hdir
                    self._last_target_dir = _hdir
                    self._last_target_time = now
                    # 追击判定:
                    # 【必须当前帧真实检测到怪(不是锁定保持)且在攻击范围内才攻击】
                    # ——怪已被打死/消失时, 锁定保持只是"按最后位置追", 若位置在
                    # 攻击范围内说明怪已死, 不能再对着空气挥拳, 直接清锁定回巡航。
                    if not _hunt_from_frame and (
                            _hunt["hd"] <= self.attack_horizontal_px
                            and _hunt["hdv"] <= self.attack_vertical_px):
                        # 怪已死/不可及: 清除锁定, 回点位
                        self._hunt_locked = None
                    elif (_hunt_from_frame
                            and _hunt["hd"] <= self.attack_horizontal_px
                            and _hunt["hdv"] <= self.attack_vertical_px):
                        # 跳跃后 2 秒禁攻击(防跳台掉落): 锁定期间不攻击, 只逼近
                        if self._jump_attack_locked:
                            return f"move_{_hdir}", "hunt_jump_lock"
                        if now >= self.next_attack_time:
                            self.next_attack_time = now + self._next_attack_delay()
                            return f"attack_{_hdir}", "hunt_attack"
                        # 冷却中: 继续逼近(朝怪走)
                        return f"move_{_hdir}", "hunt_cooldown"
                    # 不在攻击范围(或锁定保持远离) -> 朝怪走(追怪)
                    # 【追击卡住检测已取消】: 用户反馈它会怪刷不干净就走;
                    # 追怪被挡时保持追(怪仍在, 且方向防抖/漏检保持兜底)。
                    return f"move_{_hdir}", "hunt_move"
            if player and not _wp_busy:
                _pc = self._center(player)
                # 【已取消置信度低不攻击限制】: 玩家识别稳定, 不再因置信度
                # 漏攻击; 跳跃后 2 秒禁攻击由 _last_jump_at 控制(防跳台掉落)。
                if (advisory is not None and advisory.get("attack_ready")
                        and advisory.get("target_box") is not None):
                    _tb = advisory["target_box"]
                    _tc = self._center({"box": tuple(_tb)})
                    # 误检过滤: 目标中心落在玩家框内 = 把玩家自己当怪 -> 不攻击
                    _pb = player.get("box")
                    _self_bbox = (_pb is not None
                                  and _pb[0] <= _tc[0] <= _pb[0] + _pb[2]
                                  and _pb[1] <= _tc[1] <= _pb[1] + _pb[3])
                    if not _self_bbox:
                        _dir = "left" if _tc[0] < _pc[0] else "right"
                        # 记录目标方向缓存(供被撞退时兜底, 防异步延迟反向)
                        self._last_target_dir = _dir
                        self._last_target_time = now
                        # 跳跃后 2 秒禁攻击(防跳台掉落): 锁定期间不攻击不转向
                        if self._jump_attack_locked:
                            return "none", "attack_jump_lock"
                        if now >= self.next_attack_time:
                            self.next_attack_time = now + self._next_attack_delay()
                            return f"attack_{_dir}", "attack"
            if self._waypoint_patrol is not None:
                _wp = self._waypoint_patrol
                # 【休息机制已移除】(2026-08-27): 用户反馈休息机制仍不能躲过
                # 测谎仪, 暂时去掉——不再每 N 圈回初始点坐椅子休息, 永远巡航。
                # (maybe_rest 调用已移除; 相关代码保留供日后重新启用)
                _wpd = _wp.decide(
                    self._mini["map_norm"] if self._mini else None, now)
                if _wpd is not None:
                    return _wpd
            # 无路线/小地图数据: 原地待机(不做任何巡游)
            return "none", "wp_idle"
        if not player:
            # 玩家丢失: 左右巡游直到找到玩家
            return self._patrol_decision(now, player)
        if advisory is None:
            return "none", "no_advisory"

        status = advisory["status"]
        if status == "PLAYER MISSED":
            return "none", "player_missed"
        if status == "NO TARGET":
            # 定向清怪: 前方连续没怪一段时间(确认非漏检)才转身, 否则巡逻
            back_count = advisory.get("back_count", 0)
            turn_confirm_delay = 1.2  # 秒: 前方没怪持续这么久才转身
            if (back_count > 0 and self.facing_direction is not None
                    and now - self._last_front_seen >= turn_confirm_delay):
                self.facing_direction = (
                    "right" if self.facing_direction == "left" else "left")
                return f"move_{self.facing_direction}", "turn_around"
            if self.patrol_enabled:
                return self._patrol_decision(now, player)
            return "none", "no_target"
        if status == "WAITING":
            # Actively hunt instead of standing still.
            if self.patrol_enabled:
                return self._patrol_decision(now, player)
            return "none", "waiting"
        if status == "PAUSED CAMERA":
            return "none", "paused_camera"

        target_box = advisory.get("target_box")
        if target_box is None:
            if self.patrol_enabled:
                return self._patrol_decision(now, player)
            return "none", "no_target_box"

        # 前方有怪: 记录时间, 供转身确认延迟判断(避免短暂漏检就转身)
        self._last_front_seen = now

        # Recompute geometry from the boxes: the policy is self-contained and
        # does not trust advisory distance fields (which tests may construct).
        target_center = self._center({"box": tuple(target_box)})
        player_center = self._center(player)
        horizontal_distance = abs(target_center[0] - player_center[0])
        vertical_distance = abs(target_center[1] - player_center[1])
        target_is_left = target_center[0] < player_center[0]

        # Skip targets that were attacked several times but never died: they
        # are likely drop items or detection artifacts, and attacking them
        # makes the character stand in place forever.
        if self._is_target_ignored(target_center, now):
            if self.patrol_enabled:
                return self._patrol_decision(now, player)
            return "none", "target_ignored"

        # 站桩模式: 人物不动, 前方怪直接攻击, 头顶怪(上方, 纵向差>60)跳跃攻击
        if self.mode == "stationary":
            direction = "left" if target_is_left else "right"
            if horizontal_distance <= self.attack_horizontal_px:
                self.facing_direction = direction
                target_above = target_center[1] < player_center[1]
                # (已去掉) 头顶怪跳击: 跳跃太频繁会导致识别框漂移, 头顶怪也走下方直接攻击逻辑
                if vertical_distance <= self.attack_vertical_px:
                    # 前方/近怪(非头顶, 或头顶但很近) → 直接攻击
                    if now >= self.next_attack_time:
                        self.next_attack_time = now + self._next_attack_delay()
                        return f"attack_{direction}", "attack"
                    return "none", "attack_cooldown"
                # 脚下怪(下方, 纵向太远) → 不跳击, 原地等
                return "none", "below_range"
            # 横向超出攻击范围 → 不移动(站桩等怪)
            return "none", "out_of_range"

        # Melee characters attack ONLY what is in front of them: the attack
        # range is the area in front of the character, not a 360deg bubble.
        # The facing is updated on attack and while approaching, so the
        # warrior turns to face the target before it can be hit.
        in_front = (
            self.facing_direction is None
            or (self.facing_direction == "left" and target_is_left)
            or (self.facing_direction == "right" and not target_is_left)
        )
        attack_ready = (
            in_front
            and horizontal_distance <= self.attack_horizontal_px
            and vertical_distance <= self.attack_vertical_px
        )
        cooldown_ready = now >= self.next_attack_time

        # Attack first: any monster inside the front attack box gets hit
        # immediately (no wait-for-turn).
        if attack_ready and cooldown_ready:
            self.next_attack_time = now + self._next_attack_delay()
            self.facing_direction = "left" if target_is_left else "right"
            self._record_attack_attempt(target_center, now)
            return f"attack_{self.facing_direction}", "attack"

        # Melee (warrior): stand ground. In attack range but cooling down ->
        # wait in place (no back-off, no hop-away); out of range -> approach.
        if self.is_melee:
            if attack_ready:
                return "none", "attack_cooldown"
            # Approach: walk toward the target; this also turns the facing
            # so the front attack box eventually covers the target.
            direction = "left" if target_is_left else "right"
            self.facing_direction = direction
            return f"move_{direction}", "approach"

        # In attack range but cooling down: back off if the monster is too
        # close (mage should not stand next to it), otherwise wait in place.
        if attack_ready:
            if horizontal_distance < self.min_engage_px:
                # (已去掉) 跳跃躲避: 跳跃太频繁导致识别漂移, 改为水平后撤
                direction = "right" if target_is_left else "left"
                return f"move_{direction}", "keep_distance"
            return "none", "attack_cooldown"

        # Not in attack range: if the target is dangerously close, step away.
        if horizontal_distance <= self.immediate_danger_px:
            # (已去掉) 跳跃躲避: 改为水平后撤
            if vertical_distance > self.dodge_vertical_px:
                # Close but vertically unreachable and jump on cooldown:
                # back off horizontally instead of hopping repeatedly.
                direction = "right" if target_is_left else "left"
                return f"dodge_{direction}", "dodge_imminent"
            direction = "right" if target_is_left else "left"
            return f"dodge_{direction}", "dodge_imminent"

        # Not in attack range but closer than the engagement distance:
        # back off to ranged comfort instead of advancing into melee range.
        if horizontal_distance < self.min_engage_px:
            direction = "right" if target_is_left else "left"
            return f"move_{direction}", "keep_distance"

        # Otherwise approach the target horizontally.
        direction = "left" if target_is_left else "right"
        return f"move_{direction}", "approach"


# --------------------------------------------------------------------------
# Combat executor: converts commands into keyboard input.
# --------------------------------------------------------------------------
class CombatExecutor:
    def __init__(self, cfg, dry_run=False, mode="normal"):
        self.mode = mode  # "normal" | "stationary"(站桩模式攻击不转向)
        keys = cfg["key"]
        self.attack_key = keys["directional_attack"]
        self.add_hp_key = keys.get("add_hp") or ""
        self.add_mp_key = keys.get("add_mp") or ""
        self.jump_key = keys.get("jump") or "space"
        self.up_key = keys.get("up") or "up"
        self.down_key = keys.get("down") or "down"
        self.pickup_key = keys.get("pickup") or "z"  # 捡东西键(已停用, 保留字段)
        self.feed_key = keys.get("feed") or "n"      # 喂宠物键(每30分钟按一次)
        self.dry_run = dry_run
        self.counts = {}
        self.pressed_keys = []  # key names "pressed" in dry-run for tests
        self.held_move = None   # currently held movement key ("left"/"right"/None)
        self.held_vert = None   # currently held vertical key ("up"/"down"/None)
        self._last_attack_dir = None  # 上次攻击方向(方向变了才转向, 避免连续同向攻击反复走步)
        self._cur_dir = None          # 当前实际朝向(由执行的移动/跳跃/攻击命令实时更新, 用于攻击转向判断)
        # 【真实朝向 _real_dir】: 由主循环每帧根据玩家检测的真实速度(velocity_px_s)
        # 校准。相比于 _cur_dir(只按"发过什么命令"模拟, step/move 被墙挡、跳跃后
        # 都会与实际脱节), 真实朝向才是游戏角色此刻面对的方向。攻击转向判断用它,
        # 避免"怪在玩家后方却朝正前方攻击"(模拟朝向与实际脱节的偶发问题)。
        self._real_dir = None         # "left"/"right", None=未知
        self._real_dir_ts = 0.0       # 最近一次真实朝向更新时间
        self.feed_interval = 30 * 60    # 喂宠物键(N)间隔(秒): 每30分钟按一次
        self.last_feed_time = float("-inf")

        auto_cfg = cfg.get("auto_combat", {})
        self.attack_hold_seconds = float(auto_cfg.get("attack_hold_seconds", 0.08))
        self.dodge_hold_seconds = float(auto_cfg.get("dodge_hold_seconds", 0.18))
        self.step_hold_seconds = float(auto_cfg.get("step_hold_seconds", 0.06))
        # 下跳参数(minimap_waypoint): 先按住↓再起跳穿越台子, 穿过即松开防穿透下层
        _wp = cfg.get("minimap_waypoint", {})
        self.jump_down_pre_hold = float(_wp.get("jump_down_pre_hold", 0.15))
        self.jump_down_hold_seconds = float(_wp.get("jump_down_hold_seconds", 0.55))
        self.turn_hold_seconds = float(auto_cfg.get("turn_hold_seconds", 0.08))
        self.turn_pause_seconds = float(auto_cfg.get("turn_pause_seconds", 0.05))
        self.rest_hold_seconds = float(auto_cfg.get("rest_hold_seconds", 0.5))
        self.add_hp_cooldown = float(cfg["health_monitor"]["add_hp_cooldown"])
        self.add_mp_cooldown = float(cfg["health_monitor"]["add_mp_cooldown"])
        self.add_hp_threshold = float(cfg["health_monitor"]["add_hp_percent"])
        self.add_mp_threshold = float(cfg["health_monitor"]["add_mp_percent"])
        self.last_add_hp_time = float("-inf")
        self.last_add_mp_time = float("-inf")
        # Potion-protection: if HP stays below the threshold after 3 drinks in
        # a row (9s), the potion is not restoring faster than the damage, or
        # the potion pouch is empty. Stop drinking instead of burning the
        # whole pouch while the character keeps getting hit.
        self.hp_drink_streak = 0
        self.mp_drink_streak = 0

    def _count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def set_real_facing(self, velocity_x, now=None):
        """主循环每帧调用: 用玩家检测的真实速度校准真实朝向。

        velocity_x 为 player["velocity_px_s"][0](像素/秒, 平滑后):
        - vx < -阈值(明显向左运动) -> 真实朝向左
        - vx > +阈值(明显向右运动) -> 真实朝向右
        - 中间(静止/低速) -> 保持上次朝向(角色不动时面朝不变)
        用真实朝向替代纯模拟 _cur_dir, 修复"怪在玩家后方却不转向"的偶发问题。
        """
        if velocity_x is None:
            return
        self._real_dir_ts = now if now is not None else time.time()
        if velocity_x <= -30.0:
            self._real_dir = "left"
        elif velocity_x >= 30.0:
            self._real_dir = "right"
        # 低速(静止)不更新: 保持上次已知朝向

    def _press(self, key, hold_seconds):
        if not key:
            return
        if self.dry_run:
            self.pressed_keys.append(key)
            return
        press_key(key, hold_seconds)

    def _set_move(self, direction):
        """Hold or release a movement key. Stateful: only sends key events on change."""
        if direction not in (None, "left", "right"):
            raise ValueError(f"Unsupported move direction: {direction}")
        if direction == self.held_move:
            return
        if self.held_move is not None:
            if self.dry_run:
                self.pressed_keys.append(f"up_{self.held_move}")
            else:
                key_up(self.held_move)
        if direction is not None:
            if self.dry_run:
                self.pressed_keys.append(f"down_{direction}")
            else:
                key_down(direction)
        self.held_move = direction

    def _set_vert(self, direction):
        """Hold or release a vertical key (up/down). Used for ropes/ladders."""
        if direction not in (None, "up", "down"):
            raise ValueError(f"Unsupported vertical direction: {direction}")
        if direction == self.held_vert:
            return
        if self.held_vert is not None:
            if self.dry_run:
                self.pressed_keys.append(f"up_{self.held_vert}")
            else:
                key_up(self.held_vert)
        if direction is not None:
            if self.dry_run:
                self.pressed_keys.append(f"down_{direction}")
            else:
                key_down(direction)
        self.held_vert = direction

    def _tap_feed(self, now):
        """每30分钟按 N 键喂宠物一次(用户带了宠物, 不用再按 Z 捡东西)。"""
        if not self.feed_key:
            return
        if now - self.last_feed_time < self.feed_interval:
            return
        self.last_feed_time = now
        self._press(self.feed_key, 0.05)
        self._count("feed")

    def handle_potions(self, hp_percent, mp_percent, now):
        """Consume potions first; returns True if a potion was used.

        Potion protection: 3 consecutive drinks that do not lift HP/MP above
        the threshold stop the drinking loop (pouch empty or incoming damage
        exceeds the potion heal) -- otherwise the character burns the whole
        pouch while getting hit. The streak resets as soon as the bar is
        above the threshold."""
        if hp_percent is not None and hp_percent > self.add_hp_threshold:
            self.hp_drink_streak = 0
        if mp_percent is not None and mp_percent > self.add_mp_threshold:
            self.mp_drink_streak = 0
        if (
            hp_percent is not None
            and self.add_hp_key
            and hp_percent <= self.add_hp_threshold
            and now - self.last_add_hp_time >= self.add_hp_cooldown
            and self.hp_drink_streak < 3
        ):
            self.hp_drink_streak += 1
            self.last_add_hp_time = now
            self._press(self.add_hp_key, 0.05)
            self._count("add_hp")
            return True
        if (
            mp_percent is not None
            and self.add_mp_key
            and mp_percent <= self.add_mp_threshold
            and now - self.last_add_mp_time >= self.add_mp_cooldown
            and self.mp_drink_streak < 3
        ):
            self.mp_drink_streak += 1
            self.last_add_mp_time = now
            self._press(self.add_mp_key, 0.05)
            self._count("add_mp")
            return True
        return False

    def execute(self, command, reason, hp_percent=None, mp_percent=None, now=None,
                suppress_feed=False):
        if now is None:
            now = time.time()
        if self.handle_potions(hp_percent, mp_percent, now):
            self._count("potion_gate")
            return
        # 挂机(检测到其他玩家)时停止喂食等一切行为, 只喝药
        if not suppress_feed:
            self._tap_feed(now)  # 每30分钟按N喂宠物(不管什么命令都定时触发)

        if command == "none":
            self._set_move(None)
            return
        if command.startswith("press_"):
            # 通用单键按下(坐椅子 x / 任意功能键): press_<key>
            rest_key = command.split("_", 1)[1]
            self._set_move(None)
            self._press(rest_key, self.rest_hold_seconds)
            self._count(command)
            return
        if command.startswith("attack_"):
            direction = command.split("_", 1)[1]
            # patrol_hunt(巡游打怪): 攻击时保持移动(边走边打), 不停步
            if self.mode != "patrol_hunt":
                self._set_move(None)
            # 转向: stationary/patrol_hunt 不转向(只打前方怪, 朝向已对); 其他模式
            # 朝怪转向。用【当前实际朝向 _cur_dir】(由 move/step/jump/attack 实时
            # 更新)判断, 而不是 _last_attack_dir(上次攻击方向)——否则角色走位改变
            # 朝向后, 怪从另一侧进框会因方向"恰好等于上次攻击方向"而不转向, 打空气。
            # 需要转向: 目标方向与当前实际朝向相反时才转身。
            # 【优化】优先用真实朝向 _real_dir(主循环每帧用玩家真实速度校准),
            # 它比 _cur_dir(纯命令模拟, step/move 被墙挡/跳跃后易脱节)可靠——
            # 修复"怪在玩家后方却朝正前方攻击"的偶发问题。_real_dir 未知时
            # 回退 _cur_dir。
            _facing = self._real_dir or self._cur_dir
            need_turn = (self.mode not in ("stationary", "patrol_hunt")
                         and direction != _facing)
            turn_key = "left" if direction == "left" else "right"
            if need_turn:
                # 修复"玩家没转就攻击"(空挥): 之前是"点按一下方向键→松开→停顿→再
                # 点攻击键", 转身动画还没播完攻击就按下了, 游戏读到的是原朝向,
                # 于是朝无怪方向空挥。改为【按住方向键转身, 并贯穿整个攻击过程】:
                # 冒险岛方向技能在按下攻击键的瞬间读取按住的方向键, 这样无论转身
                # 动画是否播完都保证朝目标方向打出。
                if not self.dry_run:
                    key_down(turn_key)
                    time.sleep(self.turn_hold_seconds)
                else:
                    self.pressed_keys.append(f"down_{turn_key}")
            self._last_attack_dir = direction
            self._cur_dir = direction
            self._press(self.attack_key, self.attack_hold_seconds)
            if need_turn:
                # 攻击结束才松开方向键, 保证攻击全程方向键处于按住状态。
                if not self.dry_run:
                    key_up(turn_key)
                else:
                    self.pressed_keys.append(f"up_{turn_key}")
            self._count(command)
        elif command.startswith("dodge_"):
            direction = command.split("_", 1)[1]
            self._set_move(None)
            self._press(direction, self.dodge_hold_seconds)
            self._cur_dir = direction
            self._count(command)
        elif command in ("step_left", "step_right"):
            # 近距离微调步进: 极短按方向键走一小步(约 1px 小地图), 用于跳跃点
            # 精确对齐, 避免全速走冲过头(norm 更新慢)
            direction = command.split("_", 1)[1]
            self._set_move(None)
            self._press(direction, self.step_hold_seconds)
            self._cur_dir = direction
            self._count(command)
        elif command.startswith("move_"):
            direction = command.split("_", 1)[1]
            # Horizontal movement cancels any vertical hold.
            self._set_vert(None)
            self._set_move(direction)
            self._cur_dir = direction
            self._count(command)
        elif command == "jump":
            # 普通跳跃(平台间跳): 起跳【不按上】, 避免误抓绳子/影响跳点
            self._set_move(None)
            self._set_vert(None)
            self._press(self.jump_key, 0.08)
            self._count("jump")
        elif command == "jump_down":
            # 下跳(跳到下方同X平台): 【先按住↓键, 再起跳, ↓保持穿过台子后松开】。
            # 只按跳键会跳起落回原台子; 先按下键再跳, 角色从站立平台直接落到
            # 下层平台(用户要求: 先下键后跳跃)。↓保持太久会穿透更下层平台,
            # 因此穿过台子(起跳后 jump_down_hold_seconds)就松开。
            self._set_move(None)
            self._set_vert("down")                        # ①先按住↓(穿台子必需)
            if not self.dry_run:
                time.sleep(self.jump_down_pre_hold)       # 提前量: 保证按下键已生效
            self._press(self.jump_key, 0.08)              # ②按住↓状态下起跳
            if not self.dry_run:
                time.sleep(self.jump_down_hold_seconds)   # ③下落过程中保持↓穿过台子
            self._set_vert(None)                          # ④已穿过台子, 松开↓
            self._count("jump_down")
        elif command == "jump_climb":
            # 爬绳跳(旧兼容): 起跳后立刻按住上(跳起碰到绳子即抓绳)
            self._set_move(None)
            self._set_vert(None)
            self._press(self.jump_key, 0.08)
            self._set_vert("up")
            self._count("jump_climb")
        elif command.startswith("jump_climb_"):
            # 抓绳跳(带方向): 起跳 -> 空中朝绳子方向跳 -> 一直按住上抓绳
            # 用户实测: 离绳子 0.03 处朝绳跳+按上才能抓到, 太近(0.01内)抓不住
            direction = command.split("_", 2)[2]   # left / right
            self._set_move(None)
            self._set_vert(None)
            if self.dry_run:
                self.pressed_keys.append(self.jump_key)
                self.pressed_keys.append(direction)
            else:
                self._press(self.jump_key, 0.08)
                key_down(direction)                # 空中朝绳子方向跳
                time.sleep(0.12)                   # 空中移动
                key_up(direction)
            self._set_vert("up")                   # 一直按着上抓绳
            self._cur_dir = direction
            self._count(command)
        elif command == "climb_up":
            # Stateful: keeps up held until something else cancels.
            self._set_move(None)
            self._set_vert("up")
            self._count("climb_up")
        elif command == "climb_down":
            self._set_move(None)
            self._set_vert("down")
            self._count("climb_down")
        elif command.startswith("jump_attack_"):
            # 跳跃攻击: 先跳, 等角色升到空中再攻击(才能打到头顶的怪)。
            # 方向键按住并贯穿整个跳+空中攻击过程, 保证空中方向技能朝目标方向打。
            direction = command.split("_", 2)[2]
            self._set_move(None)
            need_turn = self.mode != "stationary"
            turn_key = "left" if direction == "left" else "right"
            if need_turn:
                if not self.dry_run:
                    key_down(turn_key)
                    time.sleep(self.turn_hold_seconds)
                else:
                    self.pressed_keys.append(f"down_{turn_key}")
            self._cur_dir = direction
            self._press(self.jump_key, 0.08)   # 短按跳键, 触发跳跃
            time.sleep(0.15)                   # 等角色上升到空中
            self._press(self.attack_key, self.attack_hold_seconds)  # 空中攻击
            if need_turn:
                if not self.dry_run:
                    key_up(turn_key)
                else:
                    self.pressed_keys.append(f"up_{turn_key}")
            self._count(command)
        elif command.startswith("jump_"):
            # e.g. jump_left / jump_right: 短按跳键触发跳跃, 起跳后空中按方向
            # (平台间斜跳, 起跳不按上)。空中方向按 0.16s——同层台阶跳跃需要
            # 足够的空中位移(如 0.3566->0.3156 跳 0.04), 0.12s 太短跳不到位
            direction = command.split("_", 1)[1]
            self._set_move(None)
            self._set_vert(None)
            if self.dry_run:
                self.pressed_keys.append(self.jump_key)
                self.pressed_keys.append(direction)
            else:
                # 方向跳: 先按住方向键, 再按跳键——冒险岛跳跃方向由起跳瞬间的
                # 方向键决定, 必须先方向后跳(先跳再方向角色会立定跳)
                key_down(direction)                # 先按住方向键
                time.sleep(0.05)                   # 让方向先生效
                self._press(self.jump_key, 0.08)   # 再按跳键(带方向起跳)
                time.sleep(0.12)                   # 空中继续朝方向移动
                key_up(direction)
            self._cur_dir = direction
            self._count(command)
        else:
            raise ValueError(f"Unsupported command: {command}")

    def release_all(self):
        self._set_move(None)
        self._set_vert(None)
        if not self.dry_run:
            key_up(self.attack_key)


# --------------------------------------------------------------------------
# Player detector fallback: locate the character by reading their name with
# OCR. Much more stable than the small nametag template, which is sensitive
# to font and anti-alias changes.
# --------------------------------------------------------------------------
class NameOcrPlayerDetector:
    """Locate the player by matching their character name via rapidocr."""

    def __init__(self, cfg, player_name, confidence=0.5, ocr_engine=None):
        self.player_name = player_name
        self.confidence = float(confidence)
        self.ui_y_start = int(cfg["ui_coords"]["ui_y_start"])
        overlay_cfg = cfg["perception_overlay"]
        self.box_width, self.box_height = overlay_cfg["player_box_size"]
        self.name_bottom_offset = int(
            cfg.get("auto_combat", {}).get("name_bottom_offset", 30)
        )
        # Fallback: the 33x13px nametag is too small for RapidOCR to read
        # reliably (it splits "麻超圆" into fragments). The character's title
        # badge ("新手冒险家勋章") is much larger (131x16) and OCRs at ~0.9;
        # in MapleStory the title sits BELOW the name and ABOVE the character,
        # so when the name cannot be read we locate the player via the title.
        self.title_keywords = (
            cfg.get("auto_combat", {})
            .get("title_keywords", ("勋章", "冒家", "助章", "冒险家"))
        )
        self.title_bottom_offset = int(
            cfg.get("auto_combat", {}).get("title_bottom_offset", 22)
        )
        if ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR

            ocr_engine = RapidOCR()
        self.ocr = ocr_engine

    def _zoom_read_name(self, frame, px, py):
        """用玩家中心(称号/片段定位)裁剪头顶名字区,放大 4x OCR 读名字.
        33x13px 的名字放大到 ~132x52 后 RapidOCR 可稳定读出"麻超圆"
        (全图/宽区域 OCR 时名字太小不稳定). 返回 player dict 或 None."""
        try:
            img_h, img_w = frame.shape[:2]
            x0 = max(0, int(px) - 60)
            x1 = min(img_w, int(px) + 60)
            y0 = max(0, int(py) - 60)
            y1 = min(img_h, int(py) - 10)
            if x1 - x0 < 30 or y1 - y0 < 10:
                return None
            crop = frame[y0:y1, x0:x1]
            big = cv2.resize(
                crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
            )
            result = self.ocr(big)
            if isinstance(result, tuple) and len(result) >= 1:
                boxes = result[0]
            elif (
                isinstance(result, list)
                and len(result) == 1
                and isinstance(result[0], list)
                and result[0]
                and isinstance(result[0][0], list)
            ):
                boxes = result[0]
            else:
                boxes = result
            for entry in boxes or []:
                if len(entry) < 3:
                    continue
                box, text, score = entry[0], entry[1], entry[2]
                if score < self.confidence:
                    continue
                text_s = str(text)
                # 容错: 名字区放大后 RapidOCR 仍可能认错 1-2 个字
                # ("麻超圆" -> "麻视圆人"), 且该区域已被称号锁定为玩家头顶,
                # 所以只需首尾字("麻...圆")匹配即可, 位置依然准确.
                full = self.player_name in text_s
                loose = (
                    self.player_name[0] in text_s
                    and self.player_name[-1] in text_s
                    and len(text_s) <= 5
                )
                if not (full or loose):
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                px2 = int(round((x_min + x_max) / 2.0 / 4.0)) + x0
                py2 = (
                    int(round(y_max / 4.0)) + y0 + self.name_bottom_offset
                )
                return {
                    "label": "PLAYER",
                    "score": float(score),
                    "box": (
                        max(0, px2 - self.box_width // 2),
                        max(0, py2 - self.box_height // 2),
                        self.box_width,
                        self.box_height,
                    ),
                    "nametag_box": (
                        int(round(x_min / 4.0)) + x0,
                        int(round(y_min / 4.0)) + y0,
                        int(round((x_max - x_min) / 4.0)),
                        int(round((y_max - y_min) / 4.0)),
                    ),
                    "center": (px2, py2),
                    "source": "name_zoom",
                }
        except Exception:
            pass
        return None

    def detect(self, frame, near_center=None):
        # OCR input: skip the top HUD/minimap band (character names never
        # appear there) and downscale wide frames so each inference is much
        # faster (~9s -> ~3s). All coordinates are scaled back below.
        gameplay = frame[: self.ui_y_start]
        y_offset = 0
        if gameplay.shape[0] > 220:
            y_offset = int(gameplay.shape[0] * 0.10)
            gameplay = gameplay[y_offset:]
        # Downscale only very wide frames. The name tag is ~33x13px at the
        # native 1278x750 resolution; a fixed 0.6x would shrink it to ~20x8px
        # which RapidOCR cannot read ("player not found"). Frames wider than
        # 1500px (e.g. 1942x1136) still keep a readable ~15px-tall tag.
        ocr_scale = 1.0
        if gameplay.shape[1] > 1500:
            ocr_scale = 0.6
            gameplay = cv2.resize(
                gameplay, None, fx=ocr_scale, fy=ocr_scale,
                interpolation=cv2.INTER_AREA,
            )
        result = self.ocr(gameplay)
        # rapidocr returns a list of [box, text, score]; older builds wrap it
        # as ([list], elapsed). Normalize to a flat list of entries.
        if isinstance(result, tuple) and len(result) >= 1:
            boxes = result[0]
        elif (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], list)
            and result[0]
            and isinstance(result[0][0], list)
        ):
            boxes = result[0]
        else:
            boxes = result
        if not boxes:
            return None
        best = None
        best_score = -1.0
        for entry in boxes:
            # rapidocr returns: [[box_points], text, confidence]
            if len(entry) < 3:
                continue
            box, text, score = entry[0], entry[1], entry[2]
            if score < self.confidence:
                continue
            if self.player_name not in str(text):
                continue
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # The name floats above the character's head: the body center is
            # below the bottom of the name box, not above it.
            player_x = int(round((x_min + x_max) / 2.0 / ocr_scale))
            player_y = (
                int(round(y_max / ocr_scale))
                + self.name_bottom_offset
                + y_offset
            )
            box_x = max(0, player_x - self.box_width // 2)
            box_y = max(0, player_y - self.box_height // 2)
            if score > best_score:
                best_score = score
                best = {
                    "label": "PLAYER",
                    "score": float(score),
                    "box": (box_x, box_y, self.box_width, self.box_height),
                    "nametag_box": (
                        int(round(x_min / ocr_scale)),
                        int(round(y_min / ocr_scale)) + y_offset,
                        int(round((x_max - x_min) / ocr_scale)),
                        int(round((y_max - y_min) / ocr_scale)),
                    ),
                    "center": (player_x, player_y),
                }
        if best is not None:
            return best
        # Full name unreadable. Fallback ladder:
        #  1) a name FRAGMENT (starts with the family character, e.g. "麻圆")
        #     vertically aligned with a title badge below it -> that pair is
        #     the player (title directly under name, above the sprite).
        #  2) a title badge alone (single-player maps / few badge users).
        title_cands = []   # entries matching the title keywords
        name_cands = []    # short text blocks (2-8 chars) anywhere
        for entry in boxes:
            if len(entry) < 3:
                continue
            box, text, score = entry[0], entry[1], entry[2]
            if score < self.confidence:
                continue
            text_s = str(text)
            if any(k in text_s for k in self.title_keywords):
                title_cands.append((float(score), box, text_s))
            elif len(text_s) <= 8:
                # The 33x13px nametag is frequently MIS-READ by RapidOCR
                # ("麻超圆" -> "林超"/"麻圆"/"他命"), so the name must NOT
                # be required to contain the family char. Any short text block
                # can be the name; the vertical alignment with the title badge
                # below it (name sits directly above the title) filters out
                # UI labels, map names and monster names.
                name_cands.append((float(score), box, text_s))
        for n_score, n_box, n_text in name_cands:
            n_xs = [float(p[0]) for p in n_box]
            n_ys = [float(p[1]) for p in n_box]
            n_cx = (min(n_xs) + max(n_xs)) / 2.0
            n_bot = max(n_ys)
            for t_score, t_box, t_text in title_cands:
                t_xs = [float(p[0]) for p in t_box]
                t_ys = [float(p[1]) for p in t_box]
                t_cx = (min(t_xs) + max(t_xs)) / 2.0
                t_top = min(t_ys)
                gap = t_top - n_bot
                if 0 <= gap <= 45 and abs(n_cx - t_cx) <= 90:
                    player_x = int(round(n_cx / ocr_scale))
                    player_y = (
                        int(round(n_bot / ocr_scale))
                        + self.name_bottom_offset
                        + y_offset
                    )
                    return {
                        "label": "PLAYER",
                        "score": n_score,
                        "box": (
                            max(0, player_x - self.box_width // 2),
                            max(0, player_y - self.box_height // 2),
                            self.box_width,
                            self.box_height,
                        ),
                        "nametag_box": (
                            int(round(min(n_xs) / ocr_scale)),
                            int(round(min(n_ys) / ocr_scale)) + y_offset,
                            int(round((max(n_xs) - min(n_xs)) / ocr_scale)),
                            int(round((max(n_ys) - min(n_ys)) / ocr_scale)),
                        ),
                        "center": (player_x, player_y),
                        "source": "name_title_pair",
                    }
        # 2) Title alone (last resort). Multiple players may share the title
        #    badge, so pick the one closest to the assumed player position.
        #    Without a near_center the title detector used to pick an on-character
        #    decoration badge sitting near the player's feet (a worn icon), not
        #    the title hovering over the head. Use the screen center as the
        #    default anchor: the player is normally drawn near the middle of
        #    the gameplay area, and the head title sits close to that anchor
        #    while foot decorations sit further away.
        img_h, img_w = frame.shape[:2]
        if near_center is None:
            near_center = (img_w // 2, int(img_h * 0.55))
        if title_cands:
            t_score, t_box, _ = title_cands[0]
            if len(title_cands) > 1:
                nc_x, nc_y = near_center
                best_dist = None
                for cand_score, cand_box, _ in title_cands:
                    c_xs = [float(p[0]) for p in cand_box]
                    c_ys = [float(p[1]) for p in cand_box]
                    c_cx = (min(c_xs) + max(c_xs)) / 2.0
                    c_cy = max(c_ys)
                    d = abs(c_cx - nc_x) + abs(c_cy - nc_y)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        t_score, t_box = cand_score, cand_box
            xs = [float(pt[0]) for pt in t_box]
            ys = [float(pt[1]) for pt in t_box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            player_x = int(round((x_min + x_max) / 2.0 / ocr_scale))
            player_y = (
                int(round(y_max / ocr_scale))
                + self.title_bottom_offset
                + y_offset
            )
            # Before falling back to the title badge, zoom the nametag strip
            # above the located player and re-OCR it: the 33x13px name becomes
            # ~132x52 and reads reliably ("麻超圆").
            zoomed = self._zoom_read_name(frame, player_x, player_y)
            if zoomed is not None:
                return zoomed
            return {
                "label": "PLAYER",
                "score": t_score,
                "box": (
                    max(0, player_x - self.box_width // 2),
                    max(0, player_y - self.box_height // 2),
                    self.box_width,
                    self.box_height,
                ),
                # nametag_box must be the NAME strip, not the title badge:
                # the PlayerLocator cuts its full-frame match template from
                # this box, and the title ("新手冒险家勋章") is shared by every
                # character while the name is unique. The name sits directly
                # ABOVE the title: derive it from the title box geometry.
                "nametag_box": (
                    int(round(player_x - 22)),
                    int(round(player_y - self.name_bottom_offset - 16)),
                    44,
                    14,
                ),
                "center": (player_x, player_y),
                "source": "title",
            }
        return None


# --------------------------------------------------------------------------
# Player locator: template detection every frame (fast, ~30ms), OCR fallback
# running in a background thread so the slow (~2-5s) OCR inference NEVER
# blocks the main loop. Position stickiness covers OCR latency.
# --------------------------------------------------------------------------
class PlayerLocator:
    """Player locator = multi-pose character-template matching + OCR seeding.

    The OCR thread (local crop, ~1s per run) finds "<player_name>" and seeds
    a template of the CHARACTER SPRITE (the 70x90 box around the player
    center). Every OCR success adds another pose to the template library
    (standing/walking/jumping/attacking/turning are collected over time, up
    to `max_tag_templates`). Each frame every template is matched inside a
    tight window around the last position (milliseconds), so the box follows
    a moving/jumping character in real time. If the player jumps out of the
    window the score collapses, the match is rejected, and the next OCR run
    re-seeds. Templates are deduplicated so identical poses are not re-added.
    """

    def __init__(self, template_detector, ocr_detector, refresh_frames=3,
                 sticky_seconds=1.5, ocr_max_age=None, tag_threshold=0.3,
                 tag_step=90, tag_range=110, max_tag_templates=10,
                 shift_up_px=35):
        # template_detector is kept for backward compatibility with tests and
        # is intentionally NOT consulted: the old fixed nametag template
        # mis-matched unrelated on-screen text at low scores.
        self.template_detector = template_detector
        self.ocr_detector = ocr_detector
        self.shift_up_px = int(shift_up_px)
        self.sticky_seconds = float(sticky_seconds)
        self.ocr_max_age = (
            float(ocr_max_age)
            if ocr_max_age is not None
            else max(self.sticky_seconds * 4, 8.0)
        )
        # Character-sprite template library (multi-pose).
        self.tag_templates = []      # list of BGR crops (70x90 around center)
        self.tag_templates_gray = [] # grayscale versions (fast matching)
        self.tag_center = None       # last player center on the full frame
        self.tag_threshold = float(tag_threshold)
        self.tag_step = float(tag_step)      # max px/frame a real player moves
        self.tag_range = int(tag_range)      # search half-window around center
        self.max_tag_templates = int(max_tag_templates)
        # Full-frame nametag-text template. The mage uses an offline-recorded
        # nametag (nametag/shanda_legacy_player.png, 41x23 name + avatar) and
        # full-frame matchTemplate every frame: stable, fast, no OCR needed.
        # For other characters the template is auto-loaded from
        # nametag/<character>_player.png if present, otherwise the OCR-thread
        # result is used to seed one. Approximate match (>=0.70); the OCR
        # fallback only runs when the offline template misses.
        self.name_template = None    # tight nametag text crop (BGR)
        self.name_center = None      # last nametag center on the full frame
        self.name_threshold = 0.70   # offline template is exact -> higher bar
        self._offline_name_path = None
        self._max_template_diff = 8.0  # OCR-seeded template must resemble offline to be trusted
        # Blue-weapon (渔网) localization: the player's weapon is a bright
        # blue fishing net, uniquely colored, so it can be found by HSV every
        # frame in milliseconds. The weapon-to-player-center offset is learned
        # at seed time and applied afterwards (player = weapon + offset).
        self.weapon_blue_low = (85, 40, 40)
        self.weapon_blue_high = (140, 255, 255)
        self.weapon_offset = None   # (dx, dy): weapon center -> player center
        self.weapon_last = None     # last weapon position (small-window anchor)
        self.box_width, self.box_height = 70, 90
        self.name_bottom_offset = 30
        self.title_template = None          # BGR title badge template
        self.title_tpl_gray = None
        self.title_mask = None
        self.title_size = None
        self.title_bottom_offset = 22
        if ocr_detector is not None:
            if hasattr(ocr_detector, "box_width"):
                self.box_width = int(ocr_detector.box_width)
            if hasattr(ocr_detector, "box_height"):
                self.box_height = int(ocr_detector.box_height)
            if hasattr(ocr_detector, "name_bottom_offset"):
                self.name_bottom_offset = int(ocr_detector.name_bottom_offset)
        self.last_player = None
        self.last_seen = float("-inf")
        # Key-aware tracking: the main loop reports which direction the
        # character is moving (from the combat policy commands). detect()
        # then predicts where the player should be (last pos + move vector)
        # and REJECTS visual matches that are too far from that prediction
        # (this is what stopped the box teleporting to a far-away patch of
        # similar texture).
        self._move_dir = None      # "left" | "right" | None
        self._move_time = None     # last time a move key was pressed
        self._move_speed = 130.0   # px/s, rough Maple walk speed
        self.max_track_error_px = 220.0  # reject matches farther than this
        self.async_ocr = ocr_detector is not None
        self._ocr_lock = threading.Lock()
        self._ocr_frame = None
        self._ocr_result = None
        self._ocr_stamp = 0.0
        self._stop = threading.Event()
        # 实体坐标: 玩家速度/运动状态(codex 移植, 简化版)
        self._player_prev_center = None
        self._player_prev_time = 0.0
        self._player_velocity = [0.0, 0.0]
        self._thread = None
        self._warmup_done = threading.Event()
        # Original-style nametag tracker state (SQDIFF + cached local search).
        self._last_nametag_loc = None
        # Title badge anchor state (e.g. "新手冒险家勋章").
        self._last_title_loc = None
        if self.async_ocr:
            # RapidOCR lazily loads its ONNX models on the FIRST call (~4s).
            # Warm it up on a background thread so the first real OCR seed
            # takes ~1.3s instead of ~5s (this was why the player was only
            # found after standing still for several seconds).
            threading.Thread(target=self._warmup_ocr, daemon=True).start()
            self._thread = threading.Thread(target=self._ocr_loop, daemon=True)
            self._thread.start()

    def _warmup_ocr(self):
        try:
            tiny = np.zeros((64, 256, 3), dtype=np.uint8)
            self.ocr_detector.ocr(tiny)
        except Exception:
            pass
        finally:
            self._warmup_done.set()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame):
        if not self.async_ocr:
            return
        with self._ocr_lock:
            self._ocr_frame = frame

    def _ocr_on_frame(self, frame, local_center):
        """Run OCR on a small crop around local_center (or the full frame when
        local_center is None). The crop is far smaller, so inference is much
        faster (~1s vs ~4s), which keeps the box glued to a moving/jumping
        character. Returns (player_with_global_coords, new_local_center)."""
        offset = (0, 0)
        target = frame
        if local_center is not None:
            img_h, img_w = frame.shape[:2]
            cx, cy = local_center
            x0 = max(0, int(cx) - 260)
            y0 = max(0, int(cy) - 140)
            x1 = min(img_w, int(cx) + 260)
            y1 = min(img_h, int(cy) + 140)
            if x1 - x0 >= 120 and y1 - y0 >= 100:
                target = frame[y0:y1, x0:x1]
                offset = (x0, y0)
        player = self.ocr_detector.detect(target, near_center=local_center)
        if not player:
            return None, None
        cx, cy = player["center"]
        player["center"] = (cx + offset[0], cy + offset[1])
        bx, by, bw, bh = player["box"]
        player["box"] = (bx + offset[0], by + offset[1], bw, bh)
        nx, ny, nw, nh = player["nametag_box"]
        player["nametag_box"] = (nx + offset[0], ny + offset[1], nw, nh)
        return player, player["center"]

    def _ocr_loop(self):
        # The OCR thread remembers the last detected center so it can keep
        # working on a small fast crop; if the player moves out of the crop
        # (e.g. a jump), the next run falls back to the full frame.
        # Wait for the background model warm-up so the first seed is ~1.3s
        # instead of ~5s (model load).
        self._warmup_done.wait(timeout=10.0)
        local_center = None
        while not self._stop.is_set():
            with self._ocr_lock:
                fresh = time.time() - self._ocr_stamp < 1.0
                frame = self._ocr_frame
            if fresh:
                time.sleep(0.05)
                continue
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                player, new_center = self._ocr_on_frame(frame, local_center)
                local_center = new_center
                with self._ocr_lock:
                    self._ocr_result = player
                    self._ocr_stamp = time.time()
            except Exception:
                pass

    def _find_weapon(self, frame, center, radius=130):
        """在 center 附近找蓝色渔网块,返回其中心(全局坐标).

        The fishing net's HSV (125,64,68) is nearly identical to rocky
        crevices, so a SMALL window around the last weapon position is the
        primary search (the net moves continuously with the player); only a
        wide window is used as fallback."""
        try:
            cx, cy = int(center[0]), int(center[1])
            img_h, img_w = frame.shape[:2]
            x0 = max(0, cx - radius)
            x1 = min(img_w, cx + radius)
            y0 = max(0, cy - radius // 3)
            y1 = min(img_h, cy + radius)
            region = frame[y0:y1, x0:x1]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.weapon_blue_low, self.weapon_blue_high)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
            )
            n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
            best, best_d = None, None
            for i in range(1, n):
                area, bw, bh = stats[i][4], stats[i][2], stats[i][3]
                # Fishing net is a ~12x20 vertical strip; crevices are often
                # wider/taller, so constrain the shape a bit.
                if not (40 <= area <= 600 and 8 <= bw <= 35 and 10 <= bh <= 45):
                    continue
                wx = x0 + int(cent[i][0])
                wy = y0 + int(cent[i][1])
                d = abs(wx - cx) + abs(wy - cy)
                if best_d is None or d < best_d:
                    best_d, best = d, (wx, wy)
            return best
        except Exception:
            return None

    def _seed_weapon(self, frame, player_center):
        """播种时学习"渔网 -> 玩家中心"偏移. 找不到渔网时不清空旧偏移:
        可能只是这一帧渔网被姿势/特效遮挡,旧偏移在 detect 里有位移约束
        保护,不会把框带飞;下次找到会重新学习."""
        try:
            wc = self._find_weapon(frame, player_center, radius=150)
            if wc is not None:
                self.weapon_offset = (
                    player_center[0] - wc[0],
                    player_center[1] - wc[1],
                )
                self.weapon_last = wc
        except Exception:
            pass

    def load_offline_templates(self, directory):
        """加载离线采集的角色形象模板(玩家本人不同姿势截图),作为初始
        匹配库;在线 OCR 播种后仍会补充新姿势. 返回加载数量."""
        from src.utils.common import imread_cn
        try:
            d = Path(directory)
            if not d.is_dir():
                return 0
            count = 0
            for f in sorted(d.glob("*.png")):
                tpl = imread_cn(str(f))
                if tpl is None or tpl.shape[0] < 40 or tpl.shape[1] < 40:
                    continue
                # 去重入库
                dup = any(
                    e.shape == tpl.shape and cv2.absdiff(e, tpl).mean() < 6.0
                    for e in self.tag_templates
                )
                if not dup:
                    self.tag_templates.append(tpl)
                    if tpl.ndim == 3:
                        self.tag_templates_gray.append(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))
                    else:
                        self.tag_templates_gray.append(tpl)
                    count += 1
            if len(self.tag_templates) > self.max_tag_templates:
                self.tag_templates = self.tag_templates[-self.max_tag_templates:]
                self.tag_templates_gray = self.tag_templates_gray[-self.max_tag_templates:]
            return count
        except Exception:
            return 0

    def _seed_tag(self, frame, player):
        """播种:截取玩家名字框文字模板(全图近似匹配用)+ 人物形象模板
        (小窗口兜底). 名字框文字是白字黑描边,特征独特,全图匹配不会像
        70x90 人物模板那样锁到岩石/木桩. If an offline nametag template is
        already loaded (mage-style), the OCR-seeded crop is only accepted if
        it resembles the offline one -- otherwise the OCR mis-read (e.g. a
        worn-badge decoration) would pollute the template and lock the box."""
        try:
            cx, cy = player["center"]
            img_h, img_w = frame.shape[:2]
            # 1) 名字框文字模板:优先用 OCR 的名字框坐标,否则从玩家中心
            #    几何推导(名字固定在玩家头顶上方).
            box = player.get("nametag_box")
            if box and len(box) == 4:
                nx, ny, nw, nh = (int(v) for v in box)
            else:
                nx = int(cx) - 20
                ny = int(cy) - self.name_bottom_offset - 14
                nw, nh = 40, 14
            if nw >= 16 and nh >= 8:
                x0 = max(0, nx)
                y0 = max(0, ny)
                x1 = min(img_w, nx + nw)
                y1 = min(img_h, ny + nh)
                tpl = frame[y0:y1, x0:x1]
                if tpl.size and tpl.shape[0] >= 8 and tpl.shape[1] >= 16:
                    # OCR template must resemble the offline one before we
                    # accept it; otherwise the mis-read crop would poison the
                    # matcher. (e.g. OCR picking a worn badge -> name_template
                    # becomes that badge -> matches the badge every frame.)
                    if (
                        self.name_template is None
                        or self._offline_name_path is None
                        or cv2.absdiff(tpl, self.name_template).mean()
                        < self._max_template_diff
                    ):
                        self.name_template = tpl.copy()
                        self.name_center = ((x0 + x1) // 2, (y0 + y1) // 2)
            # 2) 人物形象模板(兜底,70x90 去重入库)
            w2, h2 = self.box_width // 2, self.box_height // 2
            x0 = max(0, int(cx) - w2)
            y0 = max(0, int(cy) - h2)
            x1 = min(img_w, int(cx) + w2)
            y1 = min(img_h, int(cy) + h2)
            tpl = frame[y0:y1, x0:x1]
            if tpl.size == 0 or tpl.shape[0] < 40 or tpl.shape[1] < 40:
                return
            for existing in self.tag_templates:
                if (
                    existing.shape == tpl.shape
                    and cv2.absdiff(tpl, existing).mean() < 6.0
                ):
                    return
            self.tag_templates.append(tpl.copy())
            if tpl.ndim == 3:
                self.tag_templates_gray.append(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))
            else:
                self.tag_templates_gray.append(tpl)
            if len(self.tag_templates) > self.max_tag_templates:
                self.tag_templates.pop(0)  # 保留最新姿势,丢最旧
                self.tag_templates_gray.pop(0)
            if self.tag_center is None:
                self.tag_center = (int(cx), int(cy))
        except Exception:
            pass

    def _track_name(self, frame):
        """Original-style nametag SQDIFF match (mimics the upstream
        src/engine/MapleStoryAutoLevelUp.py:get_player_location_by_nametag):
        - BORDER_REPLICATE pad the search region
        - vertical split the template so partial occlusion still matches
        - green-screen mask (get_mask) so the name strip matches, not the
          wood/bridge backdrop of the 41x23 crop
        - prefer last-frame location as the search centre (cached match
          first), with a global fallback when the cached score is poor.
        The player sits BELOW the nametag (head-to-name geometric offset),
        not above (the upstream formula `loc_player = nametag_y - offset[1]`
        is signed backwards for this character)."""
        from src.utils.common import find_pattern_sqdiff, get_mask
        if self.name_template is None:
            return None
        img_h, img_w = frame.shape[:2]
        search = frame[: min(687, img_h)]
        th, tw = self.name_template.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        tpl_gray = self.name_template
        if tpl_gray.ndim == 3:
            tpl_gray = cv2.cvtColor(tpl_gray, cv2.COLOR_BGR2GRAY)
        # Green-screen mask for the template (treat green background as transparent).
        # Template might be BGR or gray.
        tpl_bgr = self.name_template
        mask = get_mask(tpl_bgr, (0, 255, 0)) if tpl_bgr.ndim == 3 else None
        # BORDER_REPLICATE padding
        pad_y, pad_x = th, tw
        img_roi = cv2.copyMakeBorder(
            gray, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )
        # Vertical split (upstream default split_width 20).
        num_splits = max(1, tw // 20)
        w_split = tw // num_splits
        best_loc, best_score, best_cached = None, float("inf"), False
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else tw
            split_img = tpl_gray[:, x_s:x_e]
            split_mask = mask[:, x_s:x_e] if mask is not None else None
            split_offset = x_s
            # Cached search centre is in full-frame coords, but find_pattern_sqdiff
            # receives a padded image (offset by pad_x/pad_y).
            cached = None
            if self._last_nametag_loc is not None and i == 0:
                cached = (self._last_nametag_loc[0] + split_offset + pad_x,
                          self._last_nametag_loc[1] + pad_y)
            try:
                loc, score, is_cached = find_pattern_sqdiff(
                    img_roi, split_img,
                    last_result=cached,
                    mask=split_mask,
                    global_threshold=0.3,
                )
            except cv2.error:
                return None
            if score < best_score:
                best_loc, best_score, best_cached = (loc, score, is_cached)
                best_offset = split_offset
        if best_score >= 0.3 or best_loc is None:
            return None
        # Restore full-frame coords (remove pad + split offset).
        nx = best_loc[0] - best_offset - pad_x
        ny = best_loc[1] - pad_y
        # Player sits BELOW the nametag (head geometry), fixed offset.
        player_x = nx + tw // 2
        player_y = ny + th + self.name_bottom_offset
        self.name_center = (player_x, ny + th // 2)
        self._last_nametag_loc = (nx, ny)
        return {
            "label": "PLAYER",
            "score": float(best_score),
            "box": (
                max(0, player_x - self.box_width // 2),
                max(0, player_y - self.box_height // 2),
                self.box_width,
                self.box_height,
            ),
            "nametag_box": (nx, ny, tw, th),
            "center": (player_x, player_y),
            "method": "name_track",
        }

    def load_offline_name_template(self, path):
        """Load a pre-recorded nametag template (mage-style). When present it
        is used as the primary fast matcher and OCR-seeded templates are only
        accepted if they closely resemble this one (so a mis-read OCR cannot
        pollute the template and lock the box to a rock)."""
        from src.utils.common import imread_cn
        try:
            tpl = imread_cn(str(path))
            if tpl is None or tpl.size == 0:
                return False
            self.name_template = tpl
            self._offline_name_path = str(path)
            # Always take the highest score (mage-style); the offline
            # template is exact, so a high threshold is safe.
            self.name_threshold = 0.80
            return True
        except Exception:
            return False

    def load_offline_title_template(self, path):
        """Load the title badge template (e.g. "新手冒险家勋章"). Used as a
        secondary anchor when the nametag strip is hidden by effects."""
        from src.utils.common import imread_cn, get_mask
        try:
            tpl = imread_cn(str(path))
            if tpl is None or tpl.size == 0:
                return False
            self.title_template = tpl
            self.title_size = tpl.shape[:2]  # (h, w)
            self.title_tpl_gray = (
                cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
            )
            # The badge has its own background colour; treat that as
            # transparent so the SQDIFF focuses on the badge text.
            self.title_mask = get_mask(tpl, (0, 255, 0)) if tpl.ndim == 3 else None
            return True
        except Exception:
            return False

    def _track_title(self, frame):
        """Original-style title-badge SQDIFF match (mirrors _track_name but
        for the wider title text strip below the nametag). Helps when the
        nametag is occluded by effects/dark backgrounds."""
        from src.utils.common import find_pattern_sqdiff
        if self.title_template is None:
            return None
        img_h, img_w = frame.shape[:2]
        search = frame[: min(687, img_h)]
        th, tw = self.title_size
        if search.shape[0] < th or search.shape[1] < tw:
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        pad_y, pad_x = th, tw
        img_roi = cv2.copyMakeBorder(
            gray, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )
        num_splits = max(1, tw // 20)
        w_split = tw // num_splits
        best_loc, best_score, best_offset = None, float("inf"), 0
        for i in range(num_splits):
            x_s = i * w_split
            x_e = (i + 1) * w_split if i < num_splits - 1 else tw
            split_img = self.title_tpl_gray[:, x_s:x_e]
            split_mask = self.title_mask[:, x_s:x_e] if self.title_mask is not None else None
            cached = None
            if self._last_title_loc is not None and i == 0:
                cached = (self._last_title_loc[0] + x_s + pad_x,
                          self._last_title_loc[1] + pad_y)
            try:
                loc, score, _ = find_pattern_sqdiff(
                    img_roi, split_img,
                    last_result=cached,
                    mask=split_mask,
                    global_threshold=0.3,
                )
            except cv2.error:
                return None
            if score < best_score:
                best_loc, best_score, best_offset = loc, score, x_s
        if best_score >= 0.3 or best_loc is None:
            return None
        nx = best_loc[0] - best_offset - pad_x
        ny = best_loc[1] - pad_y
        player_x = nx + tw // 2
        player_y = ny + th + self.title_bottom_offset
        self._last_title_loc = (nx, ny)
        return {
            "label": "PLAYER",
            "score": float(best_score),
            "box": (
                max(0, player_x - self.box_width // 2),
                max(0, player_y - self.box_height // 2),
                self.box_width,
                self.box_height,
            ),
            "nametag_box": (nx, ny, tw, th),
            "center": (player_x, player_y),
            "method": "title_track",
        }

    def _track_tag(self, frame):
        """FULL-FRAME multi-pose character-sprite match with cluster voting.

        Every sprite template is matched over the whole gameplay frame in
        grayscale at 0.5x (6 templates ~15ms). Each template's best location
        is a vote. The player is hit by MANY templates (multi-pose library),
        while a rock/cloud/background false positive is hit by only ONE or
        TWO. Votes are clustered (within 60px) and the cluster with the
        highest total score wins -> the player. This needs NO OCR seed:
        the box locks onto the character from the very first frame."""
        if not self.tag_templates:
            return None
        img_h, img_w = frame.shape[:2]
        th, tw = self.tag_templates[0].shape[:2]
        search = frame[: min(687, img_h)]
        if search.shape[0] < th or search.shape[1] < tw:
            return None
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        half = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # Build (or refresh) the 0.5x grayscale template list.
        lib = getattr(self, "tag_templates_gray_half", None)
        if lib is None or len(lib) != len(self.tag_templates):
            lib = [
                cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                for t in (self.tag_templates_gray or self.tag_templates)
            ]
            self.tag_templates_gray_half = lib
        hits = []  # (x2, y2, score)
        # Memory guard: on this machine the system often has <1GB free, and
        # matching ALL 7 templates per frame caused a 1.7MB allocation to
        # fail (crash). Match only the LAST-BEST template first (1 match, low
        # memory); if it does not reach the threshold, fall back to the full
        # library voting so the first frame still finds the player.
        lib_order = list(range(len(lib)))
        if getattr(self, "_last_best_tpl", None) is not None:
            lib_order = [self._last_best_tpl] + [i for i in range(len(lib)) if i != self._last_best_tpl]
        for ti in lib_order:
            tpl = lib[ti]
            if half.shape[0] < tpl.shape[0] or half.shape[1] < tpl.shape[1]:
                continue
            try:
                result = cv2.matchTemplate(half, tpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(result)
            except cv2.error:
                # Out of memory (system often has <1GB free on this machine):
                # skip this template instead of crashing the bot.
                continue
            if score < 0.45:
                continue
            # Position prior: the player is drawn in the LOWER half of the
            # gameplay area (camera follows from above); birds/bugs near
            # the sky score similarly on red/white templates but must NOT
            # win the vote -- reject any hit in the top sky band.
            y_full = int(loc[1]) * 2
            if y_full < 150:
                continue
            hits.append((int(loc[0]) * 2, y_full, float(score), ti))
        if len(hits) < 2:
            return None
        # Cluster votes: merge hits within 60px (Manhattan) of a cluster mean.
        clusters = []
        for hx, hy, hs, ti in hits:
            best_c = None
            for c in clusters:
                if abs(hx - c[0]) + abs(hy - c[1]) < 60:
                    best_c = c
                    break
            if best_c is None:
                clusters.append([hx, hy, [hs], 1])
            else:
                n = best_c[3] + 1
                best_c[0] = (best_c[0] * best_c[3] + hx) / n
                best_c[1] = (best_c[1] * best_c[3] + hy) / n
                best_c[2].append(hs)
                best_c[3] = n
        # Highest total score cluster = the player (multi-template hit).
        best_cluster = max(clusters, key=lambda c: sum(c[2]))
        cx, cy, scores, count = best_cluster
        if count < 2:
            return None
        # Score of the cluster = its best single template score; remember
        # which template produced it for next-frame priority matching.
        best_score = max(scores)
        self._last_best_tpl = hits[scores.index(best_score)][3]
        if best_score < self.tag_threshold:
            return None
        player_x = int(cx) + tw // 2
        player_y = int(cy) + th // 2
        # Anti-hijack: no teleport between frames.
        if self.tag_center is not None:
            ox, oy = self.tag_center
            if abs(player_x - ox) + abs(player_y - oy) > self.tag_step * 3:
                return None
        self.tag_center = (player_x, player_y)
        nx, ny = int(cx), int(cy)
        # Online learning: on a high-confidence match, the currently matched
        # sprite region is added to the library. Walking/attacking changes the
        # pose and background continuously; keeping fresh poses makes the next
        # frames match more reliably (this is why the mage's fixed nametag
        # template felt "fast" - it always matched). Deduplicate to avoid
        # bloating the library with near-identical frames.
        # Online learning: store new pose only when (1) the score is very high
        # (rock/wood false positives reach ~0.65 but never 0.85) AND (2) the
        # crop actually contains red pixels (player has the red wing/effect).
        # Without the colour check, all-rock templates get accumulated when
        # the player temporarily leaves the frame.
        if best_score >= 0.85:
            try:
                img_h, img_w = frame.shape[:2]
                w2, h2 = self.box_width // 2, self.box_height // 2
                sx0 = max(0, player_x - w2)
                sy0 = max(0, player_y - h2)
                sx1 = min(img_w, player_x + w2)
                sy1 = min(img_h, player_y + h2)
                tpl = frame[sy0:sy1, sx0:sx1]
                if tpl.shape == self.tag_templates[0].shape:
                    hsv = cv2.cvtColor(tpl, cv2.COLOR_BGR2HSV)
                    red = cv2.inRange(hsv, (0,120,120), (10,255,255)) | cv2.inRange(hsv, (170,120,120), (180,255,255))
                    red_pct = red.mean() / 255 * 100
                    if red_pct < 2.0:
                        # Crop has no player signature -- likely background.
                        pass
                    else:
                        dup = any(
                            e.shape == tpl.shape
                            and cv2.absdiff(e, tpl).mean() < 6.0
                            for e in self.tag_templates
                        )
                        if not dup:
                            self.tag_templates.append(tpl.copy())
                            if tpl.ndim == 3:
                                self.tag_templates_gray.append(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))
                            else:
                                self.tag_templates_gray.append(tpl)
                            if len(self.tag_templates) > self.max_tag_templates:
                                self.tag_templates.pop(0)
                                self.tag_templates_gray.pop(0)
            except Exception:
                pass
        return {
            "label": "PLAYER",
            "score": best_score,
            "box": (
                max(0, player_x - self.box_width // 2),
                max(0, player_y - self.box_height // 2),
                self.box_width,
                self.box_height,
            ),
            "nametag_box": (nx, ny, tw, th),
            "center": (player_x, player_y),
            "method": "tag_track",
        }

    def _shift_player_up(self, player, dy=None):
        """Shift the WHOLE player visual UP by dy px: both the player box and
        the attack-range box derive from player.center, so moving the center
        keeps them concentric (the old _shift_box_up only moved the box,
        which split the two yellow boxes apart). The nametag-derived center
        sits ~15px too low (feet-leaning), so a fixed small lift looks right.
        Idempotent: a "_shifted" marker prevents re-shifting the same dict
        (OCR reuses the same _ocr_result object across frames).
        dy=None uses self.shift_up_px (config-tunable)."""
        if player.get("_shifted"):
            return player
        if dy is None:
            dy = self.shift_up_px
        try:
            cx, cy = player["center"]
            cy2 = max(0, int(cy) - dy)
            player["center"] = (int(cx), cy2)
            bx, by, bw, bh = player["box"]
            player["box"] = (bx, max(0, by - dy), bw, bh)
            player["_shifted"] = True
        except Exception:
            pass
        self._attach_player_coords(player)
        return player

    def _attach_player_coords(self, player):
        """给玩家附加实体坐标(codex 风格): entity_id=P1 + 中心/速度/运动状态。"""
        player.setdefault("entity_id", "P1")
        cx, cy = player.get("center", (0, 0))
        player["center_px"] = [round(float(cx), 1), round(float(cy), 1)]
        now = time.time()
        if self._player_prev_center is not None:
            dt = max(now - self._player_prev_time, 1e-6)
            mx = (cx - self._player_prev_center[0]) / dt
            my = (cy - self._player_prev_center[1]) / dt
            self._player_velocity[0] = self._player_velocity[0] * 0.5 + mx * 0.5
            self._player_velocity[1] = self._player_velocity[1] * 0.5 + my * 0.5
        self._player_prev_center = (cx, cy)
        self._player_prev_time = now
        vx, vy = self._player_velocity
        speed = float(np.hypot(vx, vy))
        if abs(vy) >= 15.0 and abs(vy) >= abs(vx) * 0.75:
            ms = "UP" if vy < 0 else "DOWN"
        elif speed >= 20.0:
            ms = "MOVE"
        else:
            ms = "STILL"
        player["velocity_px_s"] = [round(vx, 1), round(vy, 1)]
        player["speed_px_s"] = round(speed, 1)
        player["motion_state"] = ms
        player["tracking_state"] = "DETECTED"

    def notify_move(self, command, now):
        """Called by the main loop with the command the character is about to
        execute. Records the horizontal movement direction so detect() can
        predict where the player should be."""
        if command in ("move_left", "attack_left"):
            self._move_dir, self._move_time = "left", now
        elif command in ("move_right", "attack_right"):
            self._move_dir, self._move_time = "right", now
        elif command in ("jump", "dodge_left", "dodge_right"):
            self._move_dir = None
            self._move_time = now
        else:
            self._move_dir = None
            self._move_time = now

    def _predict_center(self, now):
        """Expected player center: last seen center + move vector.
        Jumping only changes the Y coordinate by a small amount, walking only
        the X - so the prediction stays close to the last position, and when
        the visual match disagrees by a lot we simply trust the prediction."""
        if self.tag_center is None:
            return None
        if self._move_dir is None or self._move_time is None:
            return self.tag_center
        elapsed = now - self._move_time
        if elapsed < 0.0 or elapsed > 2.0:
            return self.tag_center
        dx = self._move_speed * elapsed
        if self._move_dir == "left":
            dx = -dx
        return (self.tag_center[0] + dx, self.tag_center[1])

    def _teleport_guard(self, center, now):
        """True if the match is too far from the last RELIABLE position
        (a teleport to a similar-looking patch of background). Uses the last
        accepted match, not a movement prediction - a prediction seeded from
        a false match would let the box drift to the sky."""
        if self.last_player is None:
            return False
        ox, oy = self.last_player["center"]
        dist = abs(center[0] - ox) + abs(center[1] - oy)
        return dist > self.max_track_error_px

    def detect(self, frame, now):
        # Teleport guard helper: reject a match that jumped too far from the
        # position predicted by the movement keys (a far-away false match).
        def ok(player):
            return player is not None and not self._teleport_guard(player["center"], now)
        # 1) Original-style nametag SQDIFF match (upstream
        #    get_player_location_by_nametag): cached local-search + global
        #    fallback + vertical split. The box stays locked onto the
        #    nametag once the player has been seen.
        tracked = self._track_name(frame)
        if ok(tracked):
            self.last_player = tracked
            self.last_seen = now
            return self._shift_player_up(tracked)
        # 1.5) Title badge SQDIFF (e.g. "新手冒险家勋章"): wider anchor that
        #      keeps the box locked when the nametag strip is occluded.
        tracked = self._track_title(frame)
        if ok(tracked):
            self.last_player = tracked
            self.last_seen = now
            return self._shift_player_up(tracked)
        # 2) Character-sprite SMALL-WINDOW match: ~1ms per frame.
        tracked = self._track_tag(frame)
        if ok(tracked):
            self.last_player = tracked
            self.last_seen = now
            return self._shift_player_up(tracked)
        # 3) Latest async OCR result: seeds templates and tag_center.
        if self.async_ocr:
            with self._ocr_lock:
                ocr_player = self._ocr_result
                ocr_stamp = self._ocr_stamp
            if ocr_player is not None and now - ocr_stamp <= self.ocr_max_age and ok(ocr_player):
                self._seed_tag(frame, ocr_player)
                self.last_player = ocr_player
                self.last_seen = now
                return self._shift_player_up(ocr_player)
        # 4) Reuse the last known position: when the visual match failed or
        #    was rejected by the guard, keep the box AT THE LAST RELIABLE
        #    POSITION (no movement prediction). A predicted position can
        #    drift if seeded from a false match, which made the box fly away
        #    to the sky when jumping / occluded. Sticking in place is always
        #    the safest answer.
        if self.last_player is not None and now - self.last_seen <= self.sticky_seconds:
            return self.last_player
        self.last_player = None
        return None


# --------------------------------------------------------------------------
# Color-verified monster detector: wraps the YOLO detector and drops
# detections whose dominant HSV color does not match the class. Maple monsters
# have very distinctive colors, so this cheaply removes most false positives
# (drops, ground textures) without retraining.
# --------------------------------------------------------------------------
class ColorVerifiedMonsterDetector:
    # Class label -> list of (lower_hsv, upper_hsv) acceptable color ranges.
    # Monster sprites are simple flat textures: a real box is dominated by one
    # bright color. Background wood/grass is darker, so value (V) separates
    # monsters from background materials.
    COLOR_RANGES = {
        "RED SNAIL": [
            ((0, 80, 140), (22, 255, 255)),    # bright red/orange (V>=140)
            ((160, 80, 140), (180, 255, 255)), # red hue wrap
        ],
        "BLUE SNAIL": [
            ((95, 80, 80), (135, 255, 255)),   # blue
        ],
        "STUMP": [
            ((8, 50, 50), (30, 255, 170)),     # wood brown (darker V<170)
        ],
        "SLIME": [
            ((55, 80, 80), (112, 255, 255)),   # green water slime
        ],
        "GREEN MUSHROOM": [
            ((30, 80, 80), (78, 255, 255)),    # green mushroom cap
        ],
        "FLOWER MUSHROOM": [
            ((150, 60, 80), (180, 255, 255)),  # pink/red cap
            ((0, 60, 80), (12, 255, 255)),
        ],
        "ORANGE MUSHROOM": [
            ((6, 80, 110), (26, 255, 255)),    # orange cap
        ],
    }
    MIN_COLOR_FRACTION = 0.25  # monster sprites are flat simple textures, so a real box is mostly the class color

    def __init__(self, detector):
        self.detector = detector

    @staticmethod
    def _box_color_fraction(hsv, box, ranges):
        x, y, w, h = (int(v) for v in box)
        if w < 6 or h < 6:
            return 0.0
        region = hsv[max(0, y):y + h, max(0, x):x + w]
        if region.size == 0:
            return 0.0
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(region, np.array(lower), np.array(upper))
        return float(np.count_nonzero(mask)) / float(region.shape[0] * region.shape[1])

    def detect(self, frame, player):
        detections = self.detector.detect(frame, player)
        if not detections:
            return detections
        hsv = cv2.cvtColor(frame[: self.detector.ui_y_start], cv2.COLOR_BGR2HSV)
        kept = []
        for detection in detections:
            label = detection["label"]
            ranges = self.COLOR_RANGES.get(label)
            if ranges is None:
                kept.append(detection)
                continue
            fraction = self._box_color_fraction(hsv, detection["box"], ranges)
            if fraction >= self.MIN_COLOR_FRACTION:
                kept.append(detection)
        return kept


# --------------------------------------------------------------------------
# Pure color + connected-component monster detector (optimized from the
# upstream `template_free` idea): find seed color pixels -> morphological
# close (gather surrounding) -> connected components -> area/ratio filter.
# No YOLO model, no templates - just HSV ranges per class, so CPU is tiny.
# --------------------------------------------------------------------------
class ColorMonsterDetector:
    """Detect monsters by class HSV color + connected-component analysis.

    Only searches a horizontal band around the player (level_band_half_h).
    Reuses the tuned COLOR_RANGES (same as the YOLO color-verify layer).
    """
    COLOR_RANGES = {
        # 精确紧凑范围(从用户给的白底/绿幕怪物图 kmeans 提取主色,
        # 非 YOLO 验证用的宽泛范围, 避免背景草/树/花误命中)
        "RED SNAIL": [
            ((0, 110, 120), (15, 255, 230)),
            ((165, 110, 120), (179, 255, 230)),
        ],
        "BLUE SNAIL": [((95, 80, 80), (135, 255, 255))],
        "STUMP": [((10, 60, 60), (28, 200, 160))],
        "SLIME": [((85, 45, 55), (120, 170, 130))],
        "GREEN MUSHROOM": [((30, 35, 145), (48, 90, 215))],
        "FLOWER MUSHROOM": [((12, 170, 150), (26, 255, 230))],
        "ORANGE MUSHROOM": [((6, 170, 150), (26, 255, 230))],
    }
    # 中文/别名 -> 标准大写 label(供 --monster-labels 过滤用)
    LABEL_ALIASES = {
        "树妖": "STUMP", "黑斧木妖": "STUMP",
        "红蜗牛": "RED SNAIL", "蓝蜗牛": "BLUE SNAIL",
        "绿水灵": "SLIME", "绿蘑菇": "GREEN MUSHROOM",
        "花蘑菇": "FLOWER MUSHROOM", "橙蘑菇": "ORANGE MUSHROOM",
    }

    def __init__(self, cfg, level_band_half_h=130, include_labels=None,
                 min_area=250, max_area=15000, min_ratio=0.35, max_ratio=2.8,
                 close_kernel=5, min_monster_y=100, player_box_pad=60,
                 nms_iou=0.30):
        self.ui_y_start = int(cfg["ui_coords"]["ui_y_start"])
        self.level_band_half_h = int(level_band_half_h)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.close_kernel = int(close_kernel)
        self.min_monster_y = int(min_monster_y)
        self.player_box_pad = int(player_box_pad)
        self.nms_iou = float(nms_iou)
        if include_labels:
            self.include_labels = set()
            for x in include_labels:
                key = str(x).strip().upper()
                self.include_labels.add(self.LABEL_ALIASES.get(key, key))
        else:
            self.include_labels = None
        # 加载怪物形状模板(从用户给的绿幕/白底图提取轮廓, 用于 matchShapes 验证)
        self.shape_templates = self._load_shape_templates()

    def _load_shape_templates(self):
        """从用户给的怪物图提取每类轮廓模板(去绿幕/白底后取最大轮廓)。"""
        from src.utils.common import imread_cn
        repo = Path(__file__).resolve().parents[1]
        files = {
            "STUMP": ["monster/树妖/black_axe_stump_1.png",
                      "monster/树妖/black_axe_stump_2.png"],
            "RED SNAIL": ["monster/红蜗牛/3174afc4-e187-45d7-a4b2-a2731ab1ca64.png"],
            "SLIME": ["monster/绿水灵/{66576801-2D73-474A-92BE-53F793B39FD0}.png"],
            "GREEN MUSHROOM": ["monster/绿蘑菇/{4C4130FD-87AC-455C-9E95-D5DE5C4A7D6D}.png"],
            "FLOWER MUSHROOM": ["monster/花蘑菇/ScreenShot_2026-08-12_234838_548.png"],
        }
        templates = {}
        for label, paths in files.items():
            conts = []
            for p in paths:
                try:
                    img = imread_cn(str(repo / p))
                except Exception:
                    continue
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, w = img.shape[:2]
                # 用图像边框颜色判断背景(绿幕图边框绿, 白底图边框白),
                # 不能看整图绿色占比 —— 绿水灵/绿蘑菇本身是绿色, 会被误判成绿幕
                border = np.concatenate([
                    hsv[0:3, :, :].reshape(-1, 3), hsv[h - 3:h, :, :].reshape(-1, 3),
                    hsv[:, 0:3, :].reshape(-1, 3), hsv[:, w - 3:w, :].reshape(-1, 3),
                ])
                green_ratio = ((border[:, 0] >= 40) & (border[:, 0] <= 90) &
                               (border[:, 1] > 80) & (border[:, 2] > 80)).mean()
                white_ratio = ((border[:, 1] < 30) & (border[:, 2] > 200)).mean()
                if green_ratio > white_ratio:
                    fg = cv2.bitwise_not(cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)))
                else:
                    # 白底: 抠高饱和度彩色像素(怪物), 白底低饱和自动排除
                    fg = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    conts.append(max(cs, key=cv2.contourArea))
            if conts:
                templates[label] = conts
        return templates

    def detect(self, frame, player=None):
        """Return [{label, box, score, method}] or []. Search only the player's
        horizontal band (level_band_half_h) to keep CPU low."""
        if frame is None:
            return []
        search = frame[: self.ui_y_start]
        h_full, w_full = search.shape[:2]
        band_y0, band_y1 = 0, h_full
        if player is not None and self.level_band_half_h > 0:
            try:
                bx, by, bw, bh = (int(v) for v in player["box"])
                cy = by + bh // 2
                band_y0 = max(0, cy - self.level_band_half_h)
                band_y1 = min(h_full, cy + self.level_band_half_h)
                if band_y1 <= band_y0:
                    return []
                search = search[band_y0:band_y1]
            except Exception:
                pass
        # 1) 黑色描边 -> CLOSE -> 连通域(怪物轮廓候选; 原仓库 template_free 思路)
        black = np.all(search < [30, 30, 30], axis=2).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k_close)
        num, _, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
        detections = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 800 or area > 20000:
                continue
            ratio = w / max(1.0, float(h))
            if ratio < 0.5 or ratio > 1.8:   # 怪物接近方形, 排除地形/绳子横条
                continue
            gy = int(y) + band_y0
            box = (int(x), gy, int(w), int(h))
            if gy < self.min_monster_y:
                continue
            if player is not None:
                px, py, pw, ph = (int(v) for v in player.get("box", (0, 0, 0, 0)))
                pbox = (px - self.player_box_pad, py - self.player_box_pad,
                        pw + 2 * self.player_box_pad, ph + 2 * self.player_box_pad)
                if self._iou(box, pbox) > 0.10:
                    continue
            crop = search[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            # 2) 内部颜色分类: 描边围成的区域内部有怪物主体色
            hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            best_label, best_pct = None, 0.0
            for label, ranges in self.COLOR_RANGES.items():
                if self.include_labels is not None and label not in self.include_labels:
                    continue
                m = np.zeros(hsv_c.shape[:2], dtype=np.uint8)
                for lower, upper in ranges:
                    m |= cv2.inRange(hsv_c, np.array(lower), np.array(upper))
                pct = m.mean() / 255.0
                if pct > best_pct:
                    best_pct, best_label = pct, label
            if best_label is None or best_pct < 0.12:
                continue
            # 3) 形状验证: 候选轮廓只和"颜色分类出的同类"模板比对(防止串类)
            ccrop = black[y:y + h, x:x + w]
            contours, _ = cv2.findContours(ccrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cand = max(contours, key=cv2.contourArea)
            tpls = self.shape_templates.get(best_label)
            if tpls:
                best_dist = 1e9
                for tpl in tpls:
                    try:
                        d = cv2.matchShapes(cand, tpl, cv2.CONTOURS_MATCH_I2, 0.0)
                    except cv2.error:
                        continue
                    if d < best_dist:
                        best_dist = d
                if best_dist > 0.6:
                    continue  # 形状不符, 拒绝
            detections.append({
                "label": best_label,
                "box": box,
                "score": best_pct,
                "method": "color",
            })
        return self._nms(detections)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, dets):
        if not dets:
            return dets
        dets = sorted(dets, key=lambda d: -d["score"])
        kept = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) < self.nms_iou for k in kept):
                kept.append(d)
        return kept


# --------------------------------------------------------------------------
# Original-style monster detector (src/engine/MapleStoryAutoLevelUp.py
# :get_monsters_in_range). Loads monster/<name>/*.png templates, runs CCOEFF
# NORMED + green-screen mask matching (very low false-positive rate, unlike
# SQDIFF_NORMED which produces scores near zero for almost any patch). No
# YOLO model -> minimal memory footprint (fixes the 1.7MB alloc crashes).
# --------------------------------------------------------------------------
class TemplateMonsterDetector:
    """Detect monsters via green-screen template matching (upstream approach).

    Optimized for low CPU: only searches a horizontal band around the player
    (level_band_half_h) instead of the full frame, and optionally loads only
    a whitelist of monster classes (include_labels) so the map's few monsters
    are matched instead of all 30+ mobs in monster/.
    """

    def __init__(self, cfg, monster_dir="monster", max_poses_per_mob=6,
                 level_band_half_h=0, include_labels=None):
        import glob as _glob
        self.search_scale = 0.5  # downscale ROI + templates (lighter/faster than 0.25x, more reliable scores)
        self.score_threshold = 0.82  # CCOEFF_NORMED: 1.0 = perfect, 0.82+ avoids background false positives
        self.nms_iou = 0.30
        # Skip regions occupied by the player (reduces self-detection +
        # also covers nearby allies/NPCs).
        self.player_box_pad = 60
        # Monster spawn band: ignore detections in the top HUD/sky area
        # (mouse cursor, mini-map, status icons have similar texture to some
        # monster poses and get false-matched otherwise).
        self.min_monster_y = 100
        # Horizontal band half-height around the player center to search in
        # (0 = full frame). Restricting it drastically cuts CPU: 98 templates
        # over the full 1296x687 frame takes ~2.1s/frame; over a ±110px band
        # it drops to ~150ms/frame.
        self.level_band_half_h = int(level_band_half_h or 0)
        # Optional whitelist of monster class names to load (e.g. the few
        # mobs that spawn on this map). None = load everything.
        self.include_labels = set(include_labels or []) or None
        self.templates = []  # [(label, gray, mask, h, w), ...]
        self.max_poses_per_mob = max_poses_per_mob
        loaded_per_label = {}
        for f in sorted(_glob.glob(f"{monster_dir}/*/*.png")):
            import os as _os
            label = _os.path.basename(_os.path.dirname(f))
            if self.include_labels is not None and label not in self.include_labels:
                continue
            if loaded_per_label.get(label, 0) >= self.max_poses_per_mob:
                continue
            img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            green = np.all(img == [0, 255, 0], axis=2)
            if green.all():
                continue
            mask = (~green).astype(np.uint8) * 255  # non-green pixels are the monster body
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            th, tw = mask.shape[:2]
            gray_s = cv2.resize(gray, None, fx=self.search_scale, fy=self.search_scale,
                                interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mask, None, fx=self.search_scale, fy=self.search_scale,
                                interpolation=cv2.INTER_AREA)
            _, mask_s = cv2.threshold(mask_s, 128, 255, cv2.THRESH_BINARY)
            self.templates.append((label, gray_s, mask_s, th, tw))
            loaded_per_label[label] = loaded_per_label.get(label, 0) + 1
        logger.info(f"[TemplateMonsterDetector] loaded {len(self.templates)} poses from {len(loaded_per_label)} mobs")

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, dets):
        if not dets:
            return dets
        dets = sorted(dets, key=lambda d: -d["score"])
        kept = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) < self.nms_iou for k in kept):
                kept.append(d)
        return kept

    def detect(self, frame, player=None):
        """Return [{label, box, score, method}] or [].

        When player is known and level_band_half_h > 0, only the horizontal
        band around the player center is searched (same layer only), which
        cuts CPU massively vs. scanning the full frame.
        """
        if not self.templates or frame is None:
            return []
        search = frame[:687]
        h_full, w_full = search.shape[:2]
        # Restrict search to a horizontal band around the player (if set).
        band_y0, band_y1 = 0, h_full
        if player is not None and self.level_band_half_h > 0:
            try:
                bx, by, bw, bh = (int(v) for v in player["box"])
                cy = by + bh // 2
                band_y0 = max(0, cy - self.level_band_half_h)
                band_y1 = min(h_full, cy + self.level_band_half_h)
                if band_y1 <= band_y0:
                    return []
                search = search[band_y0:band_y1]
            except Exception:
                pass
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.resize(gray, None, fx=self.search_scale, fy=self.search_scale,
                            interpolation=cv2.INTER_AREA)
        inv = int(round(1.0 / self.search_scale))
        detections = []
        for label, tpl_gray, tpl_mask, th, tw in self.templates:
            th_s, tw_s = tpl_gray.shape[:2]
            if gray_s.shape[0] < th_s or gray_s.shape[1] < tw_s:
                continue
            try:
                res = cv2.matchTemplate(gray_s, tpl_gray, cv2.TM_CCOEFF_NORMED, mask=tpl_mask)
            except cv2.error:
                continue  # out of memory: skip template
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # Guard against NaN/Inf in mask-matched CCOEFF (degenerate templates).
            if not np.isfinite(max_val):
                continue
            if max_val < self.score_threshold:
                continue
            # Restore coordinates: ROI/templates are scaled by search_scale,
            # so multiply back by 1/search_scale (0.5 -> x2). Add band offset.
            x = int(max_loc[0]) * inv
            y = int(max_loc[1]) * inv + band_y0
            box = (x, y, tw, th)
            # Reject detections in the top HUD/sky band (mouse cursor, UI
            # icons produce false positives that pass the score threshold).
            if y < self.min_monster_y:
                continue
            # Exclude detections inside the player region (no self-hit).
            if player is not None:
                px, py, pw, ph = player.get("box", (0, 0, 0, 0))
                pbox = (px - self.player_box_pad, py - self.player_box_pad,
                        pw + 2 * self.player_box_pad, ph + 2 * self.player_box_pad)
                if self._iou(box, pbox) > 0.10:
                    continue
            detections.append({
                "label": label,
                "box": box,
                "score": float(max_val),
                "method": "template",
            })
        return self._nms(detections)


# --------------------------------------------------------------------------
# Minimap navigator: read the top-left minimap (the player is a small yellow
# arrow/dot, water is blue, rock is brown, ground is yellow, pits are dark)
# and turn it into navigation commands (move, climb, jump). This is much more
# reliable than trying to recognize ropes/ledges in the game scene itself.
# --------------------------------------------------------------------------
class MinimapNavigator:
    """Parse the minimap and decide where to go.

    The minimap is a fixed region in the top-left of the game window. We
    detect the player marker by color (configurable, default yellow), then
    inspect the terrain colors around the player to decide whether to walk,
    jump a gap, or climb a rope.
    """

    # Semantic classes by HSV range (OpenCV H:0-179, S:0-255, V:0-255).
    TERRAIN_CLASSES = {
        # dark pit / void: cannot stand there -> jump over it
        "pit": ((0, 0, 0), (179, 255, 60)),
        # water: blue, cannot walk -> jump or avoid
        "water": ((80, 60, 60), (140, 255, 255)),
        # rock/ledge: brown, cannot walk (walls) but often climbable
        "rock": ((5, 40, 40), (25, 255, 160)),
        # ground: bright yellow-ish, walkable
        "ground": ((15, 70, 120), (45, 255, 255)),
    }

    def __init__(self, cfg, rng=None):
        mm_cfg = cfg.get("minimap", {})
        region = mm_cfg.get("region", [0, 0, 285, 245])
        self.region = (int(region[0]), int(region[1]),
                       int(region[2]), int(region[3]))
        # Player marker color, BGR (as stored in config) -> search in HSV.
        player_bgr = tuple(int(c) for c in mm_cfg.get(
            "player_color", (0, 255, 255)))
        tol = mm_cfg.get("player_color_tolerance", 80)
        if isinstance(tol, (list, tuple)):
            tol = int(tol[0])  # the first channel tolerance is the color one
        self.player_hsv_low, self.player_hsv_high = self._bgr_to_hsv_range(
            player_bgr, tol=int(tol))
        self.player_min_pixels = int(mm_cfg.get("player_min_pixels", 4))
        self.player_max_pixels = int(mm_cfg.get("player_max_pixels", 80))
        self.lookahead_px = int(mm_cfg.get("lookahead_px", 12))
        self.step_px = int(mm_cfg.get("step_px", 6))
        # How far up to scan for a rope channel above the player.
        self.rope_look_px = int(mm_cfg.get("rope_look_px", 40))
        self.rng = rng if rng is not None else random.Random()
        self._cache = None
        self._cache_at = 0.0
        self._cache_ttl = 0.3
        # Sticky tracking of the player marker: minimaps often have several
        # yellow blobs (other players / UI hints); we remember the last
        # confirmed marker and prefer the candidate nearest to it.
        self._last_player_center = None
        self._best_dist = float("inf")

    @staticmethod
    def _bgr_to_hsv_range(bgr, tol=80):
        """Approximate HSV range around a BGR color (used for player dot).

        The minimap player arrow is saturated yellow (BGR 0,255,255) but may
        be drawn slightly dimmer; we search a fairly tight hue band around
        yellow and a wide saturation/value band so the marker is found even
        when the arrow is small or anti-aliased. Hue tolerance stays small so
        we do not swallow the (also yellow) ground.
        """
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        pixel[0, 0] = bgr
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        h_tol = min(18, max(6, tol // 4))
        s_tol = min(105, tol + 30)
        v_tol = min(105, tol + 30)
        low = (max(0, h - h_tol), max(0, s - s_tol), max(0, v - v_tol))
        high = (min(179, h + h_tol), min(255, s + s_tol), min(255, v + v_tol))
        return low, high

    def scan(self, frame, now):
        """Return parsed minimap: player pos + terrain classes, cached."""
        if self._cache is not None and now - self._cache_at < self._cache_ttl:
            return self._cache
        x0, y0, w, h = self.region
        mm = frame[y0:y0 + h, x0:x0 + w]
        if mm.size == 0:
            return {"player": None, "player_xy": None, "terrain": np.zeros((1, 1), dtype=np.uint8)}

        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)

        # 2) Terrain classes: one label per pixel. Compute FIRST so the player
        # marker can be filtered by the terrain it sits on.
        terrain = np.zeros(mm.shape[:2], dtype=np.uint8)
        class_ids = list(self.TERRAIN_CLASSES)
        for label, (low, high) in self.TERRAIN_CLASSES.items():
            m = cv2.inRange(hsv, np.array(low), np.array(high))
            terrain[m > 0] = class_ids.index(label) + 1

        # 1) Player marker: find connected components in the player color.
        # Prefer candidates that sit on walkable terrain (ground or rock) and
        # are in the play area of the minimap (not the top/bottom UI bars).
        # Among valid candidates, prefer the one closest to the last known
        # position (sticky tracking): minimaps often have several yellow
        # markers (other players, UI hints) and picking the largest area
        # flickers between them as the player moves.
        player_xy = None
        mask = cv2.inRange(hsv, np.array(self.player_hsv_low),
                           np.array(self.player_hsv_high))
        # 玩家标记是高饱和亮黄菱形(S>=200); 地面/图标是暗黄(S<200), 收紧饱和度过滤掉它们
        mask &= cv2.inRange(hsv, (0, 200, 0), (179, 255, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best_area, best_center = 0, None
        last = self._last_player_center
        self._best_dist = float("inf")  # 每次 scan 重置: 否则 sticky 最近距离阈值跨帧累积, 玩家移动后候选全被拒绝, 坐标卡死
        y_play_top = int(h * 0.18)
        y_play_bot = int(h * 0.88)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self.player_min_pixels <= area <= self.player_max_pixels):
                continue
            cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
            cy = int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)
            # Skip markers in the top UI bar (character name / menu text).
            if not (y_play_top <= cy <= y_play_bot):
                continue
            # The player stands on walkable terrain; reject markers floating
            # over water or the void.
            on_ground = (
                0 <= cy < terrain.shape[0] and 0 <= cx < terrain.shape[1]
                and terrain[cy, cx] in (4, 3)
            )
            if not on_ground:
                continue
            # Sticky: if we have a recent position, strongly prefer the
            # nearest candidate (movement between frames is small).
            if last is not None:
                dist_to_last = float(np.hypot(cx - last[0], cy - last[1]))
                if best_center is None or dist_to_last < self._best_dist:
                    best_area, best_center = area, (cx, cy)
                    self._best_dist = dist_to_last
            elif area > best_area:
                best_area, best_center = area, (cx, cy)
        if best_center is not None:
            player_xy = (best_center[0] + x0, best_center[1] + y0)
            self._last_player_center = best_center

        result = {
            "player": player_xy,
            "player_xy": player_xy,
            "terrain": terrain,
            "region": self.region,
            "_minimap_img": mm,   # used by RouteFollower for template matching
        }
        self._cache = result
        self._cache_at = now
        return result

    def navigate(self, scan_result, player_xy, now=None):
        """Decide a navigation command from the minimap around the player.

        The minimap navigator only CORRECTS the patrol direction when there
        is an obstacle: a rope shaft to climb, a pit/water gap to jump, or
        ground only on one side (never walk into the void). When ground is
        walkable in both directions it returns None and the normal patrol
        logic (which already alternates direction randomly) takes over. This
        keeps the character from walking endlessly in one direction.

        Returns (command, reason) or None if nothing actionable.
        """
        if player_xy is None:
            return None
        x0, y0, w, h = self.region
        px = player_xy[0] - x0
        py = player_xy[1] - y0
        terrain = scan_result["terrain"]
        look = self.lookahead_px
        step = self.step_px

        # Gather terrain samples at several distances ahead (left/right).
        # 0 = unknown, 1 = pit, 2 = water, 3 = rock, 4 = ground.
        def sample(dx, dy):
            xx = px + dx
            yy = py + dy
            if 0 <= yy < terrain.shape[0] and 0 <= xx < terrain.shape[1]:
                return int(terrain[yy, xx])
            return 0

        right = [sample(d, 0) for d in (step, step * 2, look)]
        left = [sample(-d, 0) for d in (step, step * 2, look)]

        # Prefer the direction with walkable ground ahead.
        def ground_score(samples):
            return sum(1 for s in samples if s == 4)

        right_score = ground_score(right)
        left_score = ground_score(left)

        # Rope climb: scan upward from just above the player. If there is a
        # run of non-ground (the rope shaft) and then ground again (the top
        # platform), the player should climb.
        def rope_above():
            dy = step
            saw_gap = False
            while dy <= self.rope_look_px:
                t = sample(0, -dy)
                if t == 0:            # hit minimap edge before ground
                    return False
                if t == 4:            # reached a platform above
                    return saw_gap
                saw_gap = True
                dy += step
            return False

        if rope_above():
            return "climb_up", "minimap_rope"

        # Jump a gap: ground stops and resumes across a pit/water run.
        def gap_ahead(samples):
            # [near, mid, look]: near/mid non-ground, ground resumes at look.
            return samples[0] in (1, 2) and samples[-1] == 4

        if gap_ahead(right):
            return "jump_right", "minimap_gap"
        if gap_ahead(left):
            return "jump_left", "minimap_gap"

        # No walkable ground in either direction: nothing actionable.
        if right_score == 0 and left_score == 0:
            return None  # let the normal patrol handle it

        # Ground on only one side: walk toward it (never into the void).
        if right_score == 0:
            return "move_left", "minimap_no_ground_right"
        if left_score == 0:
            return "move_right", "minimap_no_ground_left"

        # Ground on both sides: leave direction choice to the normal patrol
        # (it already alternates), so the character does not walk endlessly
        # in one direction.
        return None

    def reset(self):
        self._cache = None
        self._cache_at = 0.0


# --------------------------------------------------------------------------
# Route follower: follow a per-map pre-recorded route (minimaps/{map}/route*.
# png, drawn with color-coded commands) by locating the player on the minimap
# and executing the nearest color-code command. This is how climbing ropes and
# jumping platforms works reliably across very different maps: the route is
# recorded once per map by the user (tools/routeRecorder.py), then replayed.
# --------------------------------------------------------------------------
class RouteFollower:
    """Follow a recorded route for the current map.

    The route images are color-coded: each pixel color maps to a command
    ("left none none" = walk left, "none up none" = climb, etc.). Each frame
    we locate the player on the minimap, translate to the global route map,
    and execute the nearest color-code pixel around the player.
    """

    # BGR colors from the route color_code config (parsed once).
    # Config keys are written as "R,G,B" (e.g. "255,0,0" = red); route images
    # store pixels in BGR, so keys are converted to BGR at load time.
    def __init__(self, cfg, map_name=None, rng=None):
        self.cfg = cfg
        route_cfg = cfg["route"]

        def _to_bgr(rgb_key):
            r, g, b = (int(c) for c in rgb_key.split(","))
            return (b, g, r)

        self.color_code = {
            _to_bgr(key): value
            for key, value in route_cfg["color_code"].items()
        }
        self.color_code.update({
            _to_bgr(key): value
            for key, value in route_cfg.get("color_code_up_down", {}).items()
        })
        # Search radius for the nearest route color around the player. The
        # minimap->global-map match has a few px of jitter; a small radius
        # (10) made the bot lose the route ("not following the recorded
        # path"). 40 px tolerates the jitter while staying unambiguous.
        self.search_range = int(route_cfg.get("search_range", 40))
        self.rng = rng if rng is not None else random.Random()
        self.img_map = None            # global map (BGR)
        self.img_routes = []           # route images (BGR), colors masked out
        self.idx_route = 0
        self.map_name = map_name
        self._loc_minimap_global = None
        self._minimap_offset = None    # (x0, y0) of the minimap in the frame
        self.last_player_minimap = None
        # Player's walked trail in global-map coordinates (for visualization).
        self.trail = []                # list of (gx, gy)
        self.trail_max = 2000
        # Single-route design: only the LATEST recorded route is kept (saved
        # as route1.png, new recordings overwrite it). No multi-route
        # switching, no goal-hopping.
        #
        # Return-to-route: when the player is not on the route (outside the
        # search range), find the NEAREST route pixel on the whole map and
        # walk horizontally toward it. If the vertical gap is too large (the
        # route is on another platform), stay put instead of blind-jumping
        # (saves resources; climbing is handled by the terrain scanner).
        self.vertical_gap_max = int(route_cfg.get("return_vertical_gap", 60))
        self._route_points = None      # np array (N,2) of non-black route px
        self._route_points_ready = False
        self._last_return_cmd = None   # throttle: avoid re-deciding every frame

    def load_map_routes(self, map_name=None):
        """Load minimaps/{map_name}/map.png + all route*.png (multi-route).
        Routes are sorted newest-first; the bot follows the latest by default.
        """
        if map_name:
            self.map_name = map_name
        if not self.map_name:
            return False
        base = Path("minimaps") / self.map_name
        map_file = base / "map.png"
        if not map_file.exists():
            return False
        self.img_map = imread_cn(str(map_file), cv2.IMREAD_COLOR)
        if self.img_map is None:
            return False
        self.img_routes = []
        self.route_files = []
        route_files = [p for p in base.glob("route*.png") if "rest" not in p.name]
        route_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for rf in route_files:
            route = imread_cn(str(rf), cv2.IMREAD_COLOR)
            if route is not None:
                route = self._mask_route_colors(route)
                if np.any(route != 0):
                    self.img_routes.append(route)
                    self.route_files.append(rf.name)
        self.idx_route = 0
        self._route_points_ready = False
        self._route_points = None
        return len(self.img_routes) > 0

    def _get_route_points(self):
        """Return np (N,2) array of non-black route pixels (cached)."""
        if self._route_points_ready:
            return self._route_points
        self._route_points_ready = True
        self._route_points = None
        if not self.img_routes:
            return None
        route = self.img_routes[0]
        ys, xs = np.nonzero(np.any(route != 0, axis=2))
        if len(xs) == 0:
            return None
        self._route_points = np.column_stack((xs, ys))
        return self._route_points

    def _mask_route_colors(self, route_img):
        """Black out pixels that are NOT a color code on the route image."""
        h, w = route_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for color in self.color_code:
            # Use int arithmetic: uint8 subtraction underflows (0-6 -> 250),
            # which would make lo > hi and inRange never match.
            lo = np.clip(np.array(color, dtype=np.int16) - 6, 0, 255).astype(np.uint8)
            hi = np.clip(np.array(color, dtype=np.int16) + 6, 0, 255).astype(np.uint8)
            mask |= cv2.inRange(route_img, lo, hi)
        masked = route_img.copy()
        masked[mask == 0] = 0
        return masked

    def set_minimap_offset(self, offset):
        self._minimap_offset = offset

    def locate_player(self, scan_result, player_xy):
        """Map the minimap player position onto the global route map.

        scan_result: MinimapNavigator.scan() output (has player_xy + region +
        the minimap image used for template matching).
        Returns the global player coordinate (gx, gy) or None.
        """
        if player_xy is None or self.img_map is None:
            return None
        minimap_img = scan_result.get("_minimap_img")
        if minimap_img is None:
            return None
        region = scan_result["region"]
        x0, y0, _, _ = region
        # Locate the minimap within the global map (template match).
        try:
            loc, score, _ = find_pattern_sqdiff(self.img_map, minimap_img)
        except Exception:
            return None
        if loc is None:
            return None
        self._loc_minimap_global = loc
        gx = loc[0] + (player_xy[0] - x0)
        gy = loc[1] + (player_xy[1] - y0)
        return (gx, gy)

    def decide(self, global_xy, now=None):
        """Find the nearest color-code command around the player and return
        (command, reason) or None.

        Command is in the auto_combat format: move_left/right, climb_up,
        climb_down, jump_left/right, jump, none.

        If the player is OFF the route (no color-code pixel within
        search_range), find the nearest route pixel on the whole map and walk
        HORIZONTALLY toward it. If the vertical gap to the route is too large
        (different platform), stay put ("return_wait") — climbing is left to
        the terrain scanner, and blind-jumping across layers wastes resources.
        """
        if self.img_routes is None or not self.img_routes or global_xy is None:
            return None
        route = self.img_routes[self.idx_route]
        h, w = route.shape[:2]
        gx, gy = int(global_xy[0]), int(global_xy[1])
        x_min = max(0, gx - self.search_range)
        x_max = min(w, gx + self.search_range + 1)  # inclusive
        y_min = max(0, gy - self.search_range)
        y_max = min(h, gy + self.search_range + 1)  # inclusive
        best = None
        best_dist = float("inf")
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                bgr = tuple(int(c) for c in route[y, x])
                if bgr == (0, 0, 0):
                    continue
                cmd = self.color_code.get(bgr)
                if cmd is None:
                    continue
                dist = abs(x - gx) + abs(y - gy)
                if dist < best_dist:
                    best_dist = dist
                    best = cmd
        if best is None:
            # Player is off the route: walk toward the nearest route pixel.
            return self._return_to_route(gx, gy)
        move_x, move_y, action = best.split()
        command = self._to_command(move_x, move_y, action)
        if command is None:
            return None
        return command, f"route_{best.replace(' ', '_')}"

    def _return_to_route(self, gx, gy):
        """Off-route: find nearest route pixel and move horizontally to it.

        Returns (command, reason) or None. When the vertical gap is too big
        (route on another platform), returns ("none", "return_wait") so the
        bot stays put instead of hopping blindly.
        """
        pts = self._get_route_points()
        if pts is None or len(pts) == 0:
            return None
        # Nearest route pixel (L1 distance, cheap full-map scan).
        d = np.abs(pts[:, 0] - gx) + np.abs(pts[:, 1] - gy)
        i = int(np.argmin(d))
        tx, ty = int(pts[i, 0]), int(pts[i, 1])
        dx, dy = tx - gx, ty - gy
        if abs(dy) > self.vertical_gap_max:
            return "none", "return_wait"
        if abs(dx) <= 3:
            return "none", "return_align"
        if dx < 0:
            return "move_left", "return_route"
        return "move_right", "return_route"

    def _to_command(self, move_x, move_y, action):
        if action == "goal":
            return None  # handled above
        if action == "teleport":
            action = "jump"  # no teleport key on this client
        # climb takes priority
        if move_y == "up":
            return "climb_up"
        if move_y == "down":
            return "climb_down"
        if action == "jump":
            if move_x == "left":
                return "jump_left"
            if move_x == "right":
                return "jump_right"
            return "jump"
        if action == "stop":
            return "none"
        if move_x == "left":
            return "move_left"
        if move_x == "right":
            return "move_right"
        return None

    def reset(self):
        self._loc_minimap_global = None
        self.last_player_minimap = None


# --------------------------------------------------------------------------
# 小地图坐标航点巡逻(MinimapWaypointPatrol)
# 用 codex 小地图工具返回的 稳定归一化坐标 map_norm 精确控制角色路线,
# 不依赖全局小地图图 + 模板匹配(那套会抖动), 位置绝对且稳定。
# 录制: 用户手动操控时, 每帧记录 (map_norm_x, map_norm_y, 动作) 航点序列;
# 回放: 按 map_norm 逐航点导航, 到达后执行该航点动作(跳跃/爬绳), 循环。
# 适合有大量跳跃/爬绳点的地图(如野猪领土2)。
# --------------------------------------------------------------------------
class MinimapWaypointPatrol:
    """Precise minimap-coordinate waypoint patrol (records + replays)."""

    # move: 手动打点的普通走位点(方向由 dx 决定, 到达需 dx+dy 双查)
    ACTION_MOVE = ("move_left", "move_right", "move")
    # jump_takeoff: 手动打的跳跃点(起跳点)——走到该点起跳, 落地即推进(不查落点)
    ACTION_JUMP = ("jump", "jump_climb", "jump_left", "jump_right", "jump_down",
                   "jump_takeoff")
    ACTION_CLIMB = ("climb_up", "climb_down")
    ACTION_ALL = ACTION_MOVE + ACTION_JUMP + ACTION_CLIMB

    def __init__(self, cfg, map_name=None):
        self.cfg = cfg
        wp = cfg.get("minimap_waypoint", {})
        self.min_spacing = float(wp.get("min_spacing", 0.02))
        # 普通点到达阈值: 至少小数点后4位一致(0.00005 ≈ 四舍五入4位后相等),
        # 必须完全碰到点位才算到达, 不允许提前停
        self.arrive_x = float(wp.get("arrive_x", 0.00005))
        self.arrive_y = float(wp.get("arrive_y", 0.00005))
        # 【普通点位(move)到达放宽】: 不需精确到点, 接近即算到达(跳跃点位仍
        # 保持 jump_align_x 精确对齐)。默认 0.01≈小地图1px。
        self.move_arrive_tol = float(wp.get("move_arrive_tol", 0.01))
        self.action_hold = float(wp.get("action_hold", 0.35))
        self.climb_timeout = float(wp.get("climb_timeout", 8.0))
        self.retry_timeout = float(wp.get("retry_timeout", 40.0))
        self.rest_every_rounds = int(wp.get("rest_every_rounds", 5))  # 每N圈回初始点坐椅子休息
        self.rest_sit_key = str(wp.get("rest_sit_key", "x"))          # 坐椅子按键
        self.rest_hold_seconds = float(wp.get("rest_hold_seconds", 0.5))  # 坐椅子按键时长
        self.rest_sit_delay = float(wp.get("rest_sit_delay", 3.0))    # 到达后等待秒数再按X(刚停稳按坐不上)
        # 同平台Y容差: 跳台阶失败/掉坑后恢复点必须与当前Y几乎一致(同平台,
        # 左右走就能到)。norm 抖动约 ±0.016, 平台间Y差通常 >0.05, 取 0.03
        # 覆盖抖动同时排除相邻平台点(否则恢复点到上面平台会死循环跳)。
        self.recover_y_tol = float(wp.get("recover_y_tol", 0.03))
        # 小步步进冷却间隔: 每次 step 后等这么久再允许下一步, 让小地图坐标检测
        # 有时间更新——否则每帧都 step(12fps≈83ms/步), 坐标还没反应又挪下一步,
        # 看起来碎步很多且容易在台阶边缘踏空掉下去
        self.step_interval = float(wp.get("step_interval", 0.35))
        self.max_waypoints = int(wp.get("max_waypoints", 3000))
        # 跳跃点起跳前的水平精确对齐容差(map_norm): 用户要求误差<=0.003,
        # 否则拿不到绳子; 严格对齐无锁定容忍, 偏差>此值一律重新走位
        self.jump_align_x = float(wp.get("jump_align_x", 0.003))
        # 抓绳跳(jump_climb)的对齐容差: 按着上跳过去能抓住绳, 不必精确到点(放宽)
        self.jump_climb_align_x = float(wp.get("jump_climb_align_x", 0.03))
        # 跳跃落点判定容差(放宽): 跳跃有自然偏差, 跳到平台附近即算成功,
        # 避免"跳上去了但差一点"被误判失误而反复从头(死循环)
        self.jump_arrive_x = float(wp.get("jump_arrive_x", 0.05))
        self.jump_arrive_y = float(wp.get("jump_arrive_y", 0.09))
        self.retry_max = int(wp.get("retry_max", 6))
        # 跳跃点起跳判定: 下一目标相对跳点的 Y 差 >= 此值 -> 跳+按上爬绳;
        # < 此值 -> 跳台阶(按 X 方向左跳/右跳/立定跳); <= -此值 -> 跳下
        self.climb_jump_dy = float(wp.get("climb_jump_dy", 0.05))
        # 跳台阶落点到达容差(目标点): 跳台阶失误(掉下去)则同 Y 恢复
        self.step_arrive_x = float(wp.get("step_arrive_x", 0.05))
        self.step_arrive_y = float(wp.get("step_arrive_y", 0.04))
        self.map_name = map_name
        # 动作段序列(纯录制回放): [{"action","nx","ny"}, ...]
        # action: move(普通点, 左右走) / jump_takeoff(跳跃点, 起跳位置)
        # nx/ny = 该段对应的点位坐标(move=走位目标, jump_takeoff=起跳位置)
        self.waypoints = []
        self.idx = 0             # 当前动作段索引
        self._fwd = True         # 往返方向: True=起点→终点, False=终点→起点
        self.is_recording = False
        self._patrolling = False  # 是否处于巡航模式(由 F3 开启, F4 关闭)
        # ---- 休息机制: 每 N 圈回初始点坐椅子回满再继续 ----
        self._round_count = 0     # 已完成的完整圈数(走完最后一点回到点1 +1)
        self._resting = False     # 是否处于休息状态(导航回点1/坐椅子/等待回满)
        self._rest_state = ""     # 休息子状态: "goto" 导航中 / "sit" 坐椅子 / "wait" 等待回满
        # ---- 录制状态(段式: 按键事件驱动) ----
        self._rec_prev_keys = set()   # 上一帧操作键集合
        self._rec_seg = ""            # 当前录制段: "" / move_* / climb_* / jump 系列(等待落地)
        self._rec_land_frames = 0     # jump 段落地稳定帧计数
        self._rec_land_pos = None     # jump 段落地检测位置
        self._rec_jump_start = 0.0    # jump 段开始时刻(落地超时保护)
        self._rec_jump_climb = False  # 当前 jump 段是否"抓绳跳"(跳起后按了上)
        # ---- 回放状态(逐段执行 + 失误从头) ----
        self._point_start = 0.0       # 当前段开始尝试时刻
        self._retry_count = 0         # 当前段重试次数(跳跃失败先重试, 超限从头)
        self._action_state = ""       # 当前动作执行状态
        self._action_at = 0.0         # 当前动作开始时刻
        self._climb_last_pos = None   # 爬绳期间位置(卡住判定)
        self._climb_last_ts = 0.0
        self._action_done = False     # jump 动作已执行完(等待落地)
        self._jump_land_frames = 0    # 回放: 跳跃落地稳定帧计数
        self._jump_land_pos = None    # 回放: 跳跃落地检测位置
        self._jump_takeoff_ny = None  # 回放: 起跳时的 y(抓绳成功判定)
        self._jump_mode = ""          # 回放: jump_takeoff 跳跃方式 climb/down/step_left/step_right/step_up
        self._jump_target_idx = -1    # 回放: jump_takeoff 的落点目标点索引(下一个点)
        self._jump_target_y = 0.0     # 回放: jump_takeoff 落点目标 y
        self._jump_aligned = False    # 回放: 普通跳起跳点是否已对齐(锁定, 防norm抖动横跳)
        self._jump_stable_at = 0.0    # 回放: 对齐后防抖计时(0.3s)
        self._climb_adj_pos = None    # 抓绳跳: 退远/走近时的位置(不动超时检测)
        self._climb_adj_ts = 0.0      # 抓绳跳: 退远/走近位置不变计时
        self._move_stuck_pos = None   # move 段: 步进极限兜底位置
        self._move_stuck_ts = 0.0     # move 段: 步进极限不动计时
        # move 段: 纵向不可达死循环检测位置(x已对齐但y差远/不同平台)
        self._vert_stuck_pos = None
        self._vert_stuck_ts = 0.0
        self._step_until = 0.0        # 小步步进冷却: 下次允许 step 的时刻
        self._near_start = 0.0        # move/climb 段"稳定到达"计时起点
        self.last_archive = None      # 最近一次生成的路线存档路径(供用户检查)
        # ---- 安全点(测谎仪规避: 定时去打怪暂停→走进商城) ----
        self.safe_points = []          # F10 录制的安全点 [{action, nx, ny}, ...]
        self.is_recording_safe = False  # F10 安全点录制中(F2/F3 打进 safe_points)
        # ---- 恢复路线(安全点退出商城后/跌落底层时走回巡游线) ----
        self.recall_points = []          # F11 录制的恢复路线 [{action, nx, ny}, ...]
        self.is_recording_recall = False  # F11 恢复路线录制中(F2/F3 打进 recall_points)
        self.one_shot = False          # one-shot: 走完最后一个点即停, 不循环
        self._one_shot_done = False    # one-shot 已走完(主循环据此执行商城脚本)
        self._plat_stall_ts = 0.0      # Y平台校验"整圈无同平台点"停滞警告节流
        # 完成一轮(回到第1点)时的回调(主循环注入, 用于经验统计):
        # 签名 round_complete(round_count). 未注入则无操作。
        self.on_round_complete = None

    # ---- 路径 ----
    def _path(self, map_name):
        if not map_name:
            return None
        return str(Path("minimaps") / map_name / "waypoints.json")

    @staticmethod
    def _sanitize_name(name):
        """去掉文件名非法字符(Windows: \\/:*?\"<>|)。"""
        for ch in '\\/:*?"<>|':
            name = name.replace(ch, "_")
        return name.strip() or "map"

    def _archive_dir(self, map_name=None):
        """路线存档目录: minimaps/{地图}/routes/ (每个存档=一次录制结果)。"""
        if not map_name:
            map_name = self.map_name
        if not map_name:
            return None
        return str(Path("minimaps") / map_name / "routes")

    def _latest_archive(self, map_name=None):
        """routes 目录里最新的时间戳存档(文件名 YYYYMMDD_HHMMSS_地图.json,
        字典序 == 时间序, 取最后一个)。"""
        d = self._archive_dir(map_name)
        if not d or not os.path.isdir(d):
            return None
        files = sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if f.endswith(".json"))
        return files[-1] if files else None

    def load_waypoints(self, map_name=None):
        """加载路线(动作段格式): 优先 routes/ 最新时间戳存档; 无则回退固定文件。
        旧格式(非动作段)无法可靠回放, 忽略并提示重录。"""
        if map_name:
            self.map_name = map_name
        self.waypoints = []
        self.idx = 0
        self._fwd = True
        candidates = []
        arch = None  # 实际加载来源(主文件为 None; 回退存档时为该存档路径)
        # 优先读主文件 waypoints.json(它是 F6 重定位/纠偏后写入的权威数据;
        # routes/ 时间戳存档是录制时的历史记录, 若优先读它会覆盖掉用户在
        # 游戏里按 F6 纠偏后的最新坐标——重启后又变成旧坐标)。
        p = self._path(self.map_name)
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                candidates = data if isinstance(data, list) else []
            except Exception:
                candidates = []
        # 主文件缺失/损坏时才回退 routes/ 最新时间戳存档
        if not candidates:
            arch = self._latest_archive(self.map_name)
            if arch:
                try:
                    with open(arch, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    pts = data.get("points", data) if isinstance(data, dict) else data
                    candidates = list(pts) if isinstance(pts, list) else []
                except Exception:
                    candidates = []
        # 只接受动作段格式: 每个元素必须 {action∈动作集, nx, ny} 且无旧格式 is_p 字段
        self.waypoints = [
            w for w in candidates
            if isinstance(w, dict) and w.get("action") in self.ACTION_ALL
            and "nx" in w and "ny" in w and "is_p" not in w
        ]
        if self.waypoints and len(self.waypoints) == len(candidates):
            _src = os.path.basename(arch) if arch else "waypoints.json"
            logger.info(
                f"[wp] 已加载路线 {len(self.waypoints)} 段 ({_src})")
            return True
        # 旧格式/混合格式: 无法可靠回放, 拒绝并提示重录
        self.waypoints = []
        logger.warning("[wp] 检测到旧格式路线存档, 无法回放 — 请 F4 清除后重新录制")
        return False

    def save(self, map_name=None):
        """保存路线(动作段): 写固定 waypoints.json + 时间戳存档
        minimaps/{地图}/routes/YYYYMMDD_HHMMSS_{地图}.json(用户检查用)。
        文件被占用/权限异常时只警告返回 False, 不抛异常(防闪退)。"""
        try:
            if map_name:
                self.map_name = map_name
            p = self._path(self.map_name)
            if not p or not self.waypoints:
                return False
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.waypoints, f, ensure_ascii=False, indent=1)
            # 时间戳存档(命名: 日期时间+地图名)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ad = self._archive_dir()
            os.makedirs(ad, exist_ok=True)
            arch = os.path.join(
                ad, f"{ts}_{self._sanitize_name(self.map_name)}.json")
            with open(arch, "w", encoding="utf-8") as f:
                json.dump({
                    "map": self.map_name,
                    "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "count": len(self.waypoints),
                    "points": self.waypoints,
                }, f, ensure_ascii=False, indent=2)
            self.last_archive = arch
            logger.info(f"[wp] 路线存档已生成: {arch}")
            return True
        except Exception as e:
            logger.warning(f"[wp] 路线存档失败(不中断运行): {e}")
            return False

    def clear(self):
        self.waypoints = []
        self.idx = 0
        self._fwd = True
        self._patrolling = False
        self.cancel_rest()
        self._rec_prev_keys = set()
        self._rec_seg = ""
        self._rec_land_frames = 0
        self._rec_land_pos = None
        self._reset_attempt()

    def _reset_attempt(self):
        """进入新动作段时重置回放状态。"""
        self._point_start = 0.0
        self._retry_count = 0
        self._action_state = ""
        self._action_at = 0.0
        self._action_done = False
        self._climb_last_pos = None
        self._climb_last_ts = 0.0
        self._jump_land_frames = 0
        self._jump_land_pos = None
        self._jump_takeoff_ny = None
        self._jump_mode = ""
        self._jump_target_idx = -1
        self._jump_target_y = 0.0
        self._jump_aligned = False
        self._jump_stable_at = 0.0
        self._climb_adj_pos = None
        self._climb_adj_ts = 0.0
        self._move_stuck_pos = None
        self._move_stuck_ts = 0.0
        self._vert_stuck_pos = None
        self._vert_stuck_ts = 0.0
        self._step_until = 0.0
        self._near_start = 0.0

    # ---- 录制(段式关键点) ----
    @staticmethod
    def _op_keys(held):
        """只关心方向+跳跃键(left/right/up/down/space)。"""
        return {k for k in held if k in ("left", "right", "up", "down", "space")}

    def _push_seg(self, action, nx, ny):
        """记一个动作段 {action, nx, ny}; 与上一段动作相同且终点很近 -> 合并(连续走位/爬绳)。"""
        nx, ny = round(float(nx), 4), round(float(ny), 4)
        # 空段过滤: move 段终点与上一段几乎重合(原地左右挪动产生的无意义段, 如
        # 松手瞬间又按了一下反方向) -> 丢弃, 避免回放时"到点立即又推进"的空转
        if (action in self.ACTION_MOVE and self.waypoints
                and self.waypoints[-1]["action"] in self.ACTION_MOVE
                and abs(nx - self.waypoints[-1]["nx"]) < 0.008
                and abs(ny - self.waypoints[-1]["ny"]) < 0.008):
            logger.info(
                f"[wp] 录制过滤空段: {action} 终点与上段重合({nx:.4f},{ny:.4f})")
            return
        if self.waypoints and self.waypoints[-1]["action"] == action:
            self.waypoints[-1]["nx"] = nx
            self.waypoints[-1]["ny"] = ny
            return
        self.waypoints.append({"action": action, "nx": nx, "ny": ny})
        if len(self.waypoints) > self.max_waypoints:
            self.waypoints = self.waypoints[-self.max_waypoints:]

    def record_sample(self, map_norm, held_keys, now):
        """【手动打点模式】自动按键录制已废弃: 点位由用户按 F2(普通点)/F3(跳跃点)
        手动打点生成(add_manual_point)。此方法保留签名但不再记点, 防止旧调用冲突。"""
        return

    def add_manual_point(self, action, map_norm):
        """手动打点: 记录当前位置为一个点位(F2 普通点 / F3 跳跃点)。
        action: "move" 或 "jump_takeoff"。map_norm: [nx, ny] 小地图归一化坐标。
        【不四舍五入】: 直接存检测原始值(像素/画布宽度的浮点, 精度远高于4位),
        保证记录的点位就是打点瞬间的真实坐标, 不因 round 产生偏差。"""
        return self.add_manual_point_to(self.waypoints, action, map_norm)

    def add_manual_point_to(self, target, action, map_norm):
        """同 add_manual_point, 但写入指定列表(安全点录制时写入 safe_points)。"""
        if map_norm is None:
            logger.warning("[wp] 打点失败: 无小地图坐标")
            return False
        # 原样保留原始精度(不 round): 点位必须精确
        nx, ny = float(map_norm[0]), float(map_norm[1])
        # 与上一点几乎重合 -> 忽略(防止重复打点)
        if (target and abs(nx - target[-1]["nx"]) < 0.004
                and abs(ny - target[-1]["ny"]) < 0.004):
            logger.info(
                f"[wp] 打点忽略: 与上一点重合({nx:.4f},{ny:.4f})")
            return False
        target.append({"action": action, "nx": nx, "ny": ny})
        if len(target) > self.max_waypoints:
            del target[:len(target) - self.max_waypoints]
        _cn = "普通点" if action == "move" else "跳跃点"
        logger.info(
            f"[wp] {_cn}已打 第{len(target)}个: ({nx:.4f},{ny:.4f})")
        return True

    def _step_cmd(self, dx, reason, now):
        """小步步进(带冷却): 每次 step 后等 step_interval 秒再允许下一步——
        小地图坐标检测更新慢(每帧才83ms), 连续 step 会让坐标来不及反应、
        看起来碎步很多且容易在台阶边缘踏空。冷却期间返回 none 等待。
        接近跳跃点位(jump 段)时步频再降一倍(step_interval*2, 更稳对齐)。"""
        _interval = self.step_interval
        # 当前段是跳跃点/跳台阶时: 步频减半(间隔加倍), 防快速碎步踏空
        _cur = self.waypoints[self.idx] if self.waypoints else None
        if _cur is not None and _cur.get("action") in self.ACTION_JUMP:
            _interval = self.step_interval * 2.0
        if now >= self._step_until:
            self._step_until = now + _interval
            return (("step_left" if dx < 0 else "step_right"), reason)
        return "none", reason + "_w"

    # ---- 回放(逐段执行 + 失误从头, 纯录制回放) ----
    def _find_same_platform(self, start_idx, cur_ny):
        """从 start_idx 起(环形)找第一个与 cur_ny 同平台(|ΔY|<=recover_y_tol)的段。
        返回索引; 整圈都没有返回 -1。"""
        n = len(self.waypoints)
        for _k in range(n):
            _i = (start_idx + _k) % n
            _d = abs(float(self.waypoints[_i]["ny"]) - cur_ny)
            if _d <= self.recover_y_tol + 1e-9:
                return _i
        return -1

    def decide(self, map_norm, now):
        """按动作段序列执行(纯录制回放): 直接看 JSON 里记录的到达方式执行,
        不做 dx/dy 几何逻辑判断。每段到达目标才推进下一段;
        某段失败(跳跃未达落点/超时) -> 从头开始(回到第 1 段)。

        段类型:
          - move_left/right: 按住方向走到目标 x;
          - jump/jump_left/jump_right/jump_down: 执行对应跳跃, 落地后检查落点;
          - climb_up/down: 按上/下爬到目标 y。
        返回 (command, reason) 或 None(交给边缘巡游兜底)。
        """
        if not self.waypoints or map_norm is None:
            return None
        if not self._patrolling:
            return None
        if self.idx >= len(self.waypoints):
            self.idx = 0
        # 安全点行程 one-shot 已走完: 停在最后一点(主循环执行商城脚本)
        if self._one_shot_done:
            return "none", "safe_done"
        seg = self.waypoints[self.idx]
        action = seg.get("action", "move_left")
        # 环形循环(全程正向): 不做方向翻转——每个点的跳跃语义始终按录制时的
        # "点->下一个点"判定(如点2永远是"立定跳上绳子", 不会因循环变左跳)
        tnx, tny = float(seg["nx"]), float(seg["ny"])
        nx, ny = float(map_norm[0]), float(map_norm[1])
        dx, dy = tnx - nx, tny - ny

        # 【Y平台校验(用户要求: 只有相同Y坐标的点才会被巡航选中)】:
        # 当前段点与角色不在同平台(ΔY > recover_y_tol)时不执行该点——
        # 前进方向找第一个同Y段切换过去; 整圈都没有则停滞等待(节流警告),
        # 绝不原地跳/死循环(修复: A平台角色触发B平台同X跳点原地一直跳,
        # 以及无同平台点时'恢复→重启'无限死循环导致圈走不完, 安全点/
        # 经验统计全部卡死)。爬绳段(climb)目标在上方属正常, 不校验。
        if (not self._action_state
                and action not in self.ACTION_CLIMB
                and abs(ny - tny) > self.recover_y_tol):
            _bi = self._find_same_platform(self.idx, ny)
            if _bi >= 0:
                logger.warning(
                    f"[wp] 段{self.idx + 1} 点Y={tny:.4f} 与角色Y={ny:.4f} "
                    f"不在同平台(ΔY={abs(ny - tny):+.4f}), 切换到同平台点"
                    f" 第{_bi + 1}个")
                self.idx = _bi
                self._reset_attempt()
            else:
                if now - self._plat_stall_ts > 5.0:
                    self._plat_stall_ts = now
                    logger.warning(
                        f"[wp] 角色所在平台(Y={ny:.4f}) 整圈无匹配点位, "
                        f"停止执行等待 — 请重新录制路线(F1-F4)或按 F6 重定位")
                self._point_start = now   # 停滞等待: 不触发40s超时重启
            return "none", "wp_y_platform"

        # ---- 超时保护: 当前段尝试过久 -> 从头开始 ----
        if self._point_start == 0.0:
            self._point_start = now
        if now - self._point_start > self.retry_timeout:
            logger.warning(f"[wp] 段{self.idx + 1}({action}) 超时, 从头开始")
            self._restart()
            return "none", "wp_restart"

        # ---- 偏离恢复(被怪击退/掉坑): 当前段执行中角色离目标很远 -> 重置
        # 对齐状态, 让走位重新对齐目标(而不是误判到达或卡死)。
        # 【只在 move 段启用】: 跳跃点对齐时角色离跳点 0.1~0.2 是正常的(还没走到),
        # 若按偏离恢复会反复重置防抖 -> 永远跳不了(卡死); 跳跃点有对齐/坠落/超时
        # 自己的恢复机制, 不需要偏离恢复。
        _dist = float(np.hypot(dx, dy))
        if (_dist > 0.15 and not self._action_state
                and action in self.ACTION_MOVE):
            self._jump_aligned = False
            self._jump_stable_at = 0.0
            self._action_state = ""
            self._near_start = 0.0
            logger.info(
                f"[wp] 段{self.idx + 1} 偏离目标(dist={_dist:.3f}), 重置对齐恢复")

        # ---- move 段: 按住方向走到目标点(近距离减速步进, 防冲过头) ----
        # 移动方向完全由 dx 决定(目标在左往左走/在右往右走), 不依赖录制的
        # 方向——往返行驶时角色从终点走回起点, 方向与录制相反也能走对。
        # 【到达判定必须 dx+dy 双查】: 只查 dx 的话, 角色被击退到别的平台
        # (y 差很远)也会误判"到达"而跳过该点位。
        if action in self.ACTION_MOVE:
            # 掉层保护: 目标在当前点上方(dy<0)且角色掉到下方台阶/被击退低层,
            # 水平走不到上方目标 -> 同Y恢复。
            # 【修复"平地走点位突变"误触发】: 原条件 dy<-0.08 + 1.5s —— 下坡/
            # 走位中角色 y 短暂低于目标(正常攀升)也会误触, 跳到同Y点位突变。
            # 改为【持续 3s 仍低于目标】(正常走位很快接近/超过目标, dy 不会
            # 长时间为负; 真掉层才会一直上不去)。
            if dy < -0.08 and now - self._point_start > 3.0:
                logger.warning(
                    f"[wp] 段{self.idx + 1} move 目标在上方(dy={dy:+.3f}, 掉到下方了), 同Y恢复")
                self._recover_same_y(ny)
                return "none", "wp_recover"
            # 【纵向不可达("回到不同平台点位"死循环)】: 遍历到下层的点位,
            # 角色在上层【同 X 位置来回踱步】——move 段只做水平走, 永远到不了
            # 不同平台的 y。
            # 【修复误触发】: 不能只凭"dx对齐+dy>0.10+超时"就跳——下坡/拐弯
            # 时 dx 短暂对齐、dy 暂时大, 会误判"不可达"导致点位突变(用户反馈
            # 平地走也跳点)。正确判定必须【角色 Y 长时间没有下降】(还停在高于
            # 目标的平台上, 不是正在下坡/掉层中):
            #   - 目标在下方(dy > 0.10)
            #   - 角色 Y 与目标的差距持续 > 0.10(ny 没接近目标, 即没在下降)
            #   - 横向持续对齐(dx 稳定在 ±0.05, 不依赖单帧)
            #   - 持续超时 4s
            if (abs(dx) <= 0.05 and dy > 0.10
                    and (tny - ny) > 0.10
                    and now - self._point_start > 4.0):
                # 记录首帧满足条件时刻; 需持续 1.5s 才确认(防下坡瞬时波动)
                if self._vert_stuck_pos is None:
                    self._vert_stuck_pos = (nx, ny)
                    self._vert_stuck_ts = now
                elif (abs(nx - self._vert_stuck_pos[0]) < 0.003
                        and now - self._vert_stuck_ts > 1.5):
                    logger.warning(
                        f"[wp] 段{self.idx + 1} move 目标在不同平台"
                        f"(x对齐: dx={dx:+.4f}, 角色Y未下降: dy={dy:+.3f}), "
                        f"同层走不到, 跳过该点继续")
                    self._next()
                    return "none", "wp_skip_unreachable"
            else:
                self._vert_stuck_pos = None
            # 步进极限兜底: 已非常接近(|dx|<=0.02)但按键步进粒度(≈0.004)无法精确到
            # 4位一致, 且角色 2s 内位置基本不变(微调按键已无效果) -> 接受当前最近
            # 位置为到达(避免无限微调死锁; 此时与点位差已不足 2px 小地图)
            if (abs(dx) <= 0.02 and abs(dy) <= 0.03
                    and now - self._point_start > 2.0):
                if self._move_stuck_pos is None:
                    self._move_stuck_pos = (nx, ny)
                    self._move_stuck_ts = now
                _moved = abs(nx - self._move_stuck_pos[0]) + abs(ny - self._move_stuck_pos[1])
                if _moved < 0.0005 and now - self._move_stuck_ts > 2.0:
                    logger.info(
                        f"[wp] 段{self.idx + 1} 步进极限({nx:.4f},{ny:.4f} "
                        f"vs {tnx:.4f},{tny:.4f}), 视为到达")
                    self._next()
                    return "none", "wp_arrived"
                if _moved >= 0.0005:
                    self._move_stuck_pos = (nx, ny)
                    self._move_stuck_ts = now
            else:
                self._move_stuck_pos = None
            # 【普通点位到达放宽】: 接近即可(move_arrive_tol=0.01≈1px), 不需要
            # 精确到 4 位小数(跳跃点位仍保持 jump_align_x 精确对齐)。
            if abs(dx) > self.move_arrive_tol or abs(dy) > self.move_arrive_tol:
                self._near_start = 0.0
                # 近距离(|Δx|<=0.03)用极短步进慢走: 小地图坐标更新慢, 全速走会
                # 冲过目标来回震荡(角色走到绳下却判已走过, 又走回来死循环)
                if abs(dx) <= 0.03:
                    logger.info(
                        f"[wp] move 步进 dx={dx:+.4f}(目标 {tnx:.4f}, 当前 {nx:.4f})")
                    return self._step_cmd(dx, "wp_move_step", now)
                return ("move_left", "wp_move") if dx < 0 else ("move_right", "wp_move")
            # x 且 y 都到: 稳定 0.3s(防小地图抖动误判)后推进
            if self._near_start == 0.0:
                self._near_start = now
            if now - self._near_start < 0.3 - 1e-3:
                return "none", "wp_move_wait"
            logger.info(
                f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 走达 ({tnx:.4f},{tny:.4f})")
            self._next()
            return "none", "wp_arrived"

        # ---- jump 段: 对齐起跳点 -> 执行跳跃 -> 落地 -> 检查 ----
        if action in self.ACTION_JUMP:
            # 抓绳跳(jump_climb): 走独立规则——不在正下方立定跳, 而是
            # 离绳子 0.03 左右朝绳跳 + 一直按上(用户实测: 太近抓不住)
            if action == "jump_climb":
                return self._jump_climb_decision(dx, tnx, nx, ny, now)
            # 跳跃点(jump_takeoff, 手动打点): 起跳方式由"跳点->下一个点"关系决定
            if action == "jump_takeoff":
                return self._jump_takeoff_decision(dx, tnx, tny, nx, ny, now)
            # 1) 对齐起跳点 x: 用户要求误差 <= 0.003(否则拿不到绳子)。
            #    【严格对齐, 无锁定容忍】: |Δx| > jump_align_x(0.003) 一律重新走位,
            #    绝不在偏差点起跳(走位冲过头也会走回来)。
            if abs(dx) > self.jump_align_x:
                self._jump_aligned = False
                self._jump_stable_at = 0.0
                self._action_state = ""
                self._jump_land_frames = 0
                self._jump_land_pos = None
                if abs(dx) <= 0.025:
                    logger.info(
                        f"[wp] jump 步进 dx={dx:+.4f}(目标 {tnx:.4f}, 当前 {nx:.4f})")
                    return self._step_cmd(dx, "wp_jump_step", now)
                    return ("move_left", "wp_jump_align") if dx < 0 else ("move_right", "wp_jump_align")
            else:
                self._jump_aligned = True
            # 2) 已对齐(锁定): 防抖 0.3s(小地图 norm 抖动)后直接起跳——
            #    避免"对上了就跳, 下一帧抖动又偏了"的反复横跳
            if not self._action_state:
                if self._jump_stable_at == 0.0:
                    self._jump_stable_at = now
                if now - self._jump_stable_at < 0.3 - 1e-3:
                    return "none", "wp_jump_wait"
            # 3) 起跳(按录制动作)
            _jcmd = action
            if self._action_state != _jcmd:
                self._jump_aligned = False  # 起跳后解锁(下一段重新对齐)
                self._action_state = _jcmd
                self._action_at = now
                self._jump_land_frames = 0
                self._jump_land_pos = None
                return _jcmd, "wp_jump"
            if now - self._action_at < self.action_hold:
                return "none", "wp_jump_hold"
            # 普通跳: 等待落地(位置连续 4 帧稳定)
            if (self._jump_land_pos is None
                    or abs(nx - self._jump_land_pos[0])
                    + abs(ny - self._jump_land_pos[1]) < 0.01):
                self._jump_land_frames += 1
            else:
                self._jump_land_frames = 0
            self._jump_land_pos = (nx, ny)
            if self._jump_land_frames < 4:
                return "none", "wp_jump_air"
            # 落地: 检查是否到达录制的落点(放宽: 跳到平台附近即成功,
            # 跳跃有自然偏差, 差一点不算失误)
            if abs(dx) <= self.jump_arrive_x and abs(dy) <= self.jump_arrive_y:
                logger.info(
                    f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 跳达 ({tnx:.4f},{tny:.4f})")
                self._next()
                return "none", "wp_arrived"
            # 没跳到落点: 先重试当前段(可能跳偏/没跳好), 多次失败才从头——
            # 避免一次没跳好就从头导致的死循环
            self._retry_count += 1
            if self._retry_count >= self.retry_max:
                logger.warning(
                    f"[wp] 段{self.idx + 1}({action}) 重试{self.retry_max}次未跳达, 从头开始")
                self._restart()
                return "none", "wp_restart"
            logger.warning(
                f"[wp] 段{self.idx + 1}({action}) 未跳达(落点 {nx:.3f},{ny:.3f} "
                f"vs {tnx:.3f},{tny:.3f}), 重试第{self._retry_count}次")
            # 重置动作状态, 让 decide 重新对齐起跳点再跳(保留 _retry_count)
            self._point_start = now
            self._action_state = ""
            self._action_at = 0.0
            self._jump_land_frames = 0
            self._jump_land_pos = None
            return "none", "wp_jump_retry"

        # ---- climb 段: 按上/下爬到目标 y ----
        if action in self.ACTION_CLIMB:
            # 先水平对齐(爬绳需在绳子范围)
            if action == "climb_up" and abs(dx) > self.arrive_x:
                self._near_start = 0.0
                return ("move_left", "wp_climb_align") if dx < 0 else ("move_right", "wp_climb_align")
            # 持续爬(execute 状态保持)
            if self._climb_last_pos is None:
                self._climb_last_pos = (nx, ny)
                self._climb_last_ts = now
            moved = float(np.hypot(nx - self._climb_last_pos[0],
                                   ny - self._climb_last_pos[1]))
            if now - self._climb_last_ts > self.climb_timeout and moved < 0.005:
                self._climb_last_pos = (nx, ny)
                self._climb_last_ts = now
                logger.warning("[wp] 爬绳卡住, 继续尝试")
            elif moved >= 0.005:
                self._climb_last_pos = (nx, ny)
                self._climb_last_ts = now
            # 到达目标高度: 稳定 0.3s 后推进(需 dy+dx 双查, 防止在错误绳子上
            # 爬到目标高度也被判到达)
            if abs(dy) <= self.arrive_y and abs(dx) <= self.arrive_x:
                if self._near_start == 0.0:
                    self._near_start = now
                if now - self._near_start < 0.3 - 1e-3:
                    return action, "wp_climb_up"
                logger.info(
                    f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 爬达 ({tnx:.4f},{tny:.4f})")
                self._next()
                return "none", "wp_arrived"
            self._near_start = 0.0
            return action, "wp_climb_up"

        # 未知动作: 直接推进
        logger.warning(f"[wp] 段{self.idx + 1} 未知动作 {action}, 跳过")
        self._next()
        return "none", "wp_skip"

    def _jump_climb_decision(self, dx, tnx, nx, ny, now):
        """抓绳跳(jump_climb)回放规则——基于用户实测:
          绳子 x=0.3975 时, 0.3675(左0.03)右跳+按上 和 0.4275(右0.03)左跳+按上
          能抓住; 0.3893(左0.008)/0.4075(右0.01) 太近怎么跳都抓不住。
          → 不在正下方立定跳, 而是【离绳子约 0.03 处朝绳方向跳 + 一直按上】。

          分区(dx = 绳x - 角色x, dx>0 角色在绳左):
            - 太近 |dx|<0.015: 朝远离绳子方向退, 退到起跳区再跳;
            - 起跳区 0.015<=|dx|<=0.04: 角色在左 -> jump_climb_right(右跳+按上),
              在右 -> jump_climb_left;
            - 太远 |dx|>0.04: 朝绳子走近到起跳区。
          退远/走近时若位置连续 2s 不动(卡墙/游戏未聚焦) -> 强制跳一次并提示,
          避免"太近阈值边界+角色不动"导致的无限退远卡死。
          跳起后: y 明显下降(角色在绳上上升) = 抓绳成功 -> 推进;
          1s 内未上绳 -> 重试当前段, 超限从头。"""
        # 已起跳: 抓绳成功判定
        if self._action_state.startswith("jump_climb"):
            if ny < self._jump_takeoff_ny - 0.02:
                logger.info(
                    f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 抓绳成功, 推进")
                self._next()
                return "none", "wp_arrived"
            if now - self._action_at < self.action_hold + 1.0:
                return "none", "wp_jump_air"
            self._retry_count += 1
            if self._retry_count >= self.retry_max:
                logger.warning(
                    f"[wp] 段{self.idx + 1}(jump_climb) 重试{self.retry_max}次未抓到绳, 从头开始")
                self._restart()
                return "none", "wp_restart"
            logger.warning(
                f"[wp] 段{self.idx + 1}(jump_climb) 未抓到绳, 重试第{self._retry_count}次")
            self._point_start = now
            self._action_state = ""
            self._action_at = 0.0
            return "none", "wp_jump_retry"
        # 未起跳: 按 dx 分区决策
        adx = abs(dx)
        if adx < 0.015:
            # 太近(正下方附近): 跳抓不住 -> 朝远离绳子方向退到起跳区
            if not self._climb_adj_pos:
                self._climb_adj_pos = (nx, ny)
                self._climb_adj_ts = now
            moved = abs(nx - self._climb_adj_pos[0]) + abs(ny - self._climb_adj_pos[1])
            if moved >= 0.008:
                self._climb_adj_pos = (nx, ny)
                self._climb_adj_ts = now
            elif now - self._climb_adj_ts > 2.0:
                # 退不动(卡墙/游戏未聚焦): 强制跳一次打破死循环
                self._climb_adj_pos = None
                logger.warning(
                    f"[wp] 段{self.idx + 1}(jump_climb) 退远位置不动, 强制跳一次 "
                    f"(请确认游戏窗口已聚焦)")
                _jcmd = "jump_climb_right" if dx > 0 else "jump_climb_left"
                self._action_state = _jcmd
                self._action_at = now
                self._jump_land_frames = 0
                self._jump_land_pos = None
                self._jump_takeoff_ny = ny
                return _jcmd, "wp_jump"
            if dx > 0:  # 角色在绳左 -> 向左退
                return self._step_cmd(dx, "wp_climb_back", now)
            return self._step_cmd(dx, "wp_climb_back", now)
        if adx > 0.04:
            # 太远: 朝绳子走近(步进慢走, 防冲过头)
            if not self._climb_adj_pos:
                self._climb_adj_pos = (nx, ny)
                self._climb_adj_ts = now
            moved = abs(nx - self._climb_adj_pos[0]) + abs(ny - self._climb_adj_pos[1])
            if moved >= 0.008:
                self._climb_adj_pos = (nx, ny)
                self._climb_adj_ts = now
            elif now - self._climb_adj_ts > 2.0:
                # 走近不动(卡墙/游戏未聚焦): 强制跳一次打破死循环
                self._climb_adj_pos = None
                logger.warning(
                    f"[wp] 段{self.idx + 1}(jump_climb) 走近位置不动, 强制跳一次 "
                    f"(请确认游戏窗口已聚焦)")
                _jcmd = "jump_climb_right" if dx > 0 else "jump_climb_left"
                self._action_state = _jcmd
                self._action_at = now
                self._jump_land_frames = 0
                self._jump_land_pos = None
                self._jump_takeoff_ny = ny
                return _jcmd, "wp_jump"
            if dx > 0:  # 角色在绳左 -> 向右走近
                return self._step_cmd(dx, "wp_climb_approach", now)
            return self._step_cmd(dx, "wp_climb_approach", now)
        # 起跳区(离绳 0.03 左右): 朝绳子方向跳 + 一直按上
        self._climb_adj_pos = None
        _jcmd = "jump_climb_right" if dx > 0 else "jump_climb_left"
        self._action_state = _jcmd
        self._action_at = now
        self._jump_land_frames = 0
        self._jump_land_pos = None
        self._jump_takeoff_ny = ny
        logger.info(
            f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 抓绳起跳 dx={dx:+.4f} "
            f"({tnx:.4f} vs {nx:.4f}) -> {_jcmd}")
        return _jcmd, "wp_jump"

    def _jump_takeoff_decision(self, dx, tnx, tny, nx, ny, now):
        """跳跃点(手动打点)回放——起跳方式由【跳点→下一个点】的录制关系决定,
        不用角色当前 dx 猜方向(用户要求"取消 dx 说法"):
          - 下一目标 Y 差 <= -climb_jump_dy(0.05, 目标在上方/绳子) -> 跳+按住上,
            爬绳直到到达目标 Y;
          - |Y 差| < 0.05(同层跳台阶) -> 按下一目标 X 方向: 右跳/左跳/立定跳;
          - Y 差 >= 0.05(目标在下方) -> 跳下。
        起跳位置必须精确对齐(锁定 + 防抖 0.3s)。
        跳台阶/跳下落地未达目标点(失误/掉坑) -> 同 Y 点位恢复(从同 Y 的点继续)。
        注意: Y平台校验(同平台才选中)在 decide() 顶部统一做(未起跳时),
        这里不做(起跳/空中 Y 变化大, 会误伤)。
        """
        n = len(self.waypoints)
        # 落点目标 = 下一个点(环形循环: 始终取 idx+1, 最后一个点回到第1个点)
        _nidx = self.idx + 1
        if _nidx >= n:
            _nidx = 0
        _nseg = self.waypoints[_nidx]
        _nnx, _nny = float(_nseg["nx"]), float(_nseg["ny"])

        # 0) 已起跳: 不再做对齐判定(起跳后位置必然变化), 直接处理爬绳/落地
        if self._action_state == "takeoff":
            if now - self._action_at < self.action_hold:
                return "none", "wp_jump_hold"
            # 爬绳模式: 跳+按上 -> 抓绳后持续爬, 直到到达目标 Y
            if self._jump_mode == "climb":
                if ny <= self._jump_target_y + self.arrive_y:
                    logger.info(
                        f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 爬绳到达 "
                        f"目标y={self._jump_target_y:.4f}")
                    self._next()
                    self._next()
                    return "none", "wp_arrived"
                if ny < self._jump_takeoff_ny - 0.02:   # 在绳上上升中
                    return "climb_up", "wp_climb_up"
                if now - self._action_at < self.action_hold + 1.0:
                    return "none", "wp_jump_air"
                # 未抓到绳 -> 重试起跳(有限次, 防死循环: 每次重试都重置
                # _point_start 会让 40s 超时永不触发, 必须计数)
                self._retry_count += 1
                if self._retry_count >= self.retry_max:
                    logger.warning(
                        f"[wp] 段{self.idx + 1} 抓绳失败{self._retry_count}次, "
                        f"同平台点位恢复")
                    self._recover_same_y(ny)
                    return "none", "wp_recover"
                logger.warning(
                    f"[wp] 段{self.idx + 1} 跳起未抓绳, 重试第{self._retry_count}次")
                self._point_start = now
                self._action_state = ""
                self._action_at = 0.0
                return "none", "wp_jump_retry"
            # 跳台阶/跳下: 等待落地(位置连续 4 帧稳定)
            if (self._jump_land_pos is None
                    or abs(nx - self._jump_land_pos[0])
                    + abs(ny - self._jump_land_pos[1]) < 0.01):
                self._jump_land_frames += 1
            else:
                self._jump_land_frames = 0
            self._jump_land_pos = (nx, ny)
            if self._jump_land_frames < 4:
                return "none", "wp_jump_air"
            # 落地: 检查是否到达目标点(下一跳点)——只有碰到点位才算到达
            if (abs(nx - _nnx) <= self.step_arrive_x
                    and abs(ny - _nny) <= self.step_arrive_y):
                logger.info(
                    f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 跳跃到达 "
                    f"目标点{_nidx + 1} ({_nnx:.4f},{_nny:.4f})")
                self._next()
                self._next()
                return "none", "wp_arrived"
            # 【下跳(跳下)专用】: 落地比起跳点低 = 穿台成功, 不是坠落——
            # 不允许"坠落重头", 否则成功下跳会被误判失败从头开始
            if self._jump_mode == "down":
                if ny <= self._jump_takeoff_ny + 0.02:
                    # 没穿下去(落回原平台): 有限重试下跳, 超限则同Y恢复
                    self._retry_count += 1
                    if self._retry_count >= self.retry_max:
                        logger.warning(
                            f"[wp] 段{self.idx + 1} 下跳失败{self._retry_count}次, "
                            f"同平台点位恢复")
                        self._recover_same_y(ny)
                    else:
                        self._point_start = now
                        self._action_state = ""
                        self._action_at = 0.0
                        logger.warning(
                            f"[wp] 段{self.idx + 1} 下跳未穿台(落地y={ny:.4f}≈起跳y"
                            f"={self._jump_takeoff_ny:.4f}), 重试第{self._retry_count}次")
                    return "none", "wp_jump_retry"
                # 已落到下层但没碰到目标点 -> 同 Y 恢复(目标点同层, 从目标点继续)
                self._recover_same_y(ny)
                return "none", "wp_recover"
            # 坠落检测: 落地 Y 比起跳点 Y 大(小地图 y 向下为正, y 大=更低层)
            # = 角色从台阶上掉下去了 -> 恢复循环, 从头重新开始
            if ny > self._jump_takeoff_ny + 0.02:
                logger.warning(
                    f"[wp] 段{self.idx + 1} 跳台阶坠落(落地y={ny:.4f} > "
                    f"起跳y={self._jump_takeoff_ny:.4f}), 从头重新开始")
                self._restart()
                return "none", "wp_restart"
            # 没跳到目标点(跳台阶失误/掉坑) -> 同 Y 点位恢复
            self._recover_same_y(ny)
            return "none", "wp_recover"

        # 1) 精确对齐起跳点 x: 用户要求误差 <= 0.003。严格对齐, 无锁定容忍——
        #    |Δx| > jump_align_x(0.003) 一律重新走位, 走位冲过头也会走回来
        if abs(dx) > self.jump_align_x:
            self._jump_aligned = False
            self._jump_stable_at = 0.0
            self._action_state = ""
            if abs(dx) <= 0.025:
                logger.info(
                    f"[wp] 跳点步进 dx={dx:+.4f}(目标 {tnx:.4f}, 当前 {nx:.4f})")
                return self._step_cmd(dx, "wp_jump_step", now)
            return ("move_left", "wp_jump_align") if dx < 0 else ("move_right", "wp_jump_align")
        else:
            self._jump_aligned = True
        if not self._action_state:
            if self._jump_stable_at == 0.0:
                self._jump_stable_at = now
            if now - self._jump_stable_at < 0.3 - 1e-3:
                return "none", "wp_jump_wait"

        # 2) 起跳: 按"跳点->下一目标"关系决定跳跃方式(小地图 y 向下为正,
        #    目标 y 更小 = 在上方 = 爬绳)
        _dny = _nny - tny
        _dnx = _nnx - tnx
        self._jump_target_idx = _nidx
        self._jump_target_y = _nny
        self._jump_takeoff_ny = ny
        if _dny <= -self.climb_jump_dy:
            self._jump_mode = "climb"
            _jcmd = "jump_climb"          # 目标在上方(绳子): 跳+按住上, 爬绳到目标Y
        elif _dny >= self.climb_jump_dy:
            self._jump_mode = "down"
            _jcmd = "jump_down"           # 目标在下方: 跳下
        else:
            # 同层跳台阶: 只要下一目标 X 有变化就按方向跳(左跳/右跳),
            # 只有 X 几乎重合(<0.005)才立定跳(用户要求: 按下一个点位X方向跳)
            if _dnx > 0.005:
                self._jump_mode = "step_right"
                _jcmd = "jump_right"      # 跳台阶: 目标在右
            elif _dnx < -0.005:
                self._jump_mode = "step_left"
                _jcmd = "jump_left"       # 跳台阶: 目标在左
            else:
                self._jump_mode = "step_up"
                _jcmd = "jump"            # 立定跳(X几乎重合)
        self._action_state = "takeoff"
        self._action_at = now
        self._jump_land_frames = 0
        self._jump_land_pos = None
        logger.info(
            f"[wp] 段{self.idx + 1}/{len(self.waypoints)} 跳跃方式={self._jump_mode} "
            f"目标点{_nidx + 1}({_nnx:.4f},{_nny:.4f}) -> {_jcmd}")
        return _jcmd, "wp_jump"

    def _recover_same_y(self, cur_ny):
        """跳台阶失误/掉坑后: 找【同平台】点位从该点继续——Y 必须与当前几乎一致
        (同一平台, 左右走就能到), 绝不跳到 Y 差大的点(上面平台的点走不到,
        会陷入"跳跳跳"死循环)。

        规则(用户要求: 一定要找 Y 一样的同平台点):
          1) 前进方向的同平台点(环形循环始终正向: idx+1..末尾), 满足
             |ΔY| <= recover_y_tol(0.03, 覆盖norm抖动且排除相邻平台);
          2) 前进方向没有 -> 全部点位里找同平台点(排除当前);
          3) 都没有同平台点 -> 从头开始(回到起点重新走, 不选异平台点)。
        """
        n = len(self.waypoints)
        tol = self.recover_y_tol
        best_i, best_d = -1, 1e9
        # 1) 前进方向的同平台点
        for i in range(self.idx + 1, n):
            d = abs(float(self.waypoints[i]["ny"]) - cur_ny)
            if d <= tol + 1e-9 and d < best_d:
                best_d, best_i = d, i
        # 2) 前进方向没有 -> 全部点位(排除当前)
        if best_i < 0:
            for i, p in enumerate(self.waypoints):
                if i == self.idx:
                    continue
                d = abs(float(p["ny"]) - cur_ny)
                if d <= tol + 1e-9 and d < best_d:
                    best_d, best_i = d, i
        # 3) 没有任何同平台点 -> 从头开始(绝不选异平台点, 否则死循环)
        if best_i < 0:
            logger.warning(
                f"[wp] 跳台阶失误: 当前Y={cur_ny:.4f} 无同平台点位"
                f"(容差±{tol}), 从头重新开始")
            self._restart()
            return
        self.idx = best_i
        self._reset_attempt()
        logger.warning(
            f"[wp] 跳台阶失误, 从同平台点位 第{best_i + 1}个 "
            f"(ny={self.waypoints[best_i]['ny']:.4f}, 当前Y={cur_ny:.4f}) 继续")

    def _next(self):
        """推进到下一段(环形循环): 1→2→…→N→回到第1个点重新开始, 全程正向。
        (用户要求: 走完最后一个动作后重新循环到第一个点位, 不受最后一个点位影响)
        one-shot 模式(安全点行程): 走完最后一个点不再循环, 停在最后一点。"""
        n = len(self.waypoints)
        self.idx = (self.idx + 1) % n
        if self.idx == 0:
            if self.one_shot:
                # 安全点行程: 已走完最后一个点 -> 停在该点, 通知主循环
                self._one_shot_done = True
                self.idx = max(0, n - 1)
                self._reset_attempt()
                return
            self._round_count += 1   # 走完一整圈回到起点
            logger.info(
                f"[wp] 完成第 {self._round_count} 圈巡航")
            # 经验统计回调(主循环注入): 本轮完成 -> 结算经验/耗时
            if self.on_round_complete is not None:
                try:
                    self.on_round_complete(self._round_count)
                except Exception as _e:
                    logger.warning(f"[wp] 经验统计回调失败: {_e}")
        self._reset_attempt()

    def _restart(self):
        """失误/超时: 从头开始(回到路线起点)。"""
        self.idx = 0
        self._reset_attempt()

    # ---- 安全点(测谎仪规避) ----
    def _safe_path(self, map_name):
        if not map_name:
            return None
        return str(Path("minimaps") / map_name / "safe_points.json")

    def load_safe_points(self, map_name=None):
        """加载安全点(只加载不巡航, 与主航线 waypoints.json 分开存储)。"""
        if map_name:
            self.map_name = map_name
        p = self._safe_path(self.map_name)
        if not p or not os.path.exists(p):
            self.safe_points = []
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data.get("points", data) if isinstance(data, dict) else data
            self.safe_points = [
                w for w in (pts or [])
                if isinstance(w, dict) and w.get("action") in self.ACTION_ALL
                and "nx" in w and "ny" in w
            ]
            if self.safe_points:
                logger.info(
                    f"[安全点] 已加载 {len(self.safe_points)} 个 -> {p}")
                return True
        except Exception:
            pass
        self.safe_points = []
        return False

    def save_safe_points(self, map_name=None):
        """保存安全点到 minimaps/{地图}/safe_points.json(F10 录制结束时)。"""
        try:
            if map_name:
                self.map_name = map_name
            p = self._safe_path(self.map_name)
            if not p or not self.safe_points:
                return False
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.safe_points, f, ensure_ascii=False, indent=1)
            return True
        except Exception as e:
            logger.warning(f"[安全点] 保存失败(不中断运行): {e}")
            return False

    def _begin_trip(self, points):
        """开始一次 one-shot 行程(安全点/恢复路线): 深拷贝点位为导航序列,
        走完最后一个点即停, 不循环。"""
        self.one_shot = True
        self._one_shot_done = False
        self.waypoints = [dict(w) for w in points]
        self.idx = 0
        self._reset_attempt()
        self._patrolling = True
        return len(self.waypoints) > 0

    def begin_safe_visit(self):
        return self._begin_trip(self.safe_points)

    def begin_recall(self):
        return self._begin_trip(self.recall_points)

    def _end_trip(self):
        """结束 one-shot 行程(主航线 waypoint_patrol 状态不受影响)。"""
        self.one_shot = False
        self._one_shot_done = False
        self._patrolling = False
        self.waypoints = []

    def end_safe_visit(self):
        self._end_trip()

    def end_recall(self):
        self._end_trip()

    def is_one_shot_done(self):
        return bool(self._one_shot_done)

    # ---- 恢复路线(安全点退出商城后/跌落底层时走回巡游线) ----
    def _recall_path(self, map_name):
        if not map_name:
            return None
        return str(Path("minimaps") / map_name / "recall_points.json")

    def load_recall_points(self, map_name=None):
        if map_name:
            self.map_name = map_name
        p = self._recall_path(self.map_name)
        if not p or not os.path.exists(p):
            self.recall_points = []
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data.get("points", data) if isinstance(data, dict) else data
            self.recall_points = [
                w for w in (pts or [])
                if isinstance(w, dict) and w.get("action") in self.ACTION_ALL
                and "nx" in w and "ny" in w
            ]
            if self.recall_points:
                logger.info(
                    f"[恢复路线] 已加载 {len(self.recall_points)} 个 -> {p}")
                return True
        except Exception:
            pass
        self.recall_points = []
        return False

    def save_recall_points(self, map_name=None):
        try:
            if map_name:
                self.map_name = map_name
            p = self._recall_path(self.map_name)
            if not p or not self.recall_points:
                return False
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.recall_points, f, ensure_ascii=False, indent=1)
            return True
        except Exception as e:
            logger.warning(f"[恢复路线] 保存失败(不中断运行): {e}")
            return False

    # ---- 巡航开关(F1 录制 / F2 结束 / F3 开始 / F4 清除) ----
    def start_patrol(self):
        """F3: 从第一个记忆点开始自动巡航(不在起点则先自动走过去)。"""
        if not self.waypoints:
            return False
        self.idx = 0
        self._fwd = True
        self._reset_attempt()
        self._patrolling = True
        return True

    def stop_patrol(self):
        """F4: 停止巡航(保留航点供下次 F3 使用)。"""
        self._patrolling = False
        self.cancel_rest()
        self._reset_attempt()

    def is_patrolling(self):
        return self._patrolling

    # ---- 休息机制: 每 N 圈回初始点坐椅子回满再继续 ----
    def is_resting(self):
        """是否处于休息状态(导航回点1/坐椅子/等待回满)。"""
        return self._resting

    def maybe_rest(self, map_norm, now, hp_full=False, mp_full=False):
        """主循环调用: 完成 N 圈后进入休息, 或休息中继续处理。

        返回 (command, reason) 或 None(不休息, 正常巡航)。
        休息流程: goto(导航回点1) -> sit(按x坐椅子) -> wait(等待回满) -> 完成清计数。
        """
        if not self._patrolling or not map_norm:
            return None
        # 未到休息圈数 -> 正常巡航
        if not self._resting and self._round_count < self.rest_every_rounds:
            return None
        nx, ny = float(map_norm[0]), float(map_norm[1])
        # 首次触发: 进入休息
        if not self._resting:
            self._resting = True
            self._rest_state = "goto"
            logger.info(
                f"[wp] 已巡航 {self._round_count} 圈, 触发休息: 回初始点坐椅子回满")
        # 目标 = 第一个点
        p0 = self.waypoints[0]
        tnx, tny = float(p0["nx"]), float(p0["ny"])
        dx, dy = tnx - nx, tny - ny
        # 1) goto: 走到点1(水平走位, dx 决定方向)
        if self._rest_state == "goto":
            if abs(dx) > self.arrive_x or abs(dy) > self.arrive_y:
                if abs(dx) <= 0.025:
                    return ("step_left" if dx < 0 else "step_right"), "wp_rest_goto"
                return ("move_left" if dx < 0 else "move_right"), "wp_rest_goto"
            self._rest_state = "sit_wait"
            self._rest_start = now
            logger.info("[wp] 已到达初始点, 等 3s 站稳后再坐椅子")
        # 2) sit_wait: 到达后等待 rest_sit_delay(默认3s) 让角色完全站稳
        #    (刚停下就按 X 会坐不上椅子), 期间不打怪不移动
        if self._rest_state == "sit_wait":
            if now - self._rest_start < self.rest_sit_delay:
                return "none", "wp_rest_sit_wait"
            self._rest_state = "sit"
            logger.info(f"[wp] 站稳完成, 按 x 坐椅子")
        # 3) sit: 按 x 坐椅子(按下保持一小段触发坐下)
        if self._rest_state == "sit":
            self._rest_state = "wait"
            self._rest_start = now
            return f"press_{self.rest_sit_key}", "wp_rest_sit"
        # 4) wait: 等待回满(HP 和 MP 都满), 超时兜底(最多 60s)
        if self._rest_state == "wait":
            if hp_full and mp_full:
                logger.info(
                    f"[wp] 状态回满, 结束休息, 继续巡航 "
                    f"(总圈数={self._round_count})")
                self._resting = False
                self._rest_state = ""
                self._round_count = 0
                return "none", "wp_rest_done"
            if now - self._rest_start > 60.0:
                logger.warning(
                    f"[wp] 休息 60s 仍未回满(HP满={hp_full} MP满={mp_full}), "
                    f"强制继续巡航")
                self._resting = False
                self._rest_state = ""
                self._round_count = 0
                return "none", "wp_rest_done"
            return "none", "wp_rest_wait"
        return None

    def cancel_rest(self):
        """手动停止巡航/清空时清除休息状态。"""
        self._resting = False
        self._rest_state = ""
        self._round_count = 0


# --------------------------------------------------------------------------
# Built-in route recorder: records the player's walk directly inside the
# auto_combat window (no separate routeRecorder window needed).
# 功能键(主循环处理): F1 开始录制 / F2 结束录制 / F3 开始路线巡航 / F4 清除路线.
# Reuses the same color-code config as routeRecorder so the saved routes are
# immediately replayable by RouteFollower.
# --------------------------------------------------------------------------
class RouteRecorderCore:
    """Minimal route recorder embedded in the auto_combat loop.

    It reuses the RouteFollower's loaded global map (img_map) as the base
    canvas, draws color-coded pixels from the user's held keys + the player's
    global position, and saves route*.png / map.png on F3 / F4.
    """

    def __init__(self, cfg, route_follower):
        self.cfg = cfg
        self.follower = route_follower
        route_cfg = cfg["route"]

        def _to_bgr(rgb_key):
            r, g, b = (int(c) for c in rgb_key.split(","))
            return (b, g, r)

        # action -> BGR color (reverse of color_code)
        self.action_to_bgr = {
            value: _to_bgr(key)
            for key, value in route_cfg["color_code"].items()
        }
        self.action_to_bgr.update({
            value: _to_bgr(key)
            for key, value in route_cfg.get("color_code_up_down", {}).items()
        })
        self.is_recording = False  # 启动不自动录制, 用户按 F1 手动开始
        self.img_route = None
        self.loc_last = None
        self.t_last_blob = 0.0
        self.blob_cooldown = float(
            cfg["route_recoder"]["blob_cooldown"])
        self.map_dir = None
        self.route_count = 0
        self._saved_flash_until = 0.0

    # -- map dir / counters ------------------------------------------------
    def _ensure_map_dir(self, map_name):
        if not map_name:
            return None
        self.map_dir = os.path.join("minimaps", map_name)
        os.makedirs(self.map_dir, exist_ok=True)
        return self.map_dir

    def _count_routes(self):
        if not self.map_dir or not os.path.isdir(self.map_dir):
            return 0
        n = 0
        for name in os.listdir(self.map_dir):
            if name.startswith("route") and name.endswith(".png") \
                    and "rest" not in name:
                n += 1
        return n

    def _init_route_image(self):
        """Start a fresh route canvas from the follower's global map."""
        if self.follower.img_map is None:
            return False
        self.img_route = self.follower.img_map.copy()
        # Route images must not keep map pixels that already equal a color
        # code (they would be mis-read as commands when replayed). Black them.
        for bgr in self.action_to_bgr.values():
            lo = np.clip(np.array(bgr, dtype=np.int16) - 6, 0, 255).astype(np.uint8)
            hi = np.clip(np.array(bgr, dtype=np.int16) + 6, 0, 255).astype(np.uint8)
            m = cv2.inRange(self.img_route, lo, hi)
            self.img_route[m > 0] = 0
        self.route_count = self._count_routes()
        self.loc_last = None
        return True

    # -- action from held keys ---------------------------------------------
    def _action_from_keys(self, held):
        """Map the user's currently-held keys to a route action string."""
        if "space" in held:
            if "left" in held:
                return "left none jump"
            if "right" in held:
                return "right none jump"
            if "down" in held:
                return "none down jump"
            return "none none jump"
        if "e" in held:
            if "left" in held:
                return "left none teleport"
            if "right" in held:
                return "right none teleport"
            if "down" in held:
                return "none down teleport"
            if "up" in held:
                return "none up teleport"
            return "none none teleport"
        if "up" in held:
            return "none up none"
        if "down" in held:
            return "none down none"
        if "left" in held:
            return "left none none"
        if "right" in held:
            return "right none none"
        return ""

    # -- per-frame update ----------------------------------------------------
    def update(self, gxy, held_keys, now):
        """Record one frame. gxy: player global pos; held_keys: set of keys."""
        if not self.is_recording or gxy is None:
            return
        if self.img_route is None:
            if not self._init_route_image():
                return
        action = self._action_from_keys(held_keys)
        px, py = int(gxy[0]), int(gxy[1])
        if action == "":
            # Fall back to displacement-based walk detection for short taps.
            if self.loc_last is not None:
                dx = px - self.loc_last[0]
                dy = py - self.loc_last[1]
                if 8.0 <= float(np.hypot(dx, dy)) <= 80.0 and abs(dx) >= abs(dy) * 1.5:
                    action = "right none none" if dx > 0 else "left none none"
            if action == "":
                self.loc_last = (px, py)
                return
        color_bgr = self.action_to_bgr.get(action)
        if color_bgr is None:
            self.loc_last = (px, py)
            return
        is_blob = action.endswith("jump") or action.endswith("teleport") \
            or action.endswith("goal")
        if is_blob:
            if now - self.t_last_blob >= self.blob_cooldown:
                cv2.circle(self.img_route, (px, py), 2, color_bgr, -1)
                self.t_last_blob = now
                self.loc_last = None
            return
        if self.loc_last is not None:
            cv2.line(self.img_route, self.loc_last, (px, py), color_bgr, 1)
        self.loc_last = (px, py)

    # -- save ----------------------------------------------------------------
    def save_route(self, map_name, route_name=None):
        if self.img_route is None or not self._ensure_map_dir(map_name):
            return False
        # 多路线命名: 自定义名优先, 否则用日期时间戳(保存日期)
        if route_name:
            safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(route_name))
            fname = f"route_{safe}.png"
        else:
            fname = f"route_{time.strftime('%Y%m%d_%H%M%S')}.png"
        out = os.path.join(self.map_dir, fname)
        ok, buf = cv2.imencode(".png", self.img_route)
        if not ok:
            return False
        with open(out, "wb") as f:
            f.write(buf.tobytes())
        self._saved_flash_until = time.time() + 2.0
        self.last_saved_path = out
        # Reset canvas for the next recording.
        self.img_route = self.follower.img_map.copy()
        self.loc_last = None
        return True

    def reset_recording(self):
        """清空当前录制, 重新开始(取消重录, F2)。"""
        if self.follower.img_map is None:
            return False
        self.img_route = self.follower.img_map.copy()
        self.loc_last = None
        self.route_count = self._count_routes()
        return True

    def save_map(self, map_name):
        if self.follower.img_map is None or not self._ensure_map_dir(map_name):
            return False
        out = os.path.join(self.map_dir, "map.png")
        ok, buf = cv2.imencode(".png", self.follower.img_map)
        if not ok:
            return False
        with open(out, "wb") as f:
            f.write(buf.tobytes())
        return True

    def route_pixel_count(self):
        if self.img_route is None or self.follower.img_map is None:
            return 0
        h, w = self.img_route.shape[:2]
        if h == 0 or w == 0:
            return 0
        diff = cv2.absdiff(self.img_route, self.follower.img_map[:h, :w])
        return int(np.count_nonzero(np.any(diff > 20, axis=2)))


# because the masks are small (O(width * height) on a downscaled ROI).
# --------------------------------------------------------------------------
class TerrainScanner:
    """Detect ropes and drop-offs/lifts that the bot should react to."""

    # MapleStory ropes are a warm brown, low-value vertical strip. The exact
    # hue varies a bit with the background behind them, so we use a generous
    # HSV range and then size/aspect filtering to separate them from any
    # randomly brown background.
    ROPE_HSV_LOW = (5, 35, 25)
    ROPE_HSV_HIGH = (40, 220, 130)
    MIN_ROPE_LENGTH = 50       # px tall
    MIN_ROPE_WIDTH = 3
    MAX_ROPE_WIDTH = 14
    ASPECT_RATIO_MIN = 4.0     # height / width

    # The player's "ground" is a bright low-saturation greyish-yellow. A
    # drop-off shows up as a sharp horizontal edge where the ground color
    # ends. The exact pixel values vary per map, so we use V/S bounds rather
    # than a tight hue.
    PLATFORM_V_LOW = 180
    PLATFORM_S_MAX = 80        # bright greyish = low saturation
    EDGE_GRADIENT_THRESHOLD = 60

    def __init__(self):
        self._cache = None
        self._cache_at = 0.0
        self._cache_ttl = 0.4  # seconds; the world does not change frame-by-frame

    def scan(self, frame, now, player_center):
        """Return a dict with detected ropes and platform edges near the player.

        Returns:
            {"ropes": [(x, y_top, y_bot, x_center), ...], "drops": [...]}
        """
        if (
            self._cache is not None
            and now - self._cache_at < self._cache_ttl
            and self._cache.get("player_x") == player_center[0]
        ):
            return self._cache
        masks = self._build_masks(frame)
        ropes = self._find_ropes(masks)
        drops = self._find_drops(masks, player_center)
        result = {
            "ropes": ropes,
            "drops": drops,
            "player_x": player_center[0],
            "player_y": player_center[1],
        }
        self._cache = result
        self._cache_at = now
        return result

    def _build_masks(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rope_mask = cv2.inRange(
            hsv, np.array(self.ROPE_HSV_LOW), np.array(self.ROPE_HSV_HIGH)
        )
        # Drop isolated brown speckle without flattening the actual ropes.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        rope_mask = cv2.morphologyEx(rope_mask, cv2.MORPH_OPEN, kernel_open)
        # Join vertical gaps so a rope with a knot or pixel drop still looks
        # like one continuous strip.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rope_mask = cv2.morphologyEx(rope_mask, cv2.MORPH_CLOSE, kernel_close)
        # Ground: bright, low saturation. Looks like flat platforms.
        h, s, v = cv2.split(hsv)
        ground_mask = ((v >= self.PLATFORM_V_LOW) & (s <= self.PLATFORM_S_MAX)).astype(np.uint8) * 255
        return {"rope": rope_mask, "ground": ground_mask}

    def _find_ropes(self, masks):
        """Find ropes: long, narrow, vertical brown strips surrounded by air.

        `masks['rope']` is the brown mask. To distinguish a real rope from a
        cliff edge or tree, we require the bounding box to be narrow + tall
        AND the pixel density inside the box to be sparse (a rope is thin so
        the box around it is mostly air), AND the brown must not extend
        laterally beyond the box edges (which would mean a cliff face).
        """
        mask = masks["rope"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = mask.shape
        ropes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (self.MIN_ROPE_WIDTH <= w <= self.MAX_ROPE_WIDTH):
                continue
            if h < self.MIN_ROPE_LENGTH:
                continue
            if h / max(1, w) < self.ASPECT_RATIO_MIN:
                continue
            # Internal density: a rope is thin so the box is mostly air.
            box_area = w * h
            pixel_count = cv2.contourArea(contour)
            if box_area <= 0:
                continue
            density = pixel_count / box_area
            if density > 0.55:
                # Too solid: looks like a cliff face, not a rope.
                continue
            # There's a rope only if the bounding box is in the air around it:
            # directly left and right of the box the brown should not extend.
            xl = max(0, x - 3)
            xr = min(w_img, x + w + 3)
            surrounding = mask[max(0, y - 2):min(h_img, y + h + 2), xl:xr]
            if surrounding.size == 0:
                continue
            sur_density = surrounding.mean() / 255.0
            if sur_density > 0.5:
                # Surrounded by brown = a cliff/tree, not a rope.
                continue
            ropes.append((x, y, y + h, x + w // 2))
        return ropes

    def _find_drops(self, masks, player_center):
        ground = masks["ground"]
        # Look at a horizontal band just below the player's feet: a true
        # platform extends flat across; a drop-off has the ground end abruptly.
        y_band = int(max(0, player_center[1] + 30))
        band = ground[y_band:y_band + 3, :]
        if band.size == 0:
            return []
        # Per-column: is ground present?
        col_has_ground = band.any(axis=0)
        # Run-length encode to find ground segments.
        segments = []
        start = None
        for x, present in enumerate(col_has_ground):
            if present and start is None:
                start = x
            elif not present and start is not None:
                segments.append((start, x))
                start = None
        if start is not None:
            segments.append((start, len(col_has_ground)))
        return segments


# --------------------------------------------------------------------------
# Async monster detector: run the slow YOLO inference in a background thread
# so the main loop never blocks on it. The main loop just submits frames and
# reads the latest result, keeping the frame rate stable even while moving.
# --------------------------------------------------------------------------
class AsyncMonsterDetector:
    def __init__(self, detector, max_age=2.0, min_interval=0.25):
        self.detector = detector
        self.max_age = float(max_age)
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._frame = None
        self._player = None
        self._latest = []
        self._timestamp = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit_frame(self, frame, player=None):
        with self._lock:
            self._frame = frame
            if player is not None:
                self._player = player

    def get_latest(self, now):
        with self._lock:
            if now - self._timestamp <= self.max_age:
                return list(self._latest)
            return []

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                player = self._player
            if frame is None:
                time.sleep(0.01)
                continue
            started = time.time()
            try:
                # Pass the latest player box so band-restricted detectors
                # (TemplateMonsterDetector / SpriteMonsterDetector) only scan
                # the player's horizontal band instead of the full frame.
                # Without this the background thread did a full-frame match
                # every 0.25s, pinning the CPU.
                monsters = self.detector.detect(frame, player)
                with self._lock:
                    self._latest = monsters
                    self._timestamp = time.time()
            except Exception:
                pass
            elapsed = time.time() - started
            # Keep GPU from spinning: pace inferences so we refresh at most
            # every min_interval seconds (e.g. 4x/sec) while leaving CPU/GPU
            # headroom for the perception loop.
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)


# --------------------------------------------------------------------------
# Pause control: global hotkey toggles a thread-safe pause flag.
# --------------------------------------------------------------------------
class PauseController:
    """Global hotkey hub.

    Besides pause/quit it also:
      - exposes a queue of function-key presses (F1/F3/F4) for in-window
        route recording (no separate routeRecorder window needed),
      - tracks which movement keys are currently held down so the built-in
        route recorder can draw color-coded actions from the user's input.
    """

    def __init__(self, pause_key="f8", quit_key="f9"):
        self.pause_key = pause_key.lower()
        self.quit_key = quit_key.lower()
        self.paused = True   # 启动默认暂停, 用户按 F8 手动启动打怪
        self.player_pause = False  # 检测到其他玩家(中级冒险家勋章) -> 强制暂停挂机
        self.quit_requested = False
        self._lock = threading.Lock()
        self.fn_events = []          # list of function-key names (f1/f3/f4)
        self.held_keys = set()       # currently held keys (left/right/up/down/space/e)
        self.bind_target = None      # None, or the field being key-bound
        self.bind_captured = None    # key name captured during bind mode

    def toggle(self):
        with self._lock:
            self.paused = not self.paused
        return self.paused

    def is_effectively_paused(self):
        """是否应暂停: 手动暂停 或 检测到其他玩家强制暂停。"""
        with self._lock:
            return self.paused or self.player_pause

    def set_player_pause(self, value):
        with self._lock:
            self.player_pause = bool(value)

    def resume_from_player_pause(self):
        """F8 恢复: 退出"检测到其他玩家"暂停, 重新开始挂机。"""
        with self._lock:
            self.player_pause = False
            self.paused = False

    def request_quit(self):
        with self._lock:
            self.quit_requested = True

    def is_quit_requested(self):
        with self._lock:
            return self.quit_requested

    def pop_fn_event(self):
        with self._lock:
            if self.fn_events:
                return self.fn_events.pop(0)
            return None

    def held_key_set(self):
        with self._lock:
            return set(self.held_keys)

    def start_bind(self, target):
        """Enter key-binding mode: the next pressed key becomes the value."""
        with self._lock:
            self.bind_target = target
            self.bind_captured = None

    def cancel_bind(self):
        with self._lock:
            self.bind_target = None
            self.bind_captured = None

    def pop_bind_result(self):
        """Returns (target, key) if a key was captured, else None."""
        with self._lock:
            if self.bind_captured is not None and self.bind_target is not None:
                result = (self.bind_target, self.bind_captured)
                self.bind_target = None
                self.bind_captured = None
                return result
            return None

    def is_binding(self):
        with self._lock:
            return self.bind_target is not None

    def start_listener(self):
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    name = key.name if hasattr(key, "name") else str(key)
                except Exception:
                    name = str(key)
                name = name.lower()
                if name == self.pause_key:
                    # F8: 若因"检测到其他玩家"而暂停 -> 恢复挂机(不清空航点);
                    # 否则正常暂停/继续切换。
                    if self.player_pause:
                        self.resume_from_player_pause()
                        print("[hotkey] F8 恢复挂机(玩家已离开/手动恢复)", flush=True)
                    else:
                        state = self.toggle()
                        print(f"[hotkey] paused={state}", flush=True)
                elif name == self.quit_key:
                    self.request_quit()
                    print("[hotkey] quit requested", flush=True)
                elif name in ("f1", "f2", "f3", "f4", "f6", "f10", "f11"):
                    with self._lock:
                        self.fn_events.append(name)
                else:
                    with self._lock:
                        self.held_keys.add(name)
                        if self.bind_target is not None and name not in ("shift", "esc"):
                            # Do not capture modifiers alone.
                            if name not in ("ctrl", "alt", "shift"):
                                self.bind_captured = name

            def on_release(key):
                try:
                    name = key.name if hasattr(key, "name") else str(key)
                except Exception:
                    name = str(key)
                with self._lock:
                    self.held_keys.discard(name.lower())

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.daemon = True
            listener.start()
            return listener
        except Exception as exc:
            print(f"[hotkey] listener unavailable: {exc}", flush=True)
            return None


def draw_coordinate_grid(frame, step=50, major=100):
    """Lightweight debug coordinate overlay.

    Draws a thin grid every `step` px, with a heavier line + axis label
    every `major` px, and a coordinate readout in the top-left. Lets the
    user describe visual offsets in real px ("move up 20", "left 50")
    instead of guessing from pixels.
    """
    h, w = frame.shape[:2]
    minor_color = (50, 50, 50)
    major_color = (90, 90, 90)
    label_color = (180, 220, 255)
    for x in range(0, w, step):
        col = major_color if x % major == 0 else minor_color
        cv2.line(frame, (x, 0), (x, h), col, 1)
    for y in range(0, h, step):
        col = major_color if y % major == 0 else minor_color
        cv2.line(frame, (0, y), (w, y), col, 1)
    # axis labels along the top (X) and left edge (Y), one per major step
    for x in range(0, w, major):
        cv2.putText(frame, str(x), (x + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
    for y in range(major, h, major):
        cv2.putText(frame, str(y), (2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
    # corner legend so the user knows what they are looking at
    put_text_cn(
        frame,
        f"坐标网格 {step}/{major}px  (报数: 上移=减Y, 左移=减X)",
        (8, h - 8),
        0.45,
        label_color,
        1,
        cv2.LINE_AA,
    )


def draw_attack_range(frame, player, policy):
    """Draw the per-character attack range box around the player."""
    if player is None or policy is None:
        return
    center = player.get("center")
    if not center:
        return
    x, y = int(center[0]), int(center[1])
    half_w = int(policy.attack_horizontal_px)
    half_h = int(policy.attack_vertical_px)
    top = max(0, y - half_h)
    bottom = min(frame.shape[0] - 1, y + half_h)
    left = max(0, x - half_w)
    right = min(frame.shape[1] - 1, x + half_w)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 3)
    cv2.line(frame, (x - 6, y), (x + 6, y), (0, 255, 255), 1)
    cv2.line(frame, (x, y - 6), (x, y + 6), (0, 255, 255), 1)
    put_text_cn(
        frame,
        f"攻击范围 {half_w}x{half_h}",
        (left + 4, max(16, top - 4)),
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # 【巡游打怪绿色范围】: 以玩家为中心横向 ±patrol_hunt_range_px(默认300),
    # 此范围内有怪 -> 离开航点去追怪消灭(见 decide 的 hunt_* 分支)。
    _hunt_w = int(policy.patrol_hunt_range_px)
    _hunt_top = max(0, y - half_h)
    _hunt_bottom = min(frame.shape[0] - 1, y + half_h)
    _hunt_left = max(0, x - _hunt_w)
    _hunt_right = min(frame.shape[1] - 1, x + _hunt_w)
    cv2.rectangle(frame, (_hunt_left, _hunt_top),
                  (_hunt_right, _hunt_bottom), (0, 255, 0), 1)
    put_text_cn(
        frame,
        f"巡游打怪 ±{_hunt_w}",
        (_hunt_right - 8 - len(f"巡游打怪 ±{_hunt_w}") * 8, _hunt_bottom - 4),
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


MONSTER_COLORS = {
    "STUMP": (0, 80, 255),
    "RED SNAIL": (80, 80, 255),
    "BLUE SNAIL": (255, 180, 0),
    "SLIME": (180, 255, 120),
    "GREEN MUSHROOM": (60, 200, 60),
    "FLOWER MUSHROOM": (255, 100, 200),
    "ORANGE MUSHROOM": (0, 140, 255),
    "ZOMBIE MUSHROOM": (30, 170, 255),
    "THORN MUSHROOM": (255, 180, 40),
    "MOB": (200, 200, 200),
    "MONSTER": (200, 200, 200),
}


def draw_detections(frame, player, monsters, advisory):
    """画玩家框、怪物框、实体坐标(中心点+速度箭头+运动状态)和坐标面板。"""
    player_color = (0, 255, 255)

    def arrow(img, cx, cy, vx, vy, color):
        if abs(vx) < 3 and abs(vy) < 3:
            return
        mag = float(np.hypot(vx, vy))
        ux, uy = vx / mag, vy / mag
        ex = int(cx + ux * 26)
        ey = int(cy + uy * 26)
        cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.35)

    if player is not None:
        box = player.get("box")
        if box:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), player_color, 2)
            cx, cy = int(player["center"][0]), int(player["center"][1])
            cv2.circle(frame, (cx, cy), 4, player_color, -1)
            arrow(frame, cx, cy, player.get("velocity_px_s", [0, 0])[0],
                  player.get("velocity_px_s", [0, 0])[1], player_color)
            ms = player.get("motion_state", "")
            cv2.putText(frame, f"P1 PLAYER [{ms}]", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, player_color, 1, cv2.LINE_AA)
    for monster in monsters:
        box = monster.get("box")
        if not box:
            continue
        x, y, w, h = (int(v) for v in box)
        color = MONSTER_COLORS.get(monster.get("label"), (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cx, cy = int(x + w / 2), int(y + h / 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        arrow(frame, cx, cy, monster.get("velocity_px_s", [0, 0])[0],
              monster.get("velocity_px_s", [0, 0])[1], color)
        eid = monster.get("entity_id", "")
        ms = monster.get("motion_state", "")
        rel = monster.get("relative_to_player")
        dist = f" d={rel['distance_px']}" if rel else ""
        pr = " P" if monster.get("tracking_state") == "PREDICTED" else ""
        cv2.putText(frame, f"{eid} {monster.get('label','?')} {monster.get('score',0):.2f} [{ms}]{dist}{pr}",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if advisory and advisory.get("attack_ready") and advisory.get("target_box") and player is not None:
        # 目标锁定线只在怪处于攻击范围内(attack_ready)时画——范围外的怪(旧检测/
        # 远处)不画线, 避免绿线连到空气/远处目标误导
        t = advisory["target_box"]
        tx = int(t[0] + t[2] / 2)
        ty = int(t[1] + t[3] / 2)
        pc = player.get("center")
        if pc:
            px, py = int(pc[0]), int(pc[1])
            color = (50, 70, 255) if advisory.get("dodge_risk") else (80, 220, 80)
            cv2.line(frame, (px, py), (tx, ty), color, 1)
    # 坐标面板已移除: 覆盖游戏画面右侧, 信息与右侧状态面板重复


def draw_coordinate_panel(frame, player, monsters):
    """右侧实体坐标面板(codex 风格 ENTITY COORDINATES)。"""
    h = frame.shape[0]
    pw = 200
    panel = np.full((h, pw, 3), (22, 22, 28), np.uint8)
    cv2.putText(panel, "ENTITY COORDINATES", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
    y = 44
    entities = ([] if player is None else [player]) + list(monsters)
    for e in entities[:12]:
        eid = e.get("entity_id", "?")
        label = e.get("label", "PLAYER")
        ms = e.get("motion_state", "")
        pr = " PRED" if e.get("tracking_state") == "PREDICTED" else ""
        color = (0, 255, 255) if e.get("entity_id") == "P1" else MONSTER_COLORS.get(label, (0, 255, 0))
        cv2.circle(panel, (14, y - 3), 4, color, -1)
        cv2.putText(panel, f"{eid} {label} {ms}{pr}", (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 16
        cp = e.get("center_px")
        v = e.get("velocity_px_s")
        sp = e.get("speed_px_s")
        line2 = f"  xy=({cp[0]},{cp[1]}) v=({v[0]},{v[1]}) {sp}px/s" if cp is not None and v is not None else ""
        cv2.putText(panel, line2, (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y += 16
        rel = e.get("relative_to_player")
        if rel:
            cv2.putText(panel, f"  rel=({rel['delta_px'][0]},{rel['delta_px'][1]}) d={rel['distance_px']}", (26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 120), 1, cv2.LINE_AA)
            y += 16
        y += 2
    frame[0:h, frame.shape[1] - pw:frame.shape[1]] = panel


def draw_route_overlay(frame, minimap_scan, route_follower, player_xy):
    """Draw the loaded route color-codes inside the minimap area of the frame.

    The route lives in global-map coordinates; we project it back into the
    minimap crop region so the user can see the recorded path on the small
    minimap (and how close the player is to it). Pure visualization.
    Also draws the player's actual walked trail (route_follower.trail, a list
    of global-map points) as a bright line, so you can compare what the bot
    did vs. the recorded route.
    """
    if route_follower is None or not getattr(route_follower, "img_routes", None):
        return
    if minimap_scan is None or not minimap_scan.get("player_xy"):
        return
    try:
        region = minimap_scan["region"]
        x0, y0, w, h = (int(v) for v in region)
        if w <= 0 or h <= 0:
            return
        img_map = route_follower.img_map
        route = route_follower.img_routes[route_follower.idx_route]
        if img_map is None:
            return
        # Locate the minimap inside the global map (same match as locate_player).
        from src.utils.common import find_pattern_sqdiff
        minimap_img = minimap_scan.get("_minimap_img")
        if minimap_img is None:
            return
        loc, _, _ = find_pattern_sqdiff(img_map, minimap_img)
        if loc is None:
            return
        # Draw the minimap crop (zoomed) into the top-left corner.
        mm_crop = frame[y0:y0 + h, x0:x0 + w].copy()
        # Semi-transparent dark backdrop so route colors pop.
        overlay = np.zeros_like(mm_crop)
        mh, mw = mm_crop.shape[:2]
        # Project route pixels that fall inside the minimap's global area.
        # Vectorized: build one mask over the minimap-sized global window.
        win_x0, win_y0 = loc[0], loc[1]
        win_w = min(img_map.shape[1] - win_x0, mw)
        win_h = min(img_map.shape[0] - win_y0, mh)
        if win_w > 0 and win_h > 0:
            win = route[win_y0:win_y0 + win_h, win_x0:win_x0 + win_w]
            nonblack = np.any(win != 0, axis=2)
            if nonblack.any():
                # Keep only pixels whose color maps to a known command.
                cmd_mask = np.zeros(nonblack.shape, dtype=bool)
                for bgr in route_follower.color_code:
                    lo = np.clip(np.array(bgr, dtype=np.int16) - 6, 0, 255).astype(np.uint8)
                    hi = np.clip(np.array(bgr, dtype=np.int16) + 6, 0, 255).astype(np.uint8)
                    cmd_mask |= cv2.inRange(win, lo, hi) > 0
                overlay[:win_h, :win_w][cmd_mask] = win[cmd_mask]
        # Player walked trail (global-map pts -> minimap-local pts).
        trail = getattr(route_follower, "trail", None)
        if trail:
            pts = [(gx - loc[0], gy - loc[1]) for (gx, gy) in trail
                   if 0 <= gx - loc[0] < mw and 0 <= gy - loc[1] < mh]
            if len(pts) >= 2:
                pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts_np], False, (255, 255, 255), 2)
            elif len(pts) == 1:
                cv2.circle(overlay, tuple(pts[0]), 3, (255, 255, 255), -1)
        # Player marker (red cross) at the minimap position.
        px = player_xy[0] - x0
        py = player_xy[1] - y0
        if 0 <= px < mw and 0 <= py < mh:
            cv2.line(overlay, (max(0, px - 4), py), (min(mw - 1, px + 4), py), (0, 0, 255), 1)
            cv2.line(overlay, (px, max(0, py - 4)), (px, min(mh - 1, py + 4)), (0, 0, 255), 1)
        # Blend: route colors on top of the minimap.
        vis = mm_crop.copy()
        vis[overlay > 0] = np.where(
            overlay[overlay > 0] != 0,
            overlay[overlay > 0], vis[overlay > 0])
        frame[y0:y0 + h, x0:x0 + w] = vis
    except Exception:
        pass


ROUTE_MAP_WINDOW = "Route Map (路线图)"
# Route color legend: BGR -> Chinese label. Drawn at the top of the route map
# window so the user can read what each colored pixel means.
_ROUTE_LEGEND = [
    ((0, 0, 255), "左走"), ((255, 0, 0), "右走"),
    ((0, 127, 255), "左跳"), ((255, 255, 0), "右跳"),
    ((0, 255, 127), "下跳"), ((255, 0, 255), "跳"),
    ((127, 255, 0), "停"), ((0, 255, 255), "终点"),
    ((127, 0, 255), "爬绳"), ((255, 0, 127), "下绳"),
    ((0, 127, 0), "左瞬移"), ((19, 69, 139), "右瞬移"),
    ((127, 127, 127), "↑"), ((127, 255, 255), "↓"),
]


def draw_route_map_window(frame, route_follower, recorder, minimap_scan,
                          player_xy, map_name=""):
    """Overlay the full route map as a live thumbnail on the MAIN window.

    Draws the global map with the color-coded recorded routes (walk / jump /
    climb / teleport each have their own color), the player's walked trail
    (white), the player's current position (red dot), and a small legend +
    the current route index. The thumbnail is placed in the bottom-right of
    the game frame so you always see WHICH route is being applied without
    opening a separate window.
    """
    if route_follower is None or route_follower.img_map is None:
        return
    try:
        base = route_follower.img_map
        h, w = base.shape[:2]
        canvas = base.copy()
        # Overlay ALL recorded routes (drawn dimmed), highlight the ACTIVE one.
        routes = getattr(route_follower, "img_routes", None) or []
        active = route_follower.idx_route if routes else -1
        for i, route in enumerate(routes):
            rh, rw = min(route.shape[0], h), min(route.shape[1], w)
            m = np.any(route[:rh, :rw] != 0, axis=2)
            if i == active:
                canvas[:rh, :rw][m] = route[:rh, :rw][m]
            else:
                # Dim inactive routes so the active one stands out.
                dim = (route[:rh, :rw][m].astype(np.int16) * 0.4).astype(np.uint8)
                canvas[:rh, :rw][m] = dim
        # Overlay the in-progress recording (live color pixels, bright).
        if recorder is not None and recorder.img_route is not None:
            rimg = recorder.img_route
            rh, rw = min(rimg.shape[0], h), min(rimg.shape[1], w)
            m = np.any(rimg[:rh, :rw] != 0, axis=2)
            canvas[:rh, :rw][m] = rimg[:rh, :rw][m]
        # Player walked trail (white polyline).
        trail = getattr(route_follower, "trail", None)
        if trail and len(trail) >= 2:
            pts = [(int(gx), int(gy)) for (gx, gy) in trail
                   if 0 <= int(gx) < w and 0 <= int(gy) < h]
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, (255, 255, 255), 2)
        # Player current position (red dot with ring).
        gxy = None
        if player_xy is not None and minimap_scan is not None:
            try:
                gxy = route_follower.locate_player(minimap_scan, player_xy)
            except Exception:
                gxy = None
        if gxy is not None:
            px, py = int(gxy[0]), int(gxy[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1)
        # Fit to a fixed thumbnail size (keep aspect).
        thumb_w, thumb_h = 340, 260
        scale = min(thumb_w / max(w, 1), thumb_h / max(h, 1))
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_NEAREST)
        # Legend + title strip on top.
        legend = np.full((30, thumb_w, 3), 26, dtype=np.uint8)
        title = f"路线图 {map_name or ''}  路线{active + 1}/{max(len(routes), 1)}"
        put_text_cn(legend, title, (8, 20), 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        # Place legend + thumb centered in the thumb_w box.
        out = np.full((legend.shape[0] + th, thumb_w, 3), 26, dtype=np.uint8)
        out[:30] = legend
        x_off = (thumb_w - tw) // 2
        out[30:30 + th, x_off:x_off + tw] = thumb
        # Paste onto the main frame bottom-right with a border.
        fh, fw = frame.shape[:2]
        oh, ow = out.shape[:2]
        px0, py0 = max(0, fw - ow - 12), max(0, fh - oh - 12)
        cv2.rectangle(frame, (px0 - 2, py0 - 2), (px0 + ow + 2, py0 + oh + 2),
                      (0, 180, 255), 2)
        frame[py0:py0 + oh, px0:px0 + ow] = out
    except Exception:
        pass


def _cmd_cn(command, reason):
    """Human-readable Chinese description of the current command/reason."""
    table = {
        "move_left": "← 向左走", "move_right": "→ 向右走",
        "climb_up": "↑ 爬绳", "climb_down": "↓ 下绳",
        "jump": "⤴ 原地跳", "jump_left": "⤴ 左跳", "jump_right": "⤴ 右跳",
        "jump_down": "⤵ 下跳",
        "attack_left": "⚔ 左攻击", "attack_right": "⚔ 右攻击",
        "dodge_left": "↶ 左闪避", "dodge_right": "↷ 右闪避",
        "none": "· 原地待命",
    }
    if command in table:
        return table[command]
    if command.startswith("attack"):
        return f"⚔ 攻击 ({command})"
    return command


def _reason_cn(reason):
    table = {
        "attack": "锁定目标攻击",
        "attack_cooldown": "攻击冷却中",
        "attack_cooldown_patrol": "冷却中继续巡游",
        "jump_out_of_pit": "跳坑脱困",
        "patrol_stuck_turn": "跳坑无效换向",
        "approach": "接近目标",
        "keep_distance": "保持距离",
        "dodge_imminent": "紧急闪避",
        "dodge_vertical": "垂直闪避",
        "patrol": "巡逻/寻怪",
        "route_left_none_none": "沿路线向左",
        "route_right_none_none": "沿路线向右",
        "route_none_up_none": "沿路线爬绳",
        "route_none_down_none": "沿路线下绳",
        "route_left_none_jump": "沿路线左跳",
        "route_right_none_jump": "沿路线右跳",
        "route_none_none_jump": "沿路线跳跃",
        "route_switch": "路线切换(到达目标点)",
        "player_missed": "未找到玩家",
        "no_target_box": "无目标",
        "no_advisory": "无战斗建议",
        "target_ignored": "目标已放弃",
        "paused": "已暂停",
        "not_foreground": "窗口未在前台",
    }
    return table.get(reason, reason)


def render_info_panel(
    frame,
    player,
    monsters,
    vitals,
    advisory,
    policy,
    executor,
    command,
    reason,
    fps_limit,
    measured_fps,
    paused,
    player_name,
    route_follower=None,
    recorder=None,
    map_name="",
    no_attack=False,
    editor=None,
    bind_target=None,
    exp_summary=None,
):
    """Draw a live status panel (frame rate, vitals, keys, thresholds, params,
    navigation state, route recording state) with clickable buttons.

    Returns (combined_image, buttons) where buttons is a list of
    (x0, y0, x1, y1, action) rectangles in combined-image coordinates, so the
    caller can hit-test mouse clicks.
    """
    h, w = frame.shape[:2]
    panel_w = 300
    panel = np.full((h, panel_w, 3), 24, dtype=np.uint8)
    buttons = []  # (x0, y0, x1, y1, action) in panel-local coords

    def button(x0, y0, x1, y1, label, action, fill=(55, 55, 66),
               border=(140, 140, 155), text_color=(235, 235, 235)):
        cv2.rectangle(panel, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), border, 1)
        tx = x0 + 8
        ty = y0 + (y1 - y0) // 2 + 6
        put_text_cn(panel, label, (tx, ty), 0.5, text_color, 1, cv2.LINE_AA)
        buttons.append((x0, y0, x1, y1, action))

    def section(title):
        nonlocal y
        y += 5
        cv2.rectangle(panel, (10, y - 10), (panel_w - 10, y - 8),
                      (60, 60, 70), -1)
        put_text_cn(panel, title, (14, y + 6), 0.6, (150, 210, 255), 2, cv2.LINE_AA)
        y += 22

    def text(msg, color=(225, 225, 225), size=0.55, weight=1):
        nonlocal y
        put_text_cn(panel, msg, (14, y + 8), size, color, weight, cv2.LINE_AA)
        y += 22

    def key_value(k, v, vcolor=(255, 255, 255)):
        nonlocal y
        put_text_cn(panel, k, (14, y + 8), 0.55, (170, 170, 180), 1, cv2.LINE_AA)
        # value right-aligned-ish at fixed column
        cv2.putText(panel, str(v), (200, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, vcolor, 2, cv2.LINE_AA)
        y += 22

    # ---- top status banner ------------------------------------------------
    status_color = (0, 0, 255) if paused else (
        (80, 220, 80) if reason in ("attack", "attack_cooldown", "attack_cooldown_patrol", "keep_distance")
        else (0, 200, 255))
    put_text_cn(panel, "● 运行中" if not paused else "Ⅱ 已暂停", (14, 24),
                0.72, status_color, 2, cv2.LINE_AA)
    put_text_cn(panel, _cmd_cn(command, reason), (150, 24), 0.68,
                (0, 255, 200), 2, cv2.LINE_AA)
    y = 46

    # ---- 经验效率(置顶: 面板内容多易截断, 放最上面保证可见) ----------------
    if exp_summary is not None:
        put_text_cn(panel, f"经验  {exp_summary['exp_per_hour']:.0f}/时 "
                     f"每轮{exp_summary['avg_exp']:.0f}({exp_summary['avg_duration']:.0f}s) "
                     f"{exp_summary['rounds']}轮",
                    (14, y + 8), 0.55, (60, 220, 60), 2, cv2.LINE_AA)
        y += 22

    # ---- top control buttons (mouse clickable) -----------------------------
    bw = (panel_w - 28 - 12) // 2  # two columns
    btn_h = 28
    # pause / record toggle
    button(14, y, 14 + bw, y + btn_h,
           "[暂停]" if not paused else "[继续]",
           "pause",
           fill=(0, 70, 130) if not paused else (0, 130, 70))
    button(14 + bw + 12, y, panel_w - 14, y + btn_h,
           "录制: 开" if recorder is None or recorder.is_recording else "录制: 关",
           "rec_toggle",
           fill=(140, 30, 30) if recorder is not None and recorder.is_recording else (55, 55, 66))
    y += btn_h + 8
    # save route / save map
    button(14, y, 14 + bw, y + btn_h, "保存路线 (F3)", "rec_save")
    button(14 + bw + 12, y, panel_w - 14, y + btn_h, "保存地图 (F4)", "rec_save_map")
    y += btn_h + 4

    if map_name:
        put_text_cn(panel, f"地图 {map_name}", (14, y + 10), 0.55,
                    (120, 200, 255), 1, cv2.LINE_AA)
        y += 24

    # ---- navigation --------------------------------------------------------
    section("导航状态")
    nav_cn = _reason_cn(reason)
    follow = bool(route_follower is not None and getattr(route_follower, "img_routes", None))
    if follow:
        text(f"路线跟随  开 ({len(route_follower.img_routes)} 条, 第{route_follower.idx_route + 1}条)",
             (0, 255, 160))
    else:
        text("路线跟随  未加载(巡逻兜底)", (200, 160, 80))
    # 小地图坐标航点巡逻状态
    _wp = getattr(policy, "_waypoint_patrol", None)
    if _wp is not None:
        if _wp.is_recording:
            text(f"航点录制  ● 录制中  已录 {len(_wp.waypoints)} 个", (80, 255, 120))
        elif _wp.is_patrolling():
            text(f"航点巡航  ● 巡航中 ({len(_wp.waypoints)} 个, 第{_wp.idx + 1}个)", (0, 255, 160))
        elif _wp.waypoints:
            text(f"航点待巡航  {len(_wp.waypoints)} 个  (按 F4 保存并巡航)", (0, 255, 160))
        else:
            text("航点  未录制(F1 启动录制)", (200, 160, 80))
        # 已录点位列表(手动打点: F2 普通点 / F3 跳跃点) —— 只显前6行, 防面板过长
        if _wp.waypoints:
            section("已录点位")
            _MAX_SHOW = 6
            _pts = _wp.waypoints
            for _pi, _p in enumerate(_pts[:_MAX_SHOW]):
                _tag = "步" if _p.get("action") == "move" else "跳"
                _col = (120, 230, 255) if _p.get("action") == "move" else (255, 200, 120)
                _mark = "▶" if (_wp.is_patrolling() and _pi == _wp.idx) else " "
                text(f"{_mark}{_pi + 1:>2}. [{_tag}] ({_p['nx']:.4f},{_p['ny']:.4f})",
                     _col, size=0.52)
            if len(_pts) > _MAX_SHOW:
                text(f"  …共 {len(_pts)} 个点位", (150, 150, 160), size=0.52)
    text(f"当前指令  {nav_cn}", (255, 235, 120))
    if command.startswith("move_") or "跳" in nav_cn:
        text(f"移动方向  {'← 左' if command.endswith('left') else '→ 右' if command.endswith('right') else _cmd_cn(command, reason)}",
             (120, 230, 255))
    if recorder is not None:
        rec_state = "● 录制中" if recorder.is_recording else "○ 已暂停录制"
        text(f"路线录制  {rec_state}  已录 {recorder.route_pixel_count()} px",
             (80, 255, 120) if recorder.is_recording else (180, 180, 180))
    if no_attack:
        text("打怪模式  已禁用(--no-attack)", (255, 150, 80))
    # 小地图稳定坐标(每帧由主循环写入 policy._mini): 反映玩家在地图里左右绝对位置
    if getattr(policy, "_mini", None) is not None and policy._mini.get("map_norm"):
        _mn = policy._mini["map_norm"]
        text(f"小地图位置  norm=({_mn[0]:.4f},{_mn[1]:.4f})", (0, 255, 255))

    # ---- vitals ------------------------------------------------------------
    section("状态与战斗")
    text(f"帧率  {measured_fps:5.1f} / {fps_limit} 上限", (0, 255, 255) if measured_fps >= fps_limit - 1 else (230, 200, 90))
    text(f"玩家  {'✓ 已找到' if player else '✗ 未找到'}" +
         (f" (置信{player.get('score', 0):.2f})" if player else ""),
         (100, 255, 150) if player else (100, 100, 255))
    text(f"怪物  检测到 {len(monsters)} 只", (220, 200, 120))
    # 实时角色坐标(大字显示, 供观察掉坑时 y 坐标变化)
    if player and player.get("center"):
        _pcx, _pcy = player["center"]
        text(f"角色坐标 X={int(_pcx)}  Y={int(_pcy)}",
             (0, 255, 255), size=0.78, weight=2)
    else:
        text("角色坐标 --", (120, 120, 120), size=0.78, weight=2)
    hp, mp, exp = vitals
    key_value("HP", f"{'--' if hp is None else f'{hp:5.1f}%'} (低于{executor.add_hp_threshold:.0f}%喝药)",
              (80, 80, 255))
    key_value("MP", f"{'--' if mp is None else f'{mp:5.1f}%'} (低于{executor.add_mp_threshold:.0f}%喝药)",
              (240, 140, 40))
    key_value("EXP", f"{'--' if exp is None else f'{exp:5.1f}%'}", (40, 210, 230))

    # ---- keys (click to rebind) ---------------------------------------------
    section("按键设置 (点击按钮后按任意键改键)")
    kbh = 30
    kcol_w = (panel_w - 28 - 2 * 10) // 3
    for idx, (label, val, action) in enumerate([
        ("攻击", executor.attack_key, "bind_attack"),
        ("喝HP", executor.add_hp_key or "-", "bind_hp"),
        ("喝MP", executor.add_mp_key or "-", "bind_mp"),
    ]):
        x0 = 14 + idx * (kcol_w + 10)
        active = (bind_target == action.replace("bind_", ""))
        button(x0, y, x0 + kcol_w, y + kbh, f"{label} [{val}]", action,
               fill=(0, 110, 150) if active else (55, 55, 66),
               border=(0, 230, 255) if active else (140, 140, 155))
    y += kbh + 8
    if bind_target:
        put_text_cn(panel,
                    f">> 正在绑定 [{bind_target}] 键, 请按任意键 (Esc取消)…",
                    (14, y + 8), 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24

    # ---- combat params (click to edit) --------------------------------------
    section("战斗参数 (点击按钮后输入数字回车)")
    param_defs = [
        ("范围横", f"{int(policy.attack_horizontal_px)}", "param_0"),
        ("范围纵", f"{int(policy.attack_vertical_px)}", "param_1"),
        ("保持距离", f"{int(policy.min_engage_px)}", "param_2"),
        ("HP阈值%", f"{executor.add_hp_threshold:.0f}", "param_3"),
        ("MP阈值%", f"{executor.add_mp_threshold:.0f}", "param_4"),
        ("帧率上限", f"{fps_limit}", "param_5"),
    ]
    pbh = 28
    pcol_w = (panel_w - 28 - 2 * 10) // 3
    for i, (plabel, pval, paction) in enumerate(param_defs):
        col, row = i % 3, i // 3
        x0 = 14 + col * (pcol_w + 10)
        y0 = y + row * (pbh + 6)
        active = editor is not None and editor.active and editor.editing_index == i
        button(x0, y0, x0 + pcol_w, y0 + pbh, f"{plabel} {pval}",
               paction,
               fill=(0, 110, 150) if active else (55, 55, 66),
               border=(0, 230, 255) if active else (140, 140, 155))
    y += 2 * (pbh + 6) + 4

    if advisory:
        text(f"目标  {advisory.get('target_label') or '-'}  距离 {advisory.get('distance_px')}px",
             (190, 190, 190), 0.52)

    # ---- hotkey legend -------------------------------------------------------
    section("操作图例")
    legend = [
        "F1 录制(路线+航点)  F2 清空  F3 保存  F4 存图  F7 存航点  F8 清航点",
        "F8 暂停  F9 退出; 面板按钮也可操作",
        "F6=重定位: 站到第1个点位按F6, 自动修正全部航点偏移",
        "F10=误按F1后恢复上次加载的点位",
    ]
    for line_ in legend:
        put_text_cn(panel, line_, (14, y + 6), 0.5, (140, 200, 140), 1, cv2.LINE_AA)
        y += 20
    # Active editing hint (from the key editor) so the user knows the
    # current binding/number-edit state.
    if editor is not None and editor.active:
        put_text_cn(panel, editor.status_line(), (14, y + 8), 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y += 22

    combined = np.hstack((frame, panel))
    # Shift button rects into combined-image coordinates.
    buttons_global = [(x0 + w, y0, x1 + w, y1, action)
                      for (x0, y0, x1, y1, action) in buttons]
    return combined, buttons_global


# --------------------------------------------------------------------------
# Keyboard parameter editor: pick a field with number keys, type a value,
# press Enter to apply. No sliders. Key-binding fields (bind=True) are bound
# by pressing the desired key directly after selecting the field.
# --------------------------------------------------------------------------
class ParamEditor:
    def __init__(self):
        self.fields = []   # list of dicts: key, label, getter, setter, minimum, bind
        self.editing_index = -1
        self.buffer = ""
        self.active = False

    def add(self, key, label, getter, setter, minimum=0, bind=False):
        self.fields.append(
            {
                "key": key,
                "label": label,
                "getter": getter,
                "setter": setter,
                "minimum": minimum,
                "bind": bind,
            }
        )

    def _current_field(self):
        if 0 <= self.editing_index < len(self.fields):
            return self.fields[self.editing_index]
        return None

    def handle_key(self, key_code):
        """Returns True if the key was consumed by the editor."""
        field = self._current_field() if self.active else None
        if key_code in (13, ord("\r"), ord("\n")):  # Enter -> apply
            if self.active and self.buffer:
                self._apply()
            self.active = False
            self.buffer = ""
            self.editing_index = -1
            return True
        if key_code in (27, 8):  # Esc / Backspace -> cancel
            self.active = False
            self.buffer = ""
            self.editing_index = -1
            return True
        if 48 <= key_code <= 57:  # digit 0-9
            if self.active and field and field.get("bind"):
                # Binding a key field: the digit pressed becomes the new key.
                self._apply_bind(chr(key_code))
                return True
            if self.active:
                self.buffer += chr(key_code)
                return True
            # Not editing yet: pick a field by its number key.
            for index, f in enumerate(self.fields):
                if f["key"] == key_code:
                    self.editing_index = index
                    self.active = True
                    self.buffer = ""
                    return True
            return False
        # Key-binding fields: any letter key while editing binds immediately;
        # pressing '-' clears the binding (e.g. warrior does not drink MP).
        if (
            self.active
            and field
            and field.get("bind")
            and (97 <= key_code <= 122 or key_code == 45)
        ):
            self._apply_bind("" if key_code == 45 else chr(key_code))
            return True
        return False

    def _apply(self):
        try:
            value = float(self.buffer)
        except ValueError:
            return
        field = self._current_field()
        if field is not None:
            field["setter"](max(field["minimum"], value))

    def _apply_bind(self, key_name):
        field = self._current_field()
        if field is not None:
            field["setter"](key_name)
        self.active = False
        self.buffer = ""
        self.editing_index = -1

    def status_line(self):
        if self.active and 0 <= self.editing_index < len(self.fields):
            field = self.fields[self.editing_index]
            if field.get("bind"):
                return f"> 正在绑定 [{field['label']}], 请直接按想用的键 (Esc取消)"
            return f"> 正在修改 [{field['label']}] = {self.buffer or '?'}_  (回车确认)"
        return "点击按钮或按数字键选中参数, 回车确认, Esc取消"

    def pick(self, index):
        """Activate a field by index (used by mouse clicks on the panel)."""
        if 0 <= index < len(self.fields):
            self.editing_index = index
            self.active = True
            self.buffer = ""

    def render(self, frame, start_y=70, line_h=22):
        y = start_y
        put_text_cn(
            frame,
            "实时参数 (按键类: 按数字选中后直接按新键; 数值类: 输入数字回车)",
            (18, y),
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y += line_h
        for index, field in enumerate(self.fields):
            value = field["getter"]()
            prefix = str(field["key"] - 48)
            highlight = index == self.editing_index and self.active
            color = (0, 255, 255) if highlight else (180, 180, 180)
            if field.get("bind"):
                text = f"{prefix}. {field['label']}: {value}"
            elif isinstance(value, float):
                text = f"{prefix}. {field['label']}: {value:.2f}"
            else:
                text = f"{prefix}. {field['label']}: {value}"
            put_text_cn(frame, text, (18, y), 0.5, color, 1, cv2.LINE_AA)
            y += line_h
        if self.active:
            put_text_cn(
                frame,
                self.status_line(),
                (18, y),
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


# --------------------------------------------------------------------------
# 经验统计(按轮): 以"跑完一轮录制点位"为单位, 记录每轮开始/结束的
# EXP 值数字(如 324625)和时间, 计算每轮经验、每轮耗时、每小时经验。
# 升级时 EXP 值会回绕(变回小数字), 检测到"大幅回落"则重新锚定起点。
# --------------------------------------------------------------------------
class ExpRoundStats:
    """按"完成一轮点位"统计 EXP 获取(主循环在每轮完成时调用 round_done)。"""

    MAX_ROUNDS = 20   # 保留最近 N 轮用于平均

    def __init__(self):
        self.rounds = []          # [{"duration", "exp_gain", "start_t", "end_t"}]
        # 当前轮起点(尚未完成): (起点时间, 起点EXP值)
        self._cur_start = None

    def anchor(self, exp_value, now):
        """首次锚定/升级后锚定: 若无当前轮起点(或上一轮已结算)则以当前
        (时间,EXP)为轮起点。已有进行中的轮则不覆盖。"""
        if exp_value is None:
            return
        if self._cur_start is None:
            self._cur_start = (now, float(exp_value))

    def round_done(self, exp_value, now):
        """完成一轮时调用: 传入当前 EXP 值数字, 结算本轮的耗时/经验。"""
        if exp_value is None:
            return None
        exp = float(exp_value)
        if self._cur_start is None:
            self._cur_start = (now, exp)
            return None
        start_t, start_exp = self._cur_start
        duration = max(0.01, now - start_t)
        gain = exp - start_exp
        # 升级回绕: EXP 大幅回落(升了级, 经验值清零重计) -> 本轮无效, 重新锚定
        if gain < -500000:
            self._cur_start = (now, exp)
            return None
        if gain < 0:
            gain = 0.0
        self.rounds.append({
            "duration": duration,
            "exp_gain": gain,
            "start_t": start_t,
            "end_t": now,
        })
        if len(self.rounds) > self.MAX_ROUNDS:
            self.rounds = self.rounds[-self.MAX_ROUNDS:]
        # 以完成点为新起点(下一轮)
        self._cur_start = (now, exp)
        return self._summary()

    def _summary(self):
        """返回 {rounds, avg_exp, avg_duration, exp_per_hour} 或 None(无数据)。"""
        if not self.rounds:
            return None
        total_gain = sum(r["exp_gain"] for r in self.rounds)
        total_duration = sum(r["duration"] for r in self.rounds)
        if total_duration <= 0:
            return None
        avg_exp = total_gain / len(self.rounds)
        avg_duration = total_duration / len(self.rounds)
        exp_per_hour = total_gain / total_duration * 3600.0
        return {
            "rounds": len(self.rounds),
            "avg_exp": avg_exp,
            "avg_duration": avg_duration,
            "exp_per_hour": exp_per_hour,
        }


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def load_config(name):
    cfg = load_yaml("config/config_default.yaml")
    return override_cfg(cfg, load_yaml(f"config/config_{name}.yaml"))


def target_is_foreground(window_title):
    import pygetwindow as gw

    active_window = gw.getActiveWindow()
    return bool(active_window and window_title in active_window.title)


def _acquire_instance_lock():
    """单实例锁(操作系统级文件锁): 已有 auto_combat 运行时本实例直接退出。

    用 msvcrt.locking 给锁文件加文件锁, 由操作系统持有:
    - 进程被杀/崩溃/正常退出时, 系统自动释放锁, 无残留。
    - 不依赖 PID 存活检查(venv 的 python.exe 是启动器, 会 spawn 真正的
      子进程, PID 记录对不上会导致锁失效——之前多开的根因)。
    """
    import msvcrt, tempfile
    # 锁文件放系统 temp(而非项目 log 目录): 项目目录里的旧锁文件可能被
    # 异常残留进程(如 codex runtime 的 pythonw 句柄)占用导致 WinError 32,
    # 新实例永远加不上锁("已有实例在运行"假象)。temp 目录每进程可写,
    # 锁由操作系统持有, 进程退出自动释放, 无残留。
    lock_path = Path(tempfile.gettempdir()) / "auto_combat_ms.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # 确保文件非空: msvcrt.locking 无法锁 0 字节文件(会静默失效)
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        lock_path.write_text("0", encoding="utf-8")
    lock_file = open(lock_path, "r+b")  # 二进制读写(非追加), 覆盖写 PID
    lock_file.seek(0)
    try:
        # 非阻塞尝试锁文件第 1 字节: 已被其他实例锁住则抛 OSError
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock_file.close()
        print("[auto_combat] 已有实例在运行, 本实例退出。")
        sys.exit(0)
    # 加锁成功: 覆盖写 PID 供人工排查(仅信息, 不作为锁依据)
    lock_file.seek(0)
    lock_file.write(str(os.getpid()).encode("utf-8"))
    lock_file.flush()
    return lock_file  # main 持有该文件对象引用, 锁在进程生命周期内持续有效


def main():
    _instance_lock = _acquire_instance_lock()
    parser = argparse.ArgumentParser(
        description="Closed-loop auto combat: perception -> advisory -> keyboard input."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--monster-backend", choices=("heuristic", "yolo", "template", "sprite", "color"),
        default="yolo",
    )
    parser.add_argument(
        "--collect-templates",
        action="store_true",
        help="Auto-collect new sprite templates from locked combat targets "
             "into monster_templates_final/ (learning mode).",
    )
    parser.add_argument(
        "--yolo-model",
        default="training_runs/maple_three_class_v1_balanced/weights/best.pt",
    )
    parser.add_argument("--yolo-confidence", type=float, default=None)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument(
        "--yolo-image-size",
        type=int,
        default=640,
        help="YOLO inference size (640 is much faster than 960; small monsters still detect fine).",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=None,
        help="YOLO NMS IoU 阈值(默认0.45; 野猪v4识别器用0.70)",
    )
    parser.add_argument(
        "--no-color-verify",
        action="store_true",
        help="Disable the HSV color verification that filters false positives.",
    )
    parser.add_argument("--motion-detection", action="store_true")
    parser.add_argument("--motion-threshold", type=int, default=22)
    parser.add_argument("--motion-min-area", type=int, default=45)
    parser.add_argument("--motion-max-area", type=int, default=3500)
    parser.add_argument("--motion-candidate-score", type=float, default=0.70)
    parser.add_argument("--foreground-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--summary")
    # 自动采集训练帧(F8启动巡游时自动采集玩家帧, 用于训练人物YOLO检测)
    parser.add_argument(
        "--capture-dir",
        default="training_data/player_capture_auto",
        help="自动采集玩家训练帧的保存目录(默认 training_data/player_capture_auto)",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=2.0,
        help="自动采集帧间隔秒(默认2.0)",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="关闭自动采集训练帧",
    )
    parser.add_argument(
        "--show-viz",
        action="store_true",
        help="Open a debug window that draws the attack range box around the player.",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Overlay a coordinate grid (50px minor / 100px major) so you can "
             "describe visual offsets in real px instead of guessing.",
    )
    parser.add_argument(
        "--pause-key",
        default="f8",
        help="Global hotkey to pause/resume the bot (default: f8).",
    )
    parser.add_argument(
        "--quit-key",
        default="f9",
        help="Global hotkey to stop the bot (default: f9).",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=10,
        help="Perception loop target FPS (default: 10).",
    )
    parser.add_argument(
        "--player-name",
        default="",
        help="If set, locate the player by reading this name via OCR (more reliable than the nametag template).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable the PaddleOCR player-location thread (it pins the CPU "
             "at 1500%+ while route-following). Uses only the offline "
             "player/nametag templates to track the player.",
    )
    parser.add_argument(
        "--map-name",
        default="",
        help="Explicit map name for route following, e.g. --map-name 射手训练场III. "
             "Bypasses minimap OCR (which can mis-read names like III -> 川) so "
             "minimaps/{map_name}/route*.png loads reliably.",
    )
    parser.add_argument(
        "--route-name",
        default="",
        help="Route name for saving/loading. Save (F3) writes route_{name}.png; "
             "if set at startup, load that specific route instead of the newest. "
             "Leave empty to auto-save with a date timestamp and load the newest.",
    )
    parser.add_argument(
        "--no-attack",
        action="store_true",
        help="Disable auto-attack entirely: the bot only follows the recorded "
             "route (or patrols) and never attacks monsters. Useful while "
             "monster detection is still being debugged.",
    )
    parser.add_argument(
        "--no-monster",
        action="store_true",
        help="关闭怪物检测(猪猪/YOLO): 仅巡游+受伤反击, 不主动打怪, 省GPU。用于单独验证小地图定位。",
    )
    parser.add_argument(
        "--monster-labels",
        nargs="+",
        default=None,
        help="Only load these monster classes for the template detector "
             "(e.g. --monster-labels red_snail green_mushroom slime). Cuts "
             "CPU by matching only the map's few mobs instead of all of "
             "monster/.",
    )
    # Key bindings & attack range overrides (e.g. switching to a warrior whose
    # attack key / potion keys / melee range differ from the mage defaults).
    parser.add_argument(
        "--attack-key",
        default=None,
        help="Override the attack key (e.g. --attack-key j). Default comes from the yaml 'key.directional_attack'.",
    )
    parser.add_argument(
        "--add-hp-key",
        default=None,
        help="Override the HP potion key (e.g. --add-hp-key s). Default from yaml 'key.add_hp'.",
    )
    parser.add_argument(
        "--add-mp-key",
        default=None,
        help="Override the MP potion key (e.g. --add-mp-key w). Default from yaml 'key.add_mp'.",
    )
    parser.add_argument(
        "--attack-horizontal",
        type=float,
        default=None,
        help="Override attack horizontal range in px (warrior melee is much smaller than the mage's 260).",
    )
    parser.add_argument(
        "--attack-vertical",
        type=float,
        default=None,
        help="Override attack vertical range in px.",
    )
    parser.add_argument(
        "--keep-distance",
        type=float,
        default=None,
        help="Override the minimum engagement distance in px (melee classes use a small value).",
    )
    parser.add_argument(
        "--mode",
        default="normal",
        choices=("normal", "stationary", "patrol_hunt", "minimap_patrol"),
        help="挂机模式: normal=移动打怪; stationary=站桩打怪; patrol_hunt=巡游打怪(保持巡游方向, 攻击黄圈内有怪才攻击, 不追远处怪); minimap_patrol=小地图坐标航点巡逻(按录制的小地图坐标精确走位/跳跃/爬绳, 适合大量跳跃爬绳点的地图)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不记录决策/状态日志(干净版本, 减少磁盘占用)",
    )
    parser.add_argument(
        "--no-terrain",
        action="store_true",
        help="禁用地形导航(爬绳/平台), 巡游只左右走+跳坑",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    overlay_cfg = cfg["perception_overlay"]
    auto_cfg = cfg.get("auto_combat", {})
    # Multi-character support: apply the character profile (key bindings +
    # attack range) for the given --player-name, so one script works for any
    # character. CLI overrides below still win over the profile.
    _profiles = cfg.get("character_profiles", {})
    _prof = _profiles.get(args.player_name) if args.player_name else None
    if _prof:
        if isinstance(_prof.get("key"), dict):
            cfg.setdefault("key", {}).update(
                {k: v for k, v in _prof["key"].items() if v}
            )
        if isinstance(_prof.get("attack"), dict):
            cfg.setdefault("attack_profiles", {})[args.player_name] = dict(
                _prof["attack"]
            )
        logger.info(
            f"[auto_combat] 角色档案已应用: {args.player_name} "
            f"(攻击 {_prof.get('attack', {}).get('horizontal', '?')}x"
            f"{_prof.get('attack', {}).get('vertical', '?')}, "
            f"保持距离 {_prof.get('attack', {}).get('keep_distance', '?')}, "
            f"攻击键 {cfg['key'].get('directional_attack', '?')}, "
            f"喝HP {cfg['key'].get('add_hp', '-')}, "
            f"喝MP {cfg['key'].get('add_mp', '-')})"
        )
    if args.player_name:
        # Player localization is OCR-name based (works for ANY character):
        # RapidOCR finds "<player_name>" -> its nametag becomes the tracking
        # template. The legacy fixed nametag template is NOT used because it
        # is a screenshot of a specific character's name (超团甜) and falsely
        # matches other on-screen text at low thresholds, boxing the wrong
        # target (observed score 0.333 >= 0.30 returning a wrong position).
        # --no-ocr disables the PaddleOCR background thread entirely (it
        # pegs the CPU at 1500%+ while route-following), relying only on the
        # offline player/nametag templates.
        _ocr_detector = None if args.no_ocr else NameOcrPlayerDetector(
            cfg, args.player_name)
        # 完全照搬 codex 原版玩家定位链路(OCR身份确认 + 局部模板跟踪 + 平滑 + 坐标)
        from tools.yolo_monster_viewer import (
            ReadOnlyPlayerDetector as _CodexPlayer,
            DetectionTracker as _CodexPlayerTracker,
            EntityCoordinateTracker as _CodexPlayerCoord,
            AsyncNameOcrLocator as _CodexOcr,
            resolve_gameplay_height as _resolve_gameplay_height,
        )
        _overlay = cfg["perception_overlay"]
        _safe_name = "".join(c if c.isalnum() else "_" for c in args.player_name)
        _name_tpl = f"nametag/{_safe_name}_player.png"
        _codex_ui = int(cfg["ui_coords"]["ui_y_start"])
        _codex_reference_width = cfg["ui_coords"].get("reference_width")
        _codex_ocr_enabled = not args.no_ocr and _CodexOcr.available()
        _codex_player = _CodexPlayer(
            _name_tpl,
            threshold=float(_overlay.get("player_match_threshold", 0.30)),
            box_size=tuple(_overlay["player_box_size"]),
            center_offset_y=abs(int(cfg["nametag"]["offset"][1])),
            identity_threshold=float(_overlay.get("player_identity_threshold", 0.48)),
            local_identity_threshold=float(_overlay.get("player_local_identity_threshold", 0.38)),
            identity_margin=float(_overlay.get("player_identity_margin", 0.015)),
            glyph_threshold=int(_overlay.get("player_glyph_threshold", 130)),
            glyph_weight=float(_overlay.get("player_glyph_weight", 0.70)),
            candidate_count=int(_overlay.get("player_candidate_count", 16)),
            lock_radius=float(_overlay.get("player_lock_radius", 180.0)),
            reacquire_misses=int(_overlay.get("player_reacquire_misses", 12)),
            center_weight=float(_overlay.get("player_center_weight", 0.12)),
            require_identity_seed=_codex_ocr_enabled,
            max_valid_x=_overlay.get("player_max_valid_x"),
            max_valid_y=_overlay.get("player_max_valid_y"),
            glyph_min_columns=int(_overlay.get("player_glyph_min_columns", 2)),
            color_anchor_enabled=bool(
                _overlay.get("player_color_anchor_enabled", True)
            ),
            color_anchor_name_offset_y=int(
                _overlay.get("player_color_anchor_name_offset_y", 24)
            ),
            color_anchor_min_red_fraction=float(
                _overlay.get("player_color_anchor_min_red_fraction", 0.02)
            ),
        )
        _codex_ocr = None
        if _codex_ocr_enabled:
            _codex_ocr = _CodexOcr(
                args.player_name,
                _codex_player.template.shape,
                confidence=float(_overlay.get("player_ocr_confidence", 0.70)),
                submit_interval=float(_overlay.get("player_ocr_submit_interval", 0.50)),
                refresh_interval=float(_overlay.get("player_ocr_refresh_interval", 3.0)),
                ocr_threads=int(_overlay.get("player_ocr_threads", 2)),
                title_text=_overlay.get("player_title_anchor"),
            )
        _codex_player_tracker = _CodexPlayerTracker(
            max_missed=40, smoothing=0.55, match_iou=0.05, max_center_distance=1.80)
        _codex_player_coord = _CodexPlayerCoord(
            velocity_smoothing=0.50, move_threshold_px_s=30.0,
            vertical_threshold_px_s=40.0)
        _codex_last_ocr_frame = 0

        class _CodexPlayerWrapper:
            """codex 原版玩家定位, 只做字段适配给攻击逻辑。"""

            title_strip_count = 0   # 画面中【其他玩家】蓝色称号条数量(已排除自己)
            title_strip_proposals = []  # 全量蓝色称号条候选框(可视化红框用)
            _strip_excl_center = None  # 自己玩家框中心(排除自己那条用)
            _strip_ref_color = None  # 自己称号条平均BGR(颜色参照, 滤地形误检)
            _smoothed_box = None   # 玩家框EMA平滑(静止稳/移动跟) - 抗黄框抖动
            # ---- 小地图运动估计(识别丢失时框不飘) ----
            # 校准: color_anchor 正常时记录 (mini_norm_x -> 屏幕box中心x) 样本;
            # 丢失时用最近样本的线性映射, 从当前 mini_norm_x 估计屏幕位置——
            # 角色左走/跳跃时小地图坐标同步移动, 框跟着角色而不是飘走。
            _motion_samples = []     # [(mini_x, screen_cx), ...] 最近30个
            _motion_linear = None    # (k, b) 屏幕x = k*mini_x + b
            _last_mini_x = None      # 上一帧 mini_norm_x
            _last_screen_cx = None   # 上一帧屏幕中心x(丢帧估计的起点)

            def _update_strip_ref(self, frame, gameplay_height):
                """从自己 color_anchor 定位的 anchor_box 提取称号条平均色作参照。"""
                if _codex_player is None:
                    return
                try:
                    # 直接复用检测器内部最近一次的 anchor(与 _find_color_anchor 同源)
                    anchor = getattr(_codex_player, "_last_color_anchor_box", None)
                    if anchor is not None:
                        x, y, w, h = (int(v) for v in anchor)
                        x0 = max(0, x); y0 = max(0, y)
                        x1 = min(frame.shape[1], x + w)
                        y1 = min(gameplay_height, y + h)
                        crop = frame[y0:y1, x0:x1]
                        if crop.size:
                            self._strip_ref_color = tuple(
                                crop.reshape(-1, 3).mean(axis=0).astype(int))
                except Exception:
                    pass

            def detect(self, frame, now):
                nonlocal _codex_last_ocr_frame
                gameplay_height = _resolve_gameplay_height(
                    frame.shape,
                    _codex_ui,
                    _codex_reference_width,
                )
                if _codex_ocr is not None:
                    _codex_ocr.submit(frame[:gameplay_height])
                    ocr_identity = _codex_ocr.latest(max_age=10.0)
                    if (
                        ocr_identity is not None
                        and ocr_identity["frame_id"] > _codex_last_ocr_frame
                    ):
                        _codex_player.seed_identity(ocr_identity, frame)
                        _codex_last_ocr_frame = ocr_identity["frame_id"]
                        logger.info(
                            f"[玩家定位] OCR确认: src={ocr_identity.get('identity_source')} "
                            f"text={ocr_identity.get('text')!r} loc={ocr_identity['location']} "
                            f"锚点={'有' if ocr_identity.get('anchor_box') is not None else '无'}")
                raw = _codex_player.detect(frame, gameplay_height)
                # 统计【其他玩家】称号条: 用自己定位框中心附近作为排除区(自己那条
                # 不计入)——否则"自己那条 + 打怪特效蓝色块" = 2 触发误停。
                _excl = self._strip_excl_center
                if raw is not None and raw.get("box"):
                    _b = raw["box"]
                    _excl = (int(_b[0] + _b[2] / 2.0), int(_b[1] + _b[3] / 2.0))
                    self._strip_excl_center = _excl
                # 颜色参照: 自己称号条平均色(anchor_box 区域), 滤地形误检
                _ref = self._strip_ref_color
                if raw is not None and raw.get("anchor_box"):
                    _ab = raw["anchor_box"]
                    try:
                        _ax, _ay, _aw, _ah = (int(v) for v in _ab)
                        _ax0 = max(0, _ax); _ay0 = max(0, _ay)
                        _ax1 = min(frame.shape[1], _ax + _aw)
                        _ay1 = min(gameplay_height, _ay + _ah)
                        _crop = frame[_ay0:_ay1, _ax0:_ax1]
                        if _crop.size:
                            _ref = tuple(
                                _crop.reshape(-1, 3).mean(axis=0).astype(int))
                            self._strip_ref_color = _ref
                    except Exception:
                        pass
                # 勋章检测已完全弃用(暂停判定改用小地图红点), 不再收集候选框
                self.title_strip_count = 0
                self.title_strip_proposals = []
                tracked = _codex_player_tracker.update(
                    [] if raw is None else [raw])
                if not tracked:
                    return None
                best = min(tracked, key=lambda item: (
                    item["missed_frames"], -item["confidence"]))
                enriched = _codex_player_coord.update(
                    [best], now, frame.shape[1], gameplay_height, fixed_entity_id="P1")
                p = enriched[0]
                x, y, w, h = p["box"]
                # ---- 小地图运动估计(识别丢失时框不飘) ----
                # color_anchor 正常: 校准 mini_norm_x -> 屏幕中心x 的线性映射。
                # color_anchor 丢失(tracker保持/PREdict): 用当前 mini_norm_x
                # 经映射估计屏幕中心x——角色左走/跳跃时小地图坐标同步移动,
                # 框跟着角色移动而不是停在原地/飘走。
                _mini = locate_minimap_player(frame, cfg.get("minimap", {}))
                _is_hold = bool(p.get("missed_frames", 0) > 0
                                or p.get("tracking_state") == "PREDICTED"
                                or p.get("identity_mode") == "color_anchor_hold")
                if _mini is not None and _mini.get("map_norm"):
                    _mnx = float(_mini["map_norm"][0])
                    _cx = x + w / 2.0
                    if not _is_hold:
                        # 真识别: 记录校准样本, 拟合线性映射(最近30个, 最小二乘)
                        self._motion_samples.append((_mnx, _cx))
                        if len(self._motion_samples) > 30:
                            self._motion_samples.pop(0)
                        if len(self._motion_samples) >= 3:
                            _xs = np.asarray([s[0] for s in self._motion_samples])
                            _ys = np.asarray([s[1] for s in self._motion_samples])
                            try:
                                _k, _b = np.polyfit(_xs, _ys, 1)
                                self._motion_linear = (float(_k), float(_b))
                            except Exception:
                                pass
                        self._last_mini_x = _mnx
                        self._last_screen_cx = _cx
                    else:
                        # 丢失保持: 用映射估计屏幕中心x(角色在动, 框要跟着)
                        _est_cx = None
                        if self._motion_linear is not None:
                            _est_cx = self._motion_linear[0] * _mnx + self._motion_linear[1]
                        elif self._last_mini_x is not None and self._last_screen_cx is not None:
                            # 无映射时用增量: 小地图位移 * 经验比例(屏幕宽/地图宽≈1.2)
                            _dmnx = _mnx - self._last_mini_x
                            _est_cx = self._last_screen_cx + _dmnx * frame.shape[1] * 1.2
                        if _est_cx is not None:
                            _clamp = max(w / 2.0 + 5, min(frame.shape[1] - w / 2.0 - 5, _est_cx))
                            x = float(_clamp - w / 2.0)   # 移到估计的屏幕位置
                            p["motion_state"] = "MOVE"
                # ---- 玩家框EMA平滑(抗黄框抖动) ----
                # color_anchor 每帧检测有 1~2px 抖动, 直接显示会闪。运动自适应:
                # 移动时弱平滑(alpha 高, 快速跟随), 静止时强平滑(alpha 低, 框稳)。
                # 跳跃/爬绳时用中等平滑, 避免框飘走又不过度滞后。
                _ms = p.get("motion_state", "STILL")
                _alpha = 0.85 if _ms in ("MOVE", "UP", "DOWN") else 0.35
                if self._smoothed_box is None:
                    self._smoothed_box = [float(x), float(y), float(w), float(h)]
                else:
                    _sx, _sy, _sw, _sh = self._smoothed_box
                    # 平滑只作用于位置(左上角), 尺寸取当前检测值(角色大小恒定)
                    self._smoothed_box[0] = _sx * (1.0 - _alpha) + x * _alpha
                    self._smoothed_box[1] = _sy * (1.0 - _alpha) + y * _alpha
                    self._smoothed_box[2] = float(w)
                    self._smoothed_box[3] = float(h)
                _bx = int(round(self._smoothed_box[0]))
                _by = int(round(self._smoothed_box[1]))
                _bw = int(round(self._smoothed_box[2]))
                _bh = int(round(self._smoothed_box[3]))
                # 中心也用平滑后的框重新计算(攻击/跳跃判定用一致坐标)
                _bcx = _bx + _bw // 2
                _bcy = _by + _bh // 2
                return {
                    "label": "PLAYER",
                    "score": p["confidence"],
                    "box": (_bx, _by, _bw, _bh),
                    "center": (_bcx, _bcy),
                    "nametag_box": tuple(p.get("nametag_box", p["box"])),
                    "method": "codex",
                    "entity_id": "P1",
                    "center_px": [_bcx, _bcy],
                    "velocity_px_s": p["velocity_px_s"],
                    "speed_px_s": p["speed_px_s"],
                    "motion_state": p["motion_state"],
                    "tracking_state": p["tracking_state"],
                    # codex 新版检测器字段透传(color_anchor 模式):
                    # identity_mode 通常为 "color_anchor"; anchor_box 是蓝色称号条;
                    # red_fraction 是红色角色特征比例(换角色后需重新标定)。
                    "identity_mode": p.get("identity_mode"),
                    "anchor_box": p.get("anchor_box"),
                    "red_fraction": p.get("red_fraction"),
                }

            def reset(self):
                """强制重新识别玩家: 清除身份锁定与 OCR 锚点, 解决识别框卡死。"""
                _codex_player.reset()

            def stop(self):
                if _codex_ocr is not None:
                    _codex_ocr.stop()

        player_detector = _CodexPlayerWrapper()
        # 注意: codex 新版 ReadOnlyPlayerDetector 在 no-ocr 下也用 color_anchor
        # (蓝色称号条+红色身体锚点)定位, 不是"纯模板匹配"——措辞要准确,
        # 避免用户误以为没跑优化后的代码。
        _mode_desc = (
            "OCR身份确认" if _codex_ocr_enabled
            else f"color_anchor色块锚点(no-ocr, 红分阈值"
                 f"{_overlay.get('player_color_anchor_min_red_fraction', 0.02)})"
        )
        logger.info(
            f"[auto_combat] 已用 codex 玩家定位 ({_mode_desc})({_name_tpl})")
    else:
        player_detector = PlayerDetector(cfg)
    if args.monster_backend == "yolo":
        yolo_confidence = (
            args.yolo_confidence
            if args.yolo_confidence is not None
            else float(auto_cfg.get("yolo_confidence", 0.10))
        )
        # 原封不动照搬 codex 的怪物检测完整链路(YOLO + 平滑跟踪 + 坐标),
        # 不加同层过滤/前方过滤, 只做字段适配给攻击逻辑。
        from tools.yolo_monster_viewer import (
            YoloMonsterDetector as _CodexYolo,
            DetectionTracker as _CodexTracker,
            EntityCoordinateTracker as _CodexCoord,
            attach_player_relative_coordinates as _codex_attach,
            normalize_label as _codex_norm,
        )
        _codex_labels = [_codex_norm(x) for x in (
            args.monster_labels or ["僵尸蘑菇", "刺蘑菇"])]
        yolo_iou = (
            args.yolo_iou
            if args.yolo_iou is not None
            else float(auto_cfg.get("yolo_iou", 0.45))
        )
        _codex_core = _CodexYolo(
            args.yolo_model, yolo_confidence, yolo_iou,
            args.yolo_device, args.yolo_image_size, _codex_labels)
        _codex_tracker = _CodexTracker(
            max_missed=4, smoothing=0.65,
            min_confirmed_hits=2, high_confidence_confirm=0.75)
        _codex_coord = _CodexCoord()
        _codex_mui = int(cfg["ui_coords"]["ui_y_start"])
        # ---- 树妖(木妖/stump)并行检测 ----
        # 第二个 YOLO 模型(warrior_stump_hardneg_v2_1280)专检树妖, 与野猪
        # 模型并行, 结果合并给攻击逻辑。参数沿用 codex 验证过的:
        # conf=0.70 iou=0.45 image_size=1280 device=0。不改变任何启动参数。
        _stump_model = str(auto_cfg.get(
            "stump_model",
            "training_runs/warrior_stump_hardneg_v2_1280/weights/best.pt"))
        _stump_core = None
        try:
            if os.path.exists(_stump_model) and _stump_model != args.yolo_model:
                _stump_core = _CodexYolo(
                    _stump_model,
                    float(auto_cfg.get("stump_confidence", 0.70)),
                    float(auto_cfg.get("stump_iou", 0.45)),
                    args.yolo_device,
                    int(auto_cfg.get("stump_image_size", 1280)),
                    ["stump"],
                )
                _stump_tracker = _CodexTracker(
                    max_missed=int(auto_cfg.get("stump_track_max_missed", 1)),
                    smoothing=0.65,
                    min_confirmed_hits=int(auto_cfg.get("stump_track_min_hits", 2)),
                    high_confidence_confirm=float(
                        auto_cfg.get("stump_track_high_confidence", 0.70)),
                )
                logger.info(
                    f"[auto_combat] 已加载树妖模型 {_stump_model} "
                    f"(conf={auto_cfg.get('stump_confidence', 0.70)})")
            else:
                logger.info("[auto_combat] 树妖模型未配置或与主模型相同, 跳过")
        except Exception as _stump_exc:
            _stump_core = None
            logger.warning(f"[auto_combat] 树妖模型加载失败: {_stump_exc}")

        class _CodexMonsterWrapper:
            """codex 原版怪物检测(全图, 不过滤), 只做字段适配。"""

            def detect(self, frame, player):
                dets = _codex_core.detect(frame, _codex_mui)
                dets = _codex_tracker.update(dets)
                # 合并树妖检测(第二模型, 独立 tracker)
                if _stump_core is not None:
                    try:
                        _stump_dets = _stump_core.detect(frame, _codex_mui)
                        _stump_dets = _stump_tracker.update(_stump_dets)
                        for _sd in _stump_dets:
                            _sd["label"] = "木妖"
                            dets.append(_sd)
                    except Exception:
                        pass
                dets = _codex_coord.update(
                    dets, time.time(), frame.shape[1], _codex_mui, prefix="M")
                if player is not None and player.get("center_px"):
                    dets = _codex_attach(dets, {"center_px": player["center_px"]})
                out = []
                for d in dets:
                    _bx = d["box"]
                    _bw = int(_bx[2])
                    _bh = int(_bx[3])
                    # 大小过滤: 掉落物品/噪声误检的框普遍很小(实测宽32~47高31~35),
                    # 猪猪正常框宽50+高42+, 过滤小框避免把掉落物品当怪物打。
                    # 树妖(木妖)体积可能小于野猪, 用更宽松的下限(30x30)。
                    _min_w = 30 if d.get("label") == "木妖" else 50
                    _min_h = 30 if d.get("label") == "木妖" else 40
                    if _bw < _min_w or _bh < _min_h:
                        continue
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
                        "relative_to_player": d.get("relative_to_player"),
                        "has_hp_bar": False,
                    })
                return out

        monster_detector = _CodexMonsterWrapper()
    elif args.monster_backend == "template":
        # Original-style monster detection (src/engine/MapleStoryAutoLevelUp.py
        # :get_monsters_in_range). Loads monster/<name>/*.png templates (each
        # monster has 1-N poses), strips green-screen background, runs CCOEFF
        # NORMED + mask template matching at 0.5x grayscale (fast), NMS to
        # dedupe, excludes the player's own region. No YOLO model load -> very
        # low memory footprint. CPU is kept low by searching only a horizontal
        # band around the player and (optionally) only this map's monster
        # classes via --monster-labels.
        monster_detector = TemplateMonsterDetector(
            cfg,
            level_band_half_h=float(auto_cfg.get("sprite_level_band_half_h", 110)),
            include_labels=args.monster_labels or None,
        )
    elif args.monster_backend == "sprite":
        # Sprite-template detector: real in-game sprites cropped from labeled
        # game frames (monster_templates_final/). Two-stage coarse-to-fine
        # matching + class-color verification. Accurate for the monsters it
        # has templates for; new monsters are learned at runtime by the
        # TemplateCollector (see --collect-templates).
        monster_detector = SpriteMonsterDetector(
            threshold_fine=float(auto_cfg.get("sprite_threshold_fine", 0.68)),
            threshold_coarse=float(auto_cfg.get("sprite_threshold_coarse", 0.66)),
            level_band_half_h=float(auto_cfg.get("sprite_level_band_half_h", 110)),
            level_tol_px=float(auto_cfg.get("sprite_level_tol_px", 45)),
        )
    elif args.monster_backend == "color":
        # Pure color + connected-component detector (optimized upstream
        # `template_free` idea). No YOLO/templates -> tiny CPU. Only searches
        # the player's horizontal band. Color ranges reuse COLOR_RANGES.
        monster_detector = ColorMonsterDetector(
            cfg,
            level_band_half_h=float(auto_cfg.get("color_band_half_h", 130)),
            include_labels=args.monster_labels or None,
        )
    else:
        monster_detector = MonsterDetector(cfg)
    template_collector = None
    if getattr(args, "collect_templates", False):
        template_collector = TemplateCollector(
            min_score=float(auto_cfg.get("collect_min_score", 0.80)))
    monitor = HealthMonitor(cfg, kb_controller=None)
    motion_detector = None
    if args.motion_detection:
        motion_detector = MotionDetector(
            cfg,
            threshold=args.motion_threshold,
            min_area=args.motion_min_area,
            max_area=args.motion_max_area,
            candidate_score=args.motion_candidate_score,
        )
    advisory_evaluator = AdvisoryEvaluator(cfg)
    policy = CombatPolicy(cfg, player_name=args.player_name or None, mode=args.mode)
    if args.attack_horizontal is not None:
        policy.attack_horizontal_px = args.attack_horizontal
    if args.attack_vertical is not None:
        policy.attack_vertical_px = args.attack_vertical
    if args.keep_distance is not None:
        policy.min_engage_px = args.keep_distance
    executor = CombatExecutor(cfg, dry_run=args.dry_run, mode=args.mode)
    # HP/MP 具体数 OCR 识别器(反击检测: 扣一滴血也能识别)
    hp_ocr = HpMpOcrReader(
        hp_region=cfg["health_monitor"].get("hp_ocr_region", [466, 725, 85, 18]),
        mp_region=cfg["health_monitor"].get("mp_ocr_region", [568, 725, 85, 18]),
        exp_region=cfg["health_monitor"].get("exp_ocr_region"),
        submit_interval=float(cfg["health_monitor"].get("hp_ocr_submit_interval", 0.25)),
        ocr_threads=1,
    )
    if args.attack_key:
        executor.attack_key = args.attack_key
    if args.add_hp_key:
        executor.add_hp_key = args.add_hp_key
    if args.add_mp_key:
        executor.add_mp_key = args.add_mp_key
    capture = GameWindowCapturor(cfg)
    pause_control = PauseController(args.pause_key, args.quit_key)
    pause_control.start_listener()
    terrain_scanner = TerrainScanner()
    minimap_navigator = MinimapNavigator(cfg)
    route_follower = RouteFollower(cfg)
    recorder = RouteRecorderCore(cfg, route_follower)
    active_map_name = args.map_name or ""
    # 小地图坐标航点巡逻: 用稳定的 map_norm 精确控制路线(适合大量跳跃/爬绳点的地图)
    waypoint_patrol = MinimapWaypointPatrol(cfg, active_map_name)
    policy._waypoint_patrol = waypoint_patrol
    # 安全点(测谎仪规避): 独立 one-shot 导航器, 与主航线状态完全分离
    safe_patrol = MinimapWaypointPatrol(cfg, active_map_name)
    policy._safe_patrol = safe_patrol
    # 恢复路线(安全点退出商城后/跌落底层走回巡游线): 同样独立 one-shot 导航器
    recall_patrol = MinimapWaypointPatrol(cfg, active_map_name)
    policy._recall_patrol = recall_patrol
    # 安全点定时进商城配置: 每小时 schedule_minutes(默认整点/半点)触发,
    # 每次进商城 wait_in_shop 秒(5分钟)后 ESC+回车返回。
    _safe_cfg = cfg.get("safe_point", {})
    safe_sched = sorted({int(float(m)) % 60 for m in
                         _safe_cfg.get("schedule_minutes", [0, 30])})
    safe_wait_before_shop = float(_safe_cfg.get("wait_before_shop", 5.0))
    safe_wait_in_shop = float(_safe_cfg.get("wait_in_shop", 300.0))
    safe_shop_key = str(_safe_cfg.get("shop_key", "t"))
    safe_esc_gap = 0.5   # ESC 与回车之间的间隔(秒)
    safe_wrap_wait = 1.0 # 回车后等待秒数再恢复巡航
    safe_max_trip = float(_safe_cfg.get("max_trip_seconds", 120.0))  # 走向安全点超时(卡住取消)

    def _next_safe_slot_ts():
        """下一个安全点触发时刻(本地时间, epoch 秒): 每个小时的
        schedule_minutes 分钟(默认 0=整点 / 30=半点, 如 8:30、9:00、9:30)。"""
        _now = datetime.datetime.now()
        _cands = []
        for _h in range(24):
            for _m in safe_sched:
                _cands.append(_now.replace(
                    hour=_h, minute=_m, second=0, microsecond=0))
        _cands.sort()
        for _c in _cands:
            if _c > _now:
                return float(time.mktime(_c.timetuple()))
        return float(time.mktime(_cands[0].timetuple())) + 86400.0
    # 恢复路线配置(触发判定/冷却)
    _recall_cfg = cfg.get("recall_point", {})
    recall_y_tol = float(_recall_cfg.get("trigger_y_tol", 0.03))
    recall_cooldown = float(_recall_cfg.get("cooldown_seconds", 30.0))
    recall_max_trip = float(_recall_cfg.get("max_trip_seconds", 120.0))  # 恢复行程超时(卡住取消)
    # The map name (and its recorded route) is auto-detected from the minimap
    # in a background thread: the OCR model load is slow (~5s) and must never
    # block the main loop. When it succeeds the routes become available and
    # the patrol logic automatically switches to route-following.
    # NOTE: OCR sometimes mis-reads map names (e.g. "射手训练场III" ->
    # "射手训练场川"), which makes the route dir mismatch and the bot degrade
    # to patrol. Passing --map-name bypasses OCR entirely.
    def _load_route_in_background():
        nonlocal active_map_name
        try:
            detected = args.map_name
            if not detected:
                from src.utils.minimap_ocr import get_minimap_map_name
                frame = capture.get_frame()
                if frame is None:
                    return
                detected = get_minimap_map_name(
                    frame,
                    region=cfg["minimap"].get(
                        "ocr_region",
                        cfg["minimap"].get("region")))
            if detected:
                active_map_name = detected
                recorder._ensure_map_dir(detected)
                if waypoint_patrol.load_waypoints(detected):
                    logger.info(
                        f"[auto_combat] map '{detected}' 加载 "
                        f"{len(waypoint_patrol.waypoints)} 个航点(小地图坐标)")
                    # 自动开始巡航: 之前"永远原地待命(wp_idle)"的根因是
                    # _patrolling 初始为 False, 只有用户在窗口按 F4 才会
                    # start_patrol; 重启后无人按 F4 -> decide() 返回 None ->
                    # wp_idle。这里加载到航点后直接启动, 和以前手动按 F4
                    # 行为一致(自动巡航)。
                    if waypoint_patrol.start_patrol():
                        logger.info(
                            f"[auto_combat] 航点已加载, 自动开始路线巡航 "
                            f"({len(waypoint_patrol.waypoints)} 个点)")
                safe_patrol.load_safe_points(detected)
                recall_patrol.load_recall_points(detected)
            if detected and route_follower.load_map_routes(detected):
                # --route-name 指定时, 优先选择匹配的那条路线
                if args.route_name:
                    for i, fname in enumerate(route_follower.route_files):
                        if args.route_name in fname:
                            route_follower.idx_route = i
                            break
                logger.info(
                    f"[auto_combat] Detected map '{detected}', "
                    f"loaded {len(route_follower.img_routes)} routes"
                    f"{', using ' + route_follower.route_files[route_follower.idx_route] if route_follower.route_files else ''}")
            else:
                logger.warning(
                    f"[auto_combat] Map name '{detected}' has no recorded "
                    f"route; falling back to patrol-only navigation.")
        except Exception as exc:
            logger.warning(f"[auto_combat] route load failed: {exc}")

    threading.Thread(target=_load_route_in_background, daemon=True).start()
    terrain_scan_interval = float(auto_cfg.get("terrain_scan_interval", 0.5))
    last_terrain_scan_time = 0.0
    terrain = None
    minimap_scan = None
    if args.no_monster:
        # 关闭猪猪检测: 不启动后台怪物检测线程, 怪物列表恒空(仅巡游/受伤反击)
        async_monster_detector = None
    else:
        async_monster_detector = AsyncMonsterDetector(
            monster_detector,
            max_age=float(auto_cfg.get("monster_max_age", 2.0)),
            min_interval=float(auto_cfg.get("monster_min_interval", 0.25)),
        )
        async_monster_detector.start()

    started = time.time()
    frame_count = 0
    # ---- 自动采集训练帧(F8 启动巡游/hang时采集) ----
    # 训练人物 YOLO 用: 主循环不阻塞地定期保存带标注的玩家帧。
    _capture_enabled = not args.no_capture
    _capture_interval = float(getattr(args, "capture_interval", 2.0))
    _capture_dir = Path(getattr(args, "capture_dir",
                        "training_data/player_capture_auto"))
    _capture_img_dir = _capture_dir / "images"
    _capture_lbl_dir = _capture_dir / "labels"
    _capture_count = 0
    _capture_last_at = 0.0
    if _capture_enabled:
        try:
            _capture_img_dir.mkdir(parents=True, exist_ok=True)
            _capture_lbl_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[采集] 自动采集已开启: {_capture_dir} "
                        f"(每{_capture_interval}s一张, F8解除暂停且玩家定位后采集)")
        except Exception as _e:
            _capture_enabled = False
            logger.warning(f"[采集] 目录创建失败, 采集关闭: {_e}")
    cached_monsters = []
    sticky_monsters = []           # last known monsters when detection flickers
    sticky_until = float("-inf")   # keep stale monsters until this timestamp
    advisory = None
    player = None
    vitals = (None, None, None)
    last_decision = "none"
    last_reason = ""
    last_state_log = float("-inf")  # 上次状态快照日志时间(每5秒)
    show_window = args.show_viz and not args.headless
    fps_limit = max(2, min(30, args.fps_limit))
    measured_fps = 0.0
    last_frame_time = None

    def _set_fps(value):
        nonlocal fps_limit
        fps_limit = max(2, min(30, int(value)))

    # Keyboard parameter panel (number pick -> type value -> Enter).
    # Key-binding fields (1-3): pick with the number key, then press the key
    # you want to bind. Numeric fields (4-9): pick, type, Enter.
    param_editor = ParamEditor()
    # 经验统计(按轮): 每次跑完一轮点位结算经验/耗时, 计算每小时经验
    exp_stats = ExpRoundStats()
    _exp_value_latest = None  # 最近一次 OCR 识别的 EXP 值数字(供回调结算)

    # 完成一轮回调: 主循环注入 MinimapWaypointPatrol, 用最新 EXP 值数字结算本轮
    def _on_round_complete(round_count):
        nonlocal _exp_value_latest
        _ev = _exp_value_latest
        if _ev is not None:
            _summary = exp_stats.round_done(_ev, time.time())
            if _summary is not None:
                logger.info(
                    f"[经验] 第{round_count}圈统计: "
                    f"平均每轮 {_summary['avg_exp']:.0f} EXP / "
                    f"{_summary['avg_duration']:.0f}s | "
                    f"{_summary['exp_per_hour']:.0f} EXP/小时 "
                    f"({_summary['rounds']}轮)")
    waypoint_patrol.on_round_complete = _on_round_complete
    param_editor.add(
        ord("1"), "攻击键", lambda: executor.attack_key,
        lambda v: setattr(executor, "attack_key", v), bind=True,
    )
    param_editor.add(
        ord("2"), "喝HP键", lambda: executor.add_hp_key or "-",
        lambda v: setattr(executor, "add_hp_key", v), bind=True,
    )
    param_editor.add(
        ord("3"), "喝MP键", lambda: executor.add_mp_key or "-",
        lambda v: setattr(executor, "add_mp_key", v), bind=True,
    )
    param_editor.add(
        ord("4"), "攻击范围-横向", lambda: policy.attack_horizontal_px,
        lambda v: setattr(policy, "attack_horizontal_px", float(v)), 20,
    )
    param_editor.add(
        ord("5"), "攻击范围-纵向", lambda: policy.attack_vertical_px,
        lambda v: setattr(policy, "attack_vertical_px", float(v)), 10,
    )
    param_editor.add(
        ord("6"), "保持距离", lambda: policy.min_engage_px,
        lambda v: setattr(policy, "min_engage_px", float(v)), 0,
    )
    param_editor.add(
        ord("7"), "自动喝药-HP阈值%", lambda: executor.add_hp_threshold,
        lambda v: setattr(executor, "add_hp_threshold", float(v)), 1,
    )
    param_editor.add(
        ord("8"), "自动喝药-MP阈值%", lambda: executor.add_mp_threshold,
        lambda v: setattr(executor, "add_mp_threshold", float(v)), 1,
    )
    param_editor.add(
        ord("9"), "帧率上限 FPS", lambda: float(fps_limit),
        lambda v: _set_fps(v), 2,
    )

    # Mouse click queue + button rects (panel interaction). ui_buttons is
    # refreshed by render_info_panel each displayed frame; the mouse callback
    # (registered below when show_window) pushes clicked actions to
    # click_actions.
    click_actions = []
    ui_buttons = []
    if show_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        # Combined image = game frame (~1296px) + info panel (420px). The
        # window must be at least that wide or OpenCV downscales the text and
        # everything looks blurry.
        cv2.resizeWindow(WINDOW_TITLE, 1900, 950)

        def _on_mouse(event, mx, my, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for (x0, y0, x1, y1, action) in ui_buttons:
                if x0 <= mx < x1 and y0 <= my < y1:
                    click_actions.append(action)
                    break

        cv2.setMouseCallback(WINDOW_TITLE, _on_mouse)

    _strip_first_seen = None  # 首次检测到红点时间戳
    _strip_clear_since = None  # 红点消失时间戳(满 other_player_resume_seconds 自动恢复)
    OTHER_PLAYER_RESUME_DELAY = float(
        cfg.get("minimap", {}).get("other_player_resume_seconds", 2.0)
    )  # 红点消失多久自动恢复挂机(用户要求: 消失立即恢复, 2s只是跨帧确认防闪烁)
    # ---- 安全点定时进商城(测谎仪规避)状态 ----
    # 状态: "" 空闲 / "walk" 走向安全点 / "wait_t" 到点等5s / "wait_esc" 进商城后等10s / "wrap" 收尾
    _safe_state = ""
    _safe_at = 0.0         # 当前阶段开始时刻
    _safe_next_visit = 0.0 # 下次触发时刻(0=尚未武装, 巡航开始后武装)
    _safe_pending = False  # 计时到期但正在等"走完当前圈(到最后一个巡游点)"再触发
    _last_round_count = 0  # 主航线完成的圈数(变化 = 刚走完最后一个巡游点)
    # ---- 恢复路线(安全点退出商城后/跌落底层时走回巡游线)状态 ----
    # 状态: "" 空闲 / "walk" 走恢复路线 / "done_wait" 走完站定1秒
    _recall_state = ""
    _recall_at = 0.0           # 当前阶段开始时刻
    _recall_cooldown_until = 0.0  # 恢复完成后冷却(防反复跌落触发)

    def _reset_trip_states():
        """F4/F1/F5: 全部重新来——取消安全点/恢复路线行程, 计时重新武装。
        (修复: 用户按 F4 想重开巡航, 但安全点行程状态未清, 又接着走安全点)"""
        nonlocal _safe_state, _safe_at, _safe_next_visit, _safe_pending
        nonlocal _recall_state, _recall_at, _recall_cooldown_until, _last_round_count
        policy._safe_active = False
        safe_patrol.end_safe_visit()
        _safe_state = ""
        _safe_at = 0.0
        _safe_pending = False
        _safe_next_visit = 0.0
        policy._recall_active = False
        recall_patrol.end_recall()
        _recall_state = ""
        _recall_at = 0.0
        _recall_cooldown_until = 0.0
        _last_round_count = waypoint_patrol._round_count
    # 红点跨帧确认(codex MinimapRedMarkerTracker): 连续 confirm_frames 帧
    # 同一位置出现才认定是真玩家, 滤掉地图地形/UI 的红色误检。
    _red_tracker = MinimapRedMarkerTracker(
        confirm_frames=int(cfg.get("minimap", {}).get("other_player_confirm_frames", 2)),
        max_missed=int(cfg.get("minimap", {}).get("other_player_max_missed_frames", 1)),
        max_distance=float(cfg.get("minimap", {}).get("other_player_match_distance_px", 8.0)),
    )
    try:
        while True:
            if pause_control.is_quit_requested():
                break
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            now = time.time()
            minimap_cfg = cfg.get("minimap", {})  # 小地图标定(每帧一次, 廉价)
            player = player_detector.detect(frame, now)
            # ---- 自动采集训练帧: F8 解除暂停(is_effectively_paused=False)
            # 且玩家已定位时, 每 capture_interval 秒保存一帧带标注截图 ----
            if (_capture_enabled and not pause_control.is_effectively_paused()
                    and player is not None and player.get("box")
                    and now - _capture_last_at >= _capture_interval):
                _capture_last_at = now
                try:
                    _bx, _by, _bw, _bh = (int(v) for v in player["box"])
                    _bw = min(max(_bw, 20), frame.shape[1] - _bx)
                    _bh = min(max(_bh, 20), frame.shape[0] - _by)
                    if _bw > 10 and _bh > 10 and _bx >= 0 and _by >= 0:
                        _cx = (_bx + _bw / 2.0) / float(frame.shape[1])
                        _cy = (_by + _bh / 2.0) / float(frame.shape[0])
                        _nw = min(0.30, _bw / float(frame.shape[1]))
                        _nh = min(0.35, _bh / float(frame.shape[0]))
                        _label = f"0 {_cx:.6f} {_cy:.6f} {_nw:.6f} {_nh:.6f}\n"
                        _name = f"auto_{int(now*1000)}_{_capture_count:05d}"
                        cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
                                         str(_capture_img_dir / f"{_name}.jpg"))
                        (_capture_lbl_dir / f"{_name}.txt").write_text(
                            _label, encoding="utf-8")
                        _capture_count += 1
                        if _capture_count in (1, 50, 100, 200, 500):
                            logger.info(
                                f"[采集] 已存 {_capture_count} 帧 -> {_capture_dir}")
                except Exception as _ce:
                    logger.warning(f"[采集] 保存失败(不中断): {_ce}")
            # ---- 检测其他玩家: 小地图红点(codex locate_minimap_players) ----
            # 小地图上其他玩家显示为红点, 自己为黄点。红点 >=1 即有人在场:
            # 暂停挂机(保持喝药, 不攻击/不巡航/不移动); 红点消失满
            # other_player_resume_seconds(默认2秒) 自动恢复(用户要求:
            # 检测不到红点就立即恢复, 不用手动按 F8)。
            _minimap_players = locate_minimap_players(frame, minimap_cfg)
            _red_dots = _red_tracker.update(
                _minimap_players.get("other_players") or [])
            _has_other = len(_red_dots) >= 1
            # 小地图稳定坐标: 每帧定位玩家在地图里的左右位置
            mini = _minimap_players.get("player")
            policy._mini = mini
            if _has_other:
                _strip_clear_since = None  # 红点在场: 清掉"消失计时"
                if _strip_first_seen is None:
                    _strip_first_seen = now
                    logger.warning(
                        f"[其他玩家] 检测到小地图红点 {len(_red_dots)} 个"
                        f"(map_px={[list(p['map_px']) for p in _red_dots]}), "
                        f"开始挂机(保持喝药, 不攻击/不移动; 红点消失"
                        f"{OTHER_PLAYER_RESUME_DELAY:.0f}秒后自动恢复)")
                if not pause_control.player_pause:
                    pause_control.set_player_pause(True)
                    logger.warning(
                        f"[其他玩家] 检测到 {len(_red_dots)} 个红点, 暂停挂机"
                        f"(红点消失 {OTHER_PLAYER_RESUME_DELAY:.0f} 秒后自动恢复)")
            else:
                # 红点消失: 记录消失开始时刻(仅一次), 满 delay 自动恢复
                # (修复: 旧逻辑每帧清 _strip_clear_since, "消失计时"永远只有1帧,
                #  自动恢复从未触发, 用户必须手动按 F8)
                if _strip_first_seen is not None:
                    _strip_first_seen = None
                    _strip_clear_since = now
                    logger.info(
                        f"[其他玩家] 红点已消失, {OTHER_PLAYER_RESUME_DELAY:.1f} 秒后自动恢复")
                elif (_strip_clear_since is not None
                        and pause_control.player_pause
                        and now - _strip_clear_since >= OTHER_PLAYER_RESUME_DELAY):
                    pause_control.resume_from_player_pause()
                    _strip_clear_since = None
                    logger.warning(
                        f"[其他玩家] 红点消失已满 {OTHER_PLAYER_RESUME_DELAY:.1f} 秒, "
                        f"自动恢复挂机")
            if hasattr(player_detector, "submit_frame"):
                player_detector.submit_frame(frame)
            # Pass the player box so band-restricted detectors only scan the
            # player's horizontal band (big CPU saving for template/sprite).
            if args.no_monster:
                # 不检测怪物: 只巡游 + 受伤反击(怪物列表恒空)
                cached_monsters = []
            else:
                async_monster_detector.submit_frame(frame, player)

                # Take the latest async YOLO result; keep stale detections briefly
                # so a dropped inference does not make the bot pause abruptly.
                # Sticky is time-based (not frame-count) so slow frames during
                # skill use do not prematurely drop the target.
                refreshed = async_monster_detector.get_latest(now)
                if refreshed:
                    sticky_monsters = refreshed
                    # YOLO inference takes ~1.2s on this machine; keep the last
                    # detections for 1.2s(约一帧间隔)——旧检测只保留一帧, 怪死后/
                    # 离开后 1.2s 内清空, 避免对空气/旧位置攻击浪费蓝(之前3s太长)
                    sticky_until = now + 1.2
                elif now < sticky_until:
                    pass  # keep last known monsters
                else:
                    sticky_monsters = []
                cached_monsters = sticky_monsters

            if motion_detector and frame_count % 2 == 0:
                motion_detector.detect(frame, player, cached_monsters)
            vitals = read_vitals(monitor, frame, cfg)
            # HP/MP/EXP 具体数 OCR(后台线程); 取最新结果供反击检测/经验统计
            hp_ocr.submit(frame)
            hp_mp = hp_ocr.latest(max_age=2.0)
            # 经验统计: 记录最新 EXP 值数字(供完成一轮时结算); 首次锚定轮起点
            _exp_value = (hp_mp or {}).get("exp")
            if _exp_value is not None:
                _exp_value = _exp_value[0] if isinstance(_exp_value, tuple) else int(_exp_value)
                _exp_value_latest = _exp_value
                exp_stats.anchor(_exp_value, now)

            # Evaluate advisory every frame. Camera motion does NOT reset combat
            # decisions here: our own movement changes the background, and the
            # YOLO detector already works on single frames.
            advisory = advisory_evaluator.evaluate(
                player,
                cached_monsters,
                now,
                camera_motion=False,
                facing=(policy.patrol_direction if policy.mode == "patrol_hunt" else None),  # 巡游打怪: 只攻击巡逻方向(前方)的怪, 角色方向只由边缘决定
                stationary=(policy.mode in ("stationary", "patrol_hunt")),  # 站桩/巡游打怪: 每帧选攻击范围内最近的怪, 不锁定(避免怪物死后打旧坐标)
            )

            # Template learning: while a target is tracked and confident,
            # collect its sprite into the template library (dedup inside).
            if (
                template_collector is not None
                and advisory is not None
                and advisory.get("status") in ("ATTACK READY", "TRACKING", "DODGE RISK")
                and advisory.get("target_box")
                and advisory.get("target_label")
            ):
                try:
                    fake_det = {
                        "label": advisory["target_label"],
                        "box": advisory["target_box"],
                        "score": 1.0,
                    }
                    saved_p = template_collector.collect(frame, fake_det)
                    if saved_p:
                        logger.info(f"[auto_combat] 新怪模板入库: {saved_p}")
                except Exception as exc:
                    logger.warning(f"[auto_combat] 模板采集失败: {exc}")

            # Terrain scan on a slower cadence: ropes do not move, only the
            # camera does. The scanner caches results so even calling it every
            # frame is cheap; we just do not need to.
            if now - last_terrain_scan_time >= terrain_scan_interval and player is not None:
                terrain = terrain_scanner.scan(frame, now, (player["center"][0], player["center"][1]))
                minimap_scan = minimap_navigator.scan(frame, now)
                last_terrain_scan_time = now

            # Built-in route recording: runs independently of the navigation
            # state so manual recording works (F1 toggles, F3 saves). It only
            # needs the player's global position + currently-held keys.
            if recorder.is_recording and minimap_scan is not None \
                    and minimap_scan.get("player_xy") \
                    and route_follower.img_map is not None:
                gxy_rec = route_follower.locate_player(
                    minimap_scan, minimap_scan["player_xy"])
                recorder.update(gxy_rec, pause_control.held_key_set(), now)
            elif recorder.is_recording:
                # Keep the last position so a brief detection gap does not
                # draw a wild line when the player reappears.
                recorder.loc_last = None
            # 小地图坐标航点录制(与路线录制并行独立): 用稳定的 map_norm + 手持按键
            if waypoint_patrol.is_recording and mini is not None:
                waypoint_patrol.record_sample(
                    mini["map_norm"], pause_control.held_key_set(), now)

            policy._frame_width = frame.shape[1]  # 巡游卡住检测的地图边缘判断
            _hp_cur = _hp_max = None
            if hp_mp and hp_mp.get("hp"):
                _hp_cur, _hp_max = hp_mp["hp"]
            _mp_cur = _mp_max = None
            if hp_mp and hp_mp.get("mp"):
                _mp_cur, _mp_max = hp_mp["mp"]
            command, reason = policy.decide(
                player, advisory, vitals[0], vitals[1], now,
                hp_cur=_hp_cur, hp_max=_hp_max, mp_cur=_mp_cur, mp_max=_mp_max,
                monsters=cached_monsters,
            )
            # ---- 日志记录系统(后续定位 BUG 用, 无需人工描述); --no-log 关闭 ----
            _pc = player.get("center") if player else None
            _pc_s = f"({_pc[0]:.0f},{_pc[1]:.0f})" if _pc else "(无)"
            if not args.no_log:
                if reason == "counter_attack":
                    _vel = player.get("velocity_px_s", [0, 0]) if player else [0, 0]
                    _dv = _vel[0] - (policy._last_vx if policy._last_vx is not None else 0)
                    logger.info(
                        f"[combat] 反击! HP={_hp_cur}/{_hp_max} "
                        f"坐标={_pc_s} 命令={command} vx={_vel[0]:.0f} "
                        f"Δvx={_dv:.0f} 意图={policy._last_move_dir}")
                elif reason in ("patrol_jump", "patrol_edge_turn", "patrol_forced", "patrol_stuck_keep", "patrol_stuck_turn", "rest"):
                    logger.info(f"[patrol] {reason} 命令={command} 坐标={_pc_s}")
                if reason != last_reason and reason not in ("patrol", "attack_cooldown", "attack_cooldown_patrol", "none"):
                    logger.info(
                        f"[decision] 命令={command} 原因={reason} "
                        f"HP={_hp_cur}/{_hp_max} 坐标={_pc_s}")
                if now - last_state_log >= 5.0:
                    _vel = player.get("velocity_px_s", [0, 0]) if player else [0, 0]
                    # 怪物框大小统计(观察猪猪框 vs 掉落物品小框, 用于大小过滤阈值)
                    _box_sizes = []
                    for _m in cached_monsters:
                        _b = _m.get("box")
                        if _b and len(_b) >= 4:
                            _box_sizes.append((int(_b[2]), int(_b[3])))
                    _box_str = ""
                    if _box_sizes:
                        _min_w = min(s[0] for s in _box_sizes)
                        _max_w = max(s[0] for s in _box_sizes)
                        _min_h = min(s[1] for s in _box_sizes)
                        _max_h = max(s[1] for s in _box_sizes)
                        _box_str = f" 框宽[{_min_w}~{_max_w}] 框高[{_min_h}~{_max_h}]"
                    _mm_str = ""
                    if minimap_scan is not None and minimap_scan.get("player_xy"):
                        _mxy = minimap_scan["player_xy"]
                        _mm_str = f" 小地图=({_mxy[0]:.0f},{_mxy[1]:.0f})"
                    _mini_norm_str = ""
                    if getattr(policy, "_mini", None) is not None:
                        _mn = policy._mini.get("map_norm")
                        if _mn:
                            _mini_norm_str = f" 小地图norm=({_mn[0]:.4f},{_mn[1]:.4f})"
                    _wing_str = ""
                    if player and player.get("wing_pos"):
                        _wp = player["wing_pos"]
                        _wing_str = f" 翅膀=({_wp[0]},{_wp[1]})"
                    _idm_str = ""
                    if player and player.get("identity_mode"):
                        _rf = player.get("red_fraction")
                        _idm_str = f" 检测={player['identity_mode']}"
                        if _rf is not None:
                            _idm_str += f"(红{_rf:.2f})"
                    logger.info(
                        f"[status] HP={_hp_cur}/{_hp_max} 怪物数={len(cached_monsters)} "
                        f"命令={command}/{reason} FPS={measured_fps:.1f} 坐标={_pc_s} "
                        f"vx={_vel[0]:.0f} 大方向={policy.patrol_direction} 意图={policy._last_move_dir}"
                        f" 游戏聚焦={'Y' if target_is_foreground(cfg['game_window']['title']) else 'N'}{_mini_norm_str}"
                        f"{_box_str}{_mm_str}{_wing_str}{_idm_str}")
                    last_state_log = now
            last_reason = reason
            # --no-attack: never engage monsters. Force the reason into the
            # patrol bucket so the navigation branch below (route / minimap /
            # terrain) always takes over, and neutralize any attack commands.
            if args.no_attack:
                if command.startswith("attack") or reason in (
                    "attack", "approach", "dodge_imminent", "keep_distance",
                    "attack_cooldown",
                ):
                    command = "none"
                    reason = "patrol"
            # Key-aware player tracking: tell the locator which direction the
            # character is moving so it can predict the position and reject
            # far-away false matches (stops the box teleporting).
            if hasattr(player_detector, "notify_move"):
                player_detector.notify_move(command, now)
            policy.set_move_dir(command)  # 记录移动意图, 供下一帧反击(击退)检测
            # 【跳跃后禁攻击】: 本帧发了跳跃命令(jump_*) -> 记录跳跃时刻,
            # decide() 据此在跳跃后 2 秒内禁主动攻击(防跳台攻击掉落)。
            if isinstance(command, str) and command.startswith("jump"):
                policy.set_jump_at(now)
            # 真实朝向校准: 用玩家检测的真实速度(velocity_px_s)推断角色此刻
            # 面对的方向, 供攻击转向判断(_real_dir)。比纯命令模拟可靠。
            if player is not None:
                _vel_x = player.get("velocity_px_s", [0, 0])
                if _vel_x:
                    executor.set_real_facing(float(_vel_x[0]), now)
            # Terrain overrides patrol: when no monster is engaged, prefer
            # climbing a rope or jumping a platform over random walk. The
            # recorded route is the primary navigation source (reliable for
            # any map), the minimap terrain analysis is a fallback, and the
            # scene rope scanner is the last resort.
            if (
                not args.no_monster  # 巡游测试模式: 禁路线/小地图导航接管, 方向完全由巡逻逻辑控制
                and policy.mode != "patrol_hunt"  # 巡游打怪模式: 方向只由 patrol_direction(边缘)决定, 禁路线/小地图导航接管(否则会发出与巡逻大方向相反的键 → 左右互搏)
                and (args.no_attack or reason in ("patrol", "no_target_box", "no_advisory", "player_missed", "target_ignored"))
                and advisory is not None
                and (args.no_attack or advisory.get("status") in ("NO TARGET", "WAITING", "PLAYER MISSED"))
            ):
                if (
                    route_follower.img_routes
                    and minimap_scan is not None
                    and minimap_scan.get("player_xy")
                ):
                    gxy = route_follower.locate_player(
                        minimap_scan, minimap_scan["player_xy"])
                    if gxy is not None:
                        # Record the walked trail for visualization.
                        if not route_follower.trail or (
                            abs(gxy[0] - route_follower.trail[-1][0]) +
                            abs(gxy[1] - route_follower.trail[-1][1]) > 3
                        ):
                            route_follower.trail.append((int(gxy[0]), int(gxy[1])))
                            if len(route_follower.trail) > route_follower.trail_max:
                                route_follower.trail = route_follower.trail[-route_follower.trail_max:]
                        # Built-in route recording (F1 toggles, F3 saves) is
                        # handled independently below, OUTSIDE the navigation
                        # branch, so it works while manually recording too.
                        route_cmd = route_follower.decide(gxy, now=now)
                        if route_cmd is not None:
                            command, reason = route_cmd
                elif minimap_scan is not None and minimap_scan.get("player_xy"):
                    mm_cmd = minimap_navigator.navigate(
                        minimap_scan, minimap_scan["player_xy"], now=now)
                    if mm_cmd is not None:
                        command, reason = mm_cmd
                elif terrain is not None and not args.no_terrain:
                    terrain_cmd = policy.terrain_decision(player, terrain)
                    if terrain_cmd is not None:
                        command, reason = terrain_cmd

            # 彻底禁用爬绳: 拦截所有来源(路线跟随/小地图/地形)的爬绳命令
            # (小地图坐标航点巡逻 minimap_patrol 例外: 航点里的爬绳动作是用户精确录制的, 允许执行)
            if isinstance(command, str) and command.startswith("climb") \
                    and policy.mode != "minimap_patrol":
                policy.stop_climbing()
                command, reason = "none", "climb_disabled"

            # 任何录制中(主航线/安全点/恢复路线): 抑制 bot 自身移动/攻击,
            # 完全交给用户手动操控(否则 bot 和用户抢按键, 且进行中的行程会
            # 抢键——修复: 用户 F11 录制时角色还被恢复行程拖着走)。
            if (waypoint_patrol.is_recording
                    or waypoint_patrol.is_recording_safe
                    or waypoint_patrol.is_recording_recall):
                command, reason = "none", "wp_recording"

            # 功能键: F1 开始录制 / F2 结束录制 / F3 开始路线巡航 / F4 清除路线.
            # 每轮循环处理一次.
            while True:
                fn = pause_control.pop_fn_event()
                if fn is None:
                    break
                if fn == "f1":
                    # F1: 启动录制(手动打点模式)——清空旧点, 进入录制状态
                    if waypoint_patrol.is_recording_safe:
                        logger.warning("[热键] 安全点录制中, 请先再按 F10 保存安全点")
                        continue
                    if waypoint_patrol.is_recording_recall:
                        logger.warning("[热键] 恢复路线录制中, 请先再按 F11 保存恢复路线")
                        continue
                    _reset_trip_states()   # 全部重新来: 取消安全点/恢复行程
                    recorder.is_recording = True
                    waypoint_patrol.stop_patrol()
                    waypoint_patrol.clear()
                    waypoint_patrol.is_recording = True
                    logger.info("[热键] F1 启动录制: 手动走到点位后按 F2 打普通点 / F3 打跳跃点, F4 保存")
                elif fn == "f2":
                    # F2: 打普通点(安全点/恢复路线录制中打进各自序列, 否则打主航线)
                    if waypoint_patrol.is_recording_safe:
                        _ok = waypoint_patrol.add_manual_point_to(
                            waypoint_patrol.safe_points, "move",
                            mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.safe_points)
                        _tag = ", 安全点"
                    elif waypoint_patrol.is_recording_recall:
                        _ok = waypoint_patrol.add_manual_point_to(
                            waypoint_patrol.recall_points, "move",
                            mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.recall_points)
                        _tag = ", 恢复路线"
                    else:
                        _ok = waypoint_patrol.add_manual_point(
                            "move", mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.waypoints)
                        _tag = ""
                    if _ok:
                        logger.info(
                            f"[热键] F2 普通点已打 (共 {_n} 个{_tag})")
                    else:
                        logger.warning("[热键] F2 打点失败(无小地图坐标或与上一点重合)")
                elif fn == "f3":
                    # F3: 打跳跃点(安全点/恢复路线录制中打进各自序列, 否则打主航线)
                    if waypoint_patrol.is_recording_safe:
                        _ok = waypoint_patrol.add_manual_point_to(
                            waypoint_patrol.safe_points, "jump_takeoff",
                            mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.safe_points)
                        _tag = ", 安全点"
                    elif waypoint_patrol.is_recording_recall:
                        _ok = waypoint_patrol.add_manual_point_to(
                            waypoint_patrol.recall_points, "jump_takeoff",
                            mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.recall_points)
                        _tag = ", 恢复路线"
                    else:
                        _ok = waypoint_patrol.add_manual_point(
                            "jump_takeoff", mini["map_norm"] if mini else None)
                        _n = len(waypoint_patrol.waypoints)
                        _tag = ""
                    if _ok:
                        logger.info(
                            f"[热键] F3 跳跃点已打 (共 {_n} 个{_tag})")
                    else:
                        logger.warning("[热键] F3 打点失败(无小地图坐标或与上一点重合)")
                elif fn == "f4":
                    # F4: 保存录制并开始巡航
                    if waypoint_patrol.is_recording_safe:
                        logger.warning("[热键] 安全点录制中, 请先再按 F10 保存安全点")
                        continue
                    if waypoint_patrol.is_recording_recall:
                        logger.warning("[热键] 恢复路线录制中, 请先再按 F11 保存恢复路线")
                        continue
                    _reset_trip_states()   # F4 = 全部重新来: 取消安全点/恢复行程, 计时从零
                    waypoint_patrol.is_recording = False
                    recorder.is_recording = False
                    if not waypoint_patrol.waypoints:
                        logger.warning("[热键] F4 保存失败: 还没有打点(F2/F3)")
                    else:
                        if active_map_name and waypoint_patrol.save(active_map_name):
                            logger.info(
                                f"[热键] F4 录制已保存 {len(waypoint_patrol.waypoints)} 个点 "
                                f"-> minimaps/{active_map_name}/waypoints.json")
                            if waypoint_patrol.last_archive:
                                logger.info(
                                    f"[热键] 路线存档: {waypoint_patrol.last_archive}")
                        if waypoint_patrol.start_patrol():
                            logger.info(
                                f"[热键] F4 开始路线巡航 ({len(waypoint_patrol.waypoints)} 个点, "
                                f"起点→终点→起点往返)")
                elif fn == "f5":
                    # F5: 清空录制(停巡航 + 清内存点, 磁盘存档保留)
                    _reset_trip_states()   # 全部重新来: 取消安全点/恢复行程
                    recorder.is_recording = False
                    waypoint_patrol.is_recording = False
                    waypoint_patrol.is_recording_safe = False
                    waypoint_patrol.is_recording_recall = False
                    waypoint_patrol.stop_patrol()
                    waypoint_patrol.clear()
                    logger.info("[热键] F5 录制已清空(内存)")
                elif fn == "f6":
                    # F6: 一键重定位(纠偏)——用户手动走到"第一个点位"的位置后按 F6:
                    # 取当前玩家小地图坐标, 与 waypoints[0] 的偏差 = 全局偏移,
                    # 把全部航点坐标改写(加上该偏移), 保存文件并重新加载。
                    _cur = (mini or {}).get("map_norm")
                    if _cur is None:
                        logger.warning("[热键] F6 重定位失败: 无当前小地图坐标(等小地图稳定)")
                    elif not waypoint_patrol.waypoints:
                        logger.warning("[热键] F6 重定位失败: 没有已加载的航点")
                    else:
                        _p0 = waypoint_patrol.waypoints[0]
                        _dx = _cur[0] - float(_p0["nx"])
                        _dy = _cur[1] - float(_p0["ny"])
                        if abs(_dx) < 1e-6 and abs(_dy) < 1e-6:
                            logger.info("[热键] F6 重定位: 已在第一个点位, 无需偏移")
                        else:
                            for _w in waypoint_patrol.waypoints:
                                if "nx" in _w:
                                    _w["nx"] = min(1.0, max(0.0,
                                        round(float(_w["nx"]) + _dx, 6)))
                                if "ny" in _w:
                                    _w["ny"] = min(1.0, max(0.0,
                                        round(float(_w["ny"]) + _dy, 6)))
                            _saved = waypoint_patrol.save(active_map_name)
                            logger.info(
                                f"[热键] F6 重定位完成: 偏移 X{_dx:+.4f} Y{_dy:+.4f}"
                                f" 已作用于 {len(waypoint_patrol.waypoints)} 个航点"
                                f"({'并保存(含时间戳存档)' if _saved else ', 保存失败!'})")
                            # 按用户要求: 改写文件后让应用重新读取(磁盘为准)
                            _reloaded = waypoint_patrol.load_waypoints(active_map_name)
                            _n = len(waypoint_patrol.waypoints) if _reloaded else 0
                            logger.info(
                                f"[热键] F6 重定位后重新加载航点: "
                                f"{'成功, ' + str(_n) + ' 段' if _reloaded else '失败(回退内存点)'}")
                            if _reloaded:
                                waypoint_patrol.idx = 0
                                waypoint_patrol.start_patrol()
                elif fn == "f10":
                    # F10: 安全点录制开关(测谎仪规避: 定时去打怪暂停→走进商城)
                    if waypoint_patrol.is_recording:
                        logger.warning(
                            "[热键] 主航线录制中(F1), 请先 F4 保存/F5 清空再录安全点")
                    elif waypoint_patrol.is_recording_recall:
                        logger.warning("[热键] 恢复路线录制中, 请先再按 F11 保存恢复路线")
                    elif waypoint_patrol.is_recording_safe:
                        waypoint_patrol.is_recording_safe = False
                        if not waypoint_patrol.safe_points:
                            logger.warning("[热键] 安全点保存失败: 没有打点(F2/F3)")
                        elif active_map_name and waypoint_patrol.save_safe_points(active_map_name):
                            logger.info(
                                f"[热键] 安全点已保存 {len(waypoint_patrol.safe_points)} 个 "
                                f"-> minimaps/{active_map_name}/safe_points.json")
                            waypoint_patrol.load_safe_points(active_map_name)
                        else:
                            logger.warning("[热键] 安全点保存失败(无地图名或写入异常)")
                    else:
                        _reset_trip_states()   # 录入安全点=手动操控, 取消进行中的行程
                        waypoint_patrol.is_recording_safe = True
                        waypoint_patrol.safe_points = []
                        logger.info(
                            "[热键] F10 开始录制安全点: 走位按 F2 普通点 / F3 跳跃点, "
                            "走到最后的商城位再按 F10 保存")
                elif fn == "f11":
                    # F11: 恢复路线录制开关(安全点退出商城后/跌落底层时走回巡游线)
                    if waypoint_patrol.is_recording:
                        logger.warning(
                            "[热键] 主航线录制中(F1), 请先 F4 保存/F5 清空再录恢复路线")
                    elif waypoint_patrol.is_recording_safe:
                        logger.warning("[热键] 安全点录制中, 请先再按 F10 保存安全点")
                    elif waypoint_patrol.is_recording_recall:
                        waypoint_patrol.is_recording_recall = False
                        if not waypoint_patrol.recall_points:
                            logger.warning("[热键] 恢复路线保存失败: 没有打点(F2/F3)")
                        elif active_map_name and waypoint_patrol.save_recall_points(active_map_name):
                            logger.info(
                                f"[热键] 恢复路线已保存 {len(waypoint_patrol.recall_points)} 个 "
                                f"-> minimaps/{active_map_name}/recall_points.json")
                            recall_patrol.load_recall_points(active_map_name)
                        else:
                            logger.warning("[热键] 恢复路线保存失败(无地图名或写入异常)")
                    else:
                        _reset_trip_states()   # 录入恢复路线=手动操控, 取消进行中的行程
                        waypoint_patrol.is_recording_recall = True
                        waypoint_patrol.recall_points = []
                        logger.info(
                            "[热键] F11 开始录制恢复路线(从跌落/商城出口位置开始, "
                            "走回巡游线): F2 普通点 / F3 跳跃点, 走到巡游线上再按 F11 保存")
                elif fn == "f7":
                    # F7: 仅保存航点到磁盘(方便下次直接 F3 加载, 不启动巡航)
                    if waypoint_patrol.save(active_map_name):
                        logger.info(
                            f"[waypoint] 航点已保存 {len(waypoint_patrol.waypoints)} 个 "
                            f"-> minimaps/{active_map_name}/waypoints.json")
                    else:
                        logger.warning("[waypoint] 航点保存失败(无航点或未加载地图)")
                # 注意: F8 是全局暂停/恢复键(pause_key), 由 PauseController 监听器
                # 处理——检测到其他玩家暂停后按 F8 恢复挂机, 不经过 fn_events。
                # 清空航点请用 F5。

            # ---- 安全点定时进商城(测谎仪规避): 触发 + 状态机 ----
            # 挂机(F8/红点)期间整个状态机冻结, 恢复后从原阶段继续。
            if not pause_control.is_effectively_paused():
                if _safe_state == "walk":
                    if not waypoint_patrol.is_patrolling():
                        # 巡游被停止(F5/手动): 取消安全点行程
                        logger.warning("[安全点] 巡游已停止, 取消安全点行程")
                        policy._safe_active = False
                        safe_patrol.end_safe_visit()
                        _safe_state = ""
                    elif safe_patrol.is_one_shot_done():
                        executor.release_all()   # 到点了: 松开全部按键, 站定
                        logger.info(
                            f"[安全点] 已走到最后一个安全点, "
                            f"{safe_wait_before_shop:.0f} 秒后按 {safe_shop_key} 进商城")
                        _safe_state = "wait_t"
                        _safe_at = now
                    elif now - _safe_at >= safe_max_trip:
                        # 行程超时(导航卡住/掉到不可达平台): 取消, 恢复巡游,
                        # 下一个安全点时刻顺延——避免永远卡在安全点路上
                        logger.warning(
                            f"[安全点] 走向安全点超时({safe_max_trip:.0f}s), "
                            f"取消行程恢复巡游")
                        policy._safe_active = False
                        safe_patrol.end_safe_visit()
                        _safe_state = ""
                        _safe_next_visit = _next_safe_slot_ts()
                elif _safe_state == "wait_t":
                    if now - _safe_at >= safe_wait_before_shop:
                        press_key(safe_shop_key, 0.15)
                        logger.info(f"[安全点] 已按 {safe_shop_key} 进入商城")
                        _safe_state = "wait_esc"
                        _safe_at = now
                elif _safe_state == "wait_esc":
                    if now - _safe_at >= safe_wait_in_shop:
                        press_key("esc", 0.15)
                        time.sleep(safe_esc_gap)
                        press_key("enter", 0.15)
                        logger.info("[安全点] 按 ESC + 回车返回游戏")
                        _safe_state = "wrap"
                        _safe_at = now
                elif _safe_state == "wrap":
                    if now - _safe_at >= safe_wrap_wait:
                        _safe_state = ""
                        policy._safe_active = False
                        safe_patrol.end_safe_visit()
                        _safe_next_visit = _next_safe_slot_ts()
                        if recall_patrol.recall_points:
                            # 退出商城后人物可能掉到别处(不在巡游点平台):
                            # 走恢复路线回到巡游线
                            if recall_patrol.begin_recall():
                                policy._recall_active = True
                                _recall_state = "walk"
                                _recall_at = now
                                logger.info(
                                    "[恢复路线] 安全点商城返回, 走恢复路线回巡游路线")
                            else:
                                logger.warning("[恢复路线] 触发失败: 恢复路线为空")
                        else:
                            logger.info(
                                f"[安全点] 商城流程完成, 恢复巡游"
                                f"(下次 {datetime.datetime.fromtimestamp(_safe_next_visit).strftime('%H:%M')})")
                else:
                    # 空闲: 安全点时刻 = 每小时 schedule_minutes(整点/半点);
                    # 到点后不立即打断, 而是挂起(_safe_pending), 等主航线
                    # 【走完这一圈、到达最后一个巡游点】时才触发(用户要求:
                    # 防止中途点位走不到安全点)。
                    if not waypoint_patrol.is_patrolling():
                        if _safe_pending:
                            logger.info("[安全点] 巡游已停止, 取消待触发的安全点")
                            _safe_pending = False
                        _last_round_count = waypoint_patrol._round_count
                    elif (waypoint_patrol.is_recording
                          or waypoint_patrol.is_recording_safe
                          or waypoint_patrol.is_recording_recall):
                        _last_round_count = waypoint_patrol._round_count
                    elif (policy.mode == "minimap_patrol"
                          and safe_patrol.safe_points):
                        if _safe_next_visit == 0.0:
                            _safe_next_visit = _next_safe_slot_ts()
                        _rc = waypoint_patrol._round_count
                        if not _safe_pending:
                            if now >= _safe_next_visit:
                                _safe_pending = True
                                logger.info(
                                    f"[安全点] 已到安全点时刻 "
                                    f"({datetime.datetime.fromtimestamp(_safe_next_visit).strftime('%H:%M')}), "
                                    f"等待走完当前这一圈(到最后一个巡游点)再进安全点")
                        elif _rc != _last_round_count:
                            # 刚走完最后一个巡游点(回到起点的瞬间) -> 触发:
                            # 角色正站在最后巡游点, 从此点连接安全点[0]
                            _mnx0 = float(mini["map_norm"][0])
                            _mny0 = float(mini["map_norm"][1])
                            if safe_patrol.begin_safe_visit():
                                policy._safe_active = True
                                _safe_state = "walk"
                                _safe_at = now
                                _safe_pending = False
                                _safe_next_visit = _next_safe_slot_ts()
                                _sp0 = safe_patrol.safe_points[0]
                                logger.info(
                                    f"[安全点] 触发! 从最后巡游点(当前 "
                                    f"{_mnx0:.4f},{_mny0:.4f}) 连接安全点[0]"
                                    f"({float(_sp0['nx']):.4f},{float(_sp0['ny']):.4f}) "
                                    f"→ 走向安全点({len(safe_patrol.safe_points)} 个)")
                            else:
                                logger.warning("[安全点] 触发失败: 安全点序列为空")
                                _safe_pending = False
                        _last_round_count = _rc
                    else:
                        _last_round_count = waypoint_patrol._round_count
            # ---- 恢复路线(安全点退出商城后/跌落底层时走回巡游线) ----
            if _recall_state == "walk":
                if not waypoint_patrol.is_patrolling():
                    logger.warning("[恢复路线] 巡游已停止, 取消恢复行程")
                    policy._recall_active = False
                    recall_patrol.end_recall()
                    _recall_state = ""
                elif recall_patrol.is_one_shot_done():
                    _recall_state = "done_wait"
                    _recall_at = now
                    logger.info("[恢复路线] 已走回, 稍后恢复正常巡游")
                elif now - _recall_at >= recall_max_trip:
                    # 恢复行程超时(卡住/走不回): 取消恢复巡游, 冷却后另判
                    logger.warning(
                        f"[恢复路线] 恢复行程超时({recall_max_trip:.0f}s), "
                        f"取消行程恢复巡游")
                    policy._recall_active = False
                    recall_patrol.end_recall()
                    _recall_state = ""
                    _recall_cooldown_until = now + recall_cooldown
            elif _recall_state == "done_wait":
                if now - _recall_at >= 1.0:
                    _recall_state = ""
                    policy._recall_active = False
                    recall_patrol.end_recall()
                    _recall_cooldown_until = now + recall_cooldown
                    logger.info(
                        f"[恢复路线] 恢复完成, 恢复正常巡游({recall_cooldown:.0f}s冷却)")
            else:
                # 跌落触发(用户要求: 只按 Y 判定): 玩家Y与恢复路线第一个点
                # 同水平(±trigger_y_tol) == 掉到最下层, 立刻走恢复路线——
                # 不再要求主航线目标在上方(那会漏判: 掉下来后主航线仍巡航,
                # 用户反馈"掉下来还是巡航状态"非常严重)。
                if (policy.mode == "minimap_patrol"
                        and waypoint_patrol.is_patrolling()
                        and not waypoint_patrol.is_recording
                        and not waypoint_patrol.is_recording_safe
                        and not waypoint_patrol.is_recording_recall
                        and not _safe_state
                        and recall_patrol.recall_points
                        and mini is not None and mini.get("map_norm")
                        and now >= _recall_cooldown_until):
                    _rny = float(recall_patrol.recall_points[0]["ny"])
                    _ny = float(mini["map_norm"][1])
                    if abs(_ny - _rny) <= recall_y_tol:
                        if recall_patrol.begin_recall():
                            policy._recall_active = True
                            _recall_state = "walk"
                            _recall_at = now
                            logger.warning(
                                f"[恢复路线] 跌落检测! 玩家Y={_ny:.4f} 与恢复路线"
                                f"起点Y={_rny:.4f} 同水平(±{recall_y_tol}), "
                                f"立即走恢复路线回巡游线")
                        else:
                            logger.warning("[恢复路线] 触发失败: 恢复路线为空")
                # 商城/恢复站定阶段: 不打怪不移动
                if _safe_state in ("wait_t", "wait_esc", "wrap") or _recall_state == "done_wait":
                    command, reason = "none", "safe_store"

            # 【安全点/恢复路线行程中绝对禁攻击】(最终兜底): 任何来源的攻击
            # 命令(含未来改动/覆盖逻辑引入的)在行程中一律不执行——
            # 用户要求: 安全WALK/恢复WALK 不打怪不追怪只走路。
            if (policy._safe_active or policy._recall_active):
                if isinstance(command, str) and command.startswith("attack"):
                    command, reason = "none", "trip_no_attack"

            if args.foreground_gate and not target_is_foreground(
                cfg["game_window"]["title"]
            ):
                command, reason = "none", "not_foreground"

            if pause_control.is_effectively_paused():
                command, reason = "none", (
                    "player_pause" if pause_control.player_pause else "paused")
                executor.release_all()
                policy.stop_climbing()

            if args.dry_run or not args.foreground_gate or target_is_foreground(
                cfg["game_window"]["title"]
            ):
                executor.execute(
                    command,
                    reason,
                    hp_percent=vitals[0],
                    mp_percent=vitals[1],
                    now=now,
                    suppress_feed=pause_control.player_pause,
                )
            last_decision, last_reason = command, reason

            if last_frame_time is not None:
                instant = 1.0 / max(now - last_frame_time, 1e-6)
                measured_fps = instant if measured_fps == 0 else measured_fps * 0.85 + instant * 0.15
            last_frame_time = now

            if show_window:
                if getattr(args, "show_grid", False):
                    draw_coordinate_grid(frame, step=50, major=100)
                # 注意: draw_attack_range 在 draw_detections 之后调用(见下方),
                # 让攻击范围框画在玩家检测框之上, 避免被 70x90 玩家框遮挡。
                # ---- 小地图坐标 HUD(醒目显示玩家在地图里的位置, 替代误检的名字框坐标) ----
                _mini_hud = policy._mini
                if _mini_hud is not None and _mini_hud.get("map_norm"):
                    _mnx, _mny = _mini_hud["map_norm"]
                    if _mnx > policy._mini_right_norm:
                        _pos = "右端→往左走"
                    elif _mnx < policy._mini_left_norm:
                        _pos = "左端→往右走"
                    else:
                        _pos = "中间"
                    put_text_cn(
                        frame,
                        f"小地图 norm=({_mnx:.4f},{_mny:.4f})  玩家在地图{_pos}",
                        (14, 24), 0.62, (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    # 顶部醒目显示录制/巡航状态(F1~F4 热键反馈)
                    _wp_hud = getattr(policy, "_waypoint_patrol", None)
                    if _wp_hud is not None:
                        # 【录制可视化】: 显示正在录什么点位 + 数量 + 最近一点,
                        # 并把已录的点画在小地图上(蓝=主航线, 橙=安全点, 粉=恢复路线)
                        _rec_list = None
                        _rec_color = None
                        if _wp_hud.is_recording:
                            _n = len(waypoint_patrol.waypoints)
                            _last = waypoint_patrol.waypoints[-1] if _n else None
                            _ls = (f" 最近({float(_last['nx']):.3f},{float(_last['ny']):.3f})"
                                   if _last else "")
                            _hud_txt, _hud_col = (
                                f"● 录制:主航线 已{_n}点{_ls} (F2普通/F3跳跃/F4保存)",
                                (80, 255, 120))
                            _rec_list, _rec_color = waypoint_patrol.waypoints, (0, 255, 255)
                        elif _wp_hud.is_recording_safe:
                            _n = len(waypoint_patrol.safe_points)
                            _last = waypoint_patrol.safe_points[-1] if _n else None
                            _ls = (f" 最近({float(_last['nx']):.3f},{float(_last['ny']):.3f})"
                                   if _last else "")
                            _hud_txt, _hud_col = (
                                f"● 录制:安全点 已{_n}点{_ls} (F2普通/F3跳跃/F10保存)",
                                (0, 165, 255))
                            _rec_list, _rec_color = waypoint_patrol.safe_points, (0, 165, 255)
                        elif _wp_hud.is_recording_recall:
                            _n = len(waypoint_patrol.recall_points)
                            _last = waypoint_patrol.recall_points[-1] if _n else None
                            _ls = (f" 最近({float(_last['nx']):.3f},{float(_last['ny']):.3f})"
                                   if _last else "")
                            _hud_txt, _hud_col = (
                                f"● 录制:恢复路线 已{_n}点{_ls} (F2普通/F3跳跃/F11保存)",
                                (180, 105, 255))
                            _rec_list, _rec_color = waypoint_patrol.recall_points, (180, 105, 255)
                        elif _wp_hud.is_patrolling():
                            _hud_txt, _hud_col = "● 巡航中 (自动, F4 停止)", (0, 255, 160)
                        else:
                            _hud_txt, _hud_col = "○ 待命 (F1 录 / F3 巡航)", (210, 210, 120)
                        if _safe_state:
                            _hud_txt += f" | 安全点:{_safe_state}"
                        if _recall_state:
                            _hud_txt += f" | 恢复:{_recall_state}"
                        put_text_cn(frame, _hud_txt, (14, 46), 0.55, _hud_col, 2, cv2.LINE_AA)
                        # 已录点位画到小地图上(norm -> 画布像素: frame = canvas + norm*canvas_size)
                        if (_rec_list is not None and _mini_hud is not None
                                and _mini_hud.get("canvas_frame_box")
                                and _mini_hud.get("canvas_size")):
                            _cb = _mini_hud["canvas_frame_box"]
                            _cs = _mini_hud["canvas_size"]
                            _pts = []
                            for _rp in _rec_list:
                                _px = int(_cb[0] + float(_rp["nx"]) * _cs[0])
                                _py = int(_cb[1] + float(_rp["ny"]) * _cs[1])
                                _pts.append((_px, _py))
                            for _i in range(max(0, len(_pts) - 1)):
                                cv2.line(frame, _pts[_i], _pts[_i + 1],
                                         _rec_color, 1, cv2.LINE_AA)
                            for _p in _pts:
                                cv2.circle(frame, _p, 3, _rec_color, -1, cv2.LINE_AA)
                            if _pts:  # 最后一个点绿色圈标出(当前/最近打的点)
                                cv2.circle(frame, _pts[-1], 6, (0, 255, 0), 1, cv2.LINE_AA)
                        # 游戏窗口聚焦状态(角色不动常因游戏未聚焦: 按键发到了别的窗口)
                        _fg = target_is_foreground(cfg["game_window"]["title"])
                        _fg_txt = "游戏窗口: 聚焦 ✓" if _fg else "游戏窗口: 未聚焦 ✗ (自动置前中)"
                        _fg_col = (0, 230, 0) if _fg else (0, 130, 255)
                        put_text_cn(frame, _fg_txt, (14, 66), 0.5, _fg_col, 2, cv2.LINE_AA)
                    # 在小地图玩家标记处画十字圈(直观显示小地图定位到的玩家位置)
                    _fx, _fy = _mini_hud.get("frame_px", (0, 0))
                    if 0 <= _fx < frame.shape[1] and 0 <= _fy < frame.shape[0]:
                        cv2.circle(frame, (int(_fx), int(_fy)), 9, (0, 255, 255), 2)
                        cv2.line(frame, (int(_fx) - 14, int(_fy)), (int(_fx) + 14, int(_fy)), (0, 255, 255), 1)
                        cv2.line(frame, (int(_fx), int(_fy) - 14), (int(_fx), int(_fy) + 14), (0, 255, 255), 1)
                # ---- 其他玩家(小地图红点)状态 HUD ----
                # R1/R2 = 已确认的其他玩家红点(跨帧确认后); 挂机时显示"挂机中"
                if len(_red_dots) > 0:
                    _rd_txt = " ".join(
                        f"R{i+1}({int(p['map_px'][0])},{int(p['map_px'][1])})"
                        for i, p in enumerate(_red_dots[:4]))
                    put_text_cn(
                        frame, f"其他玩家红点 {len(_red_dots)}个: {_rd_txt}",
                        (14, 88), 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                if pause_control.player_pause:
                    put_text_cn(
                        frame, "■ 挂机中(检测到其他玩家, 仅喝药; F8 恢复)",
                        (14, 108), 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                draw_detections(frame, player, cached_monsters, advisory)
                # 攻击范围+巡游打怪框画在检测框之后(最上层), 防被玩家框遮挡
                draw_attack_range(frame, player, policy)
                if route_follower.img_routes and minimap_scan is not None:
                    draw_route_overlay(
                        frame, minimap_scan, route_follower,
                        minimap_scan.get("player_xy"))
                # Route map thumbnail overlaid on the main window (shows
                # WHICH route is active + player dot + trail).
                if route_follower.img_map is not None:
                    draw_route_map_window(
                        frame, route_follower, recorder, minimap_scan,
                        minimap_scan.get("player_xy") if minimap_scan else None,
                        map_name=active_map_name)
                # Full live status panel + mouse-clickable buttons.
                combined, ui_buttons = render_info_panel(
                    frame,
                    player,
                    cached_monsters,
                    vitals,
                    advisory,
                    policy,
                    executor,
                    command,
                    reason,
                    fps_limit,
                    measured_fps,
                    pause_control.paused,
                    args.player_name,
                    route_follower=route_follower,
                    recorder=recorder,
                    map_name=active_map_name,
                    no_attack=args.no_attack,
                    editor=param_editor,
                    bind_target=pause_control.bind_target,
                    exp_summary=exp_stats._summary(),
                )
                cv2.imshow(WINDOW_TITLE, combined)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc: cancel binding/editing, do not quit
                    if pause_control.is_binding():
                        pause_control.cancel_bind()
                    if param_editor.active:
                        param_editor.active = False
                        param_editor.buffer = ""
                        param_editor.editing_index = -1
                elif key == ord("q"):
                    break
                param_editor.handle_key(key)

            # Process mouse clicks from the panel buttons.
            for _ in range(len(click_actions)):
                if not click_actions:
                    break
                action = click_actions.pop(0)
                if action == "pause":
                    pause_control.toggle()
                elif action == "rec_toggle":
                    recorder.is_recording = not recorder.is_recording
                    logger.info(f"[recorder] recording = {recorder.is_recording}")
                elif action == "rec_save":
                    if recorder.save_route(active_map_name, args.route_name or None):
                        logger.info(f"[recorder] route saved (mouse): {recorder.last_saved_path}")
                        # Reload so the route-map window shows the new route
                        # and the bot can follow it immediately.
                        if active_map_name:
                            route_follower.load_map_routes(active_map_name)
                elif action == "rec_save_map":
                    if recorder.save_map(active_map_name):
                        logger.info("[recorder] map saved (mouse)")
                elif action in ("bind_attack", "bind_hp", "bind_mp"):
                    pause_control.start_bind(action.replace("bind_", ""))
                elif action.startswith("param_"):
                    # Panel param buttons map to editor fields 3..8 (fields
                    # 0-2 are the key-binding entries).
                    param_editor.pick(int(action.split("_")[1]) + 3)
            # Apply a captured key binding (click button -> press any key).
            bind_result = pause_control.pop_bind_result()
            if bind_result is not None:
                target, key_name = bind_result
                if target == "attack":
                    executor.attack_key = key_name
                    logger.info(f"[bind] attack key -> {key_name}")
                elif target == "hp":
                    executor.add_hp_key = key_name
                    logger.info(f"[bind] HP potion key -> {key_name}")
                elif target == "mp":
                    executor.add_mp_key = key_name
                    logger.info(f"[bind] MP potion key -> {key_name}")

            if args.max_frames and frame_count >= args.max_frames:
                break
            if args.duration > 0 and now - started >= args.duration:
                break

            target_duration = 1.0 / fps_limit
            elapsed = time.time() - now
            if elapsed < target_duration:
                time.sleep(target_duration - elapsed)
    finally:
        executor.release_all()
        if hasattr(player_detector, "stop"):
            player_detector.stop()
        async_monster_detector.stop()
        capture.stop()
        if show_window:
            cv2.destroyAllWindows()

    if args.snapshot and player is not None:
        from tools.live_perception_viewer import save_image

        if getattr(args, "show_grid", False):
            draw_coordinate_grid(frame, step=50, major=100)
        draw_detections(frame, player, cached_monsters, advisory)
        draw_attack_range(frame, player, policy)   # 画在检测框之上
        snapshot_frame = render_info_panel(
            frame,
            player,
            cached_monsters,
            vitals,
            advisory,
            policy,
            executor,
            last_decision,
            last_reason,
            fps_limit,
            measured_fps,
            pause_control.paused,
            args.player_name,
        )
        save_image(args.snapshot, snapshot_frame)

    summary = {
        "mode": "auto_combat",
        "dry_run": args.dry_run,
        "monster_backend": args.monster_backend,
        "foreground_gate": args.foreground_gate,
        "frames": frame_count,
        "elapsed_seconds": round(time.time() - started, 2),
        "paused_at_end": pause_control.paused,
        "player_found": player is not None,
        "monsters": len(cached_monsters),
        "vitals": {
            "hp_percent": None if vitals[0] is None else round(float(vitals[0]), 2),
            "mp_percent": None if vitals[1] is None else round(float(vitals[1]), 2),
            "exp_percent": None if vitals[2] is None else round(float(vitals[2]), 2),
        },
        "advisory": None if advisory is None else {
            "status": advisory["status"],
            "attack_ready": advisory["attack_ready"],
            "dodge_risk": advisory["dodge_risk"],
            "target_label": advisory["target_label"],
            "distance_px": None if advisory["distance_px"] is None else round(advisory["distance_px"], 2),
        },
        "last_command": last_decision,
        "last_reason": last_reason,
        "input_issued": not args.dry_run,
        "action_counts": executor.counts,
    }
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
