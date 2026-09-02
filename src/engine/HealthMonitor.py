'''
Health Monitor take game windows as input to calculate player's HP/MP/EXP Bar percentage.
When player's HP/MP drop to specific threshold, it'd press key to drink potion 
'''

# Standard Import
import threading
import time
import cv2
import numpy as np

# Local Import
from src.utils.logger import logger
from src.utils.common import get_bar_percent
from src.input.KeyBoardController import press_key

class HealthMonitor:
    '''
    Independent health monitoring thread that can heal while other actions are running
    '''
    def __init__(self, cfg, kb_controller):
        self.cfg = cfg
        self.kb = kb_controller
        self.is_terminated = False
        self.enabled = True
        self.thread = None # health monitor thread

        # Health monitoring state
        self.hp_percent = 100
        self.mp_percent = 100
        self.exp_percent = 100

        # Timers
        self.t_last_heal = 0
        self.t_last_mp = 0
        self.t_last_hp_reduce = 0
        self.t_last_run = 0
        self.t_hp_watch_dog = time.time()

        # Frame data (will be updated by main thread)
        self.img_frame = None
        self.frame_lock = threading.Lock()

        # FPS settings
        self.fps_limit = self.cfg["health_monitor"]["fps_limit"]
        self.fps = 0

        # Debug information
        # hp/mp/exp bars loc and size, [(x,y,w,h), ...]
        self.loc_size_bars = [(0, 0, 0, 0),
                              (0, 0, 0, 0),
                              (0, 0, 0, 0)]

        logger.info("[Health Monitor] Init done")

    def start(self):
        '''
        Start health monitoring thread
        '''
        if not self.is_terminated:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("[Health Monitor] Started")

    def stop(self):
        '''
        Stop health monitoring thread
        '''
        self.is_terminated = True
        if self.thread:
            self.thread.join()
            logger.info("[Health Monitor] Terminated")

    def enable(self):
        '''
        Enable health monitoring
        '''
        self.enabled = True

    def disable(self):
        '''
        Disable health monitoring
        '''
        self.enabled = False

    def update_frame(self, img_frame):
        '''
        Update frame data from main thread
        '''
        with self.frame_lock:
            self.img_frame = img_frame

    def get_hp_mp_exp_percent(self):
        '''
        Extracts the player's HP, MP, and EXP ratios from game frame.

        This function:
        - Crops the predefined HP, MP, and EXP bar regions from the game frame.
        - Identifies empty areas in each bar.
        - Computes the fill ratio for each bar as: 1 - (empty_pixels / total_pixels).

        Returns:
            tuple: (hp_percent, mp_percent, exp_percent), each a float between 0 and 1.
        '''
        if self.img_frame is None:
            return None, None, None

        with self.frame_lock:
            img_frame = self.img_frame.copy()

        configured_result = self._get_configured_bar_percentages(img_frame)
        if configured_result is not None:
            return configured_result

        img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(img_frame_gray, 240, 255)
        # cv2.imshow("white_mask", white_mask)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        loc_size_bars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # for game window resolution 752x1282, w/h == 7.5, w*h == 3630
            if 4 < w/h < 10 and 2500 < w*h < 5000:
                loc_size_bars.append((x, y, w, h))

        # sort contours by x coordinate
        loc_size_bars = sorted(loc_size_bars, key=lambda bar: bar[0])
        if len(loc_size_bars) != 3:
            return (None, None, None)

        # Update loc_size_bars
        self.loc_size_bars = loc_size_bars

        # Get bar filled ratio
        percent_bars = []
        for x, y, w, h in loc_size_bars:
            percent_bars.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
        return percent_bars

    def _get_configured_bar_percentages(self, img_frame):
        regions = self.cfg["health_monitor"].get("bar_regions")
        colors = self.cfg["health_monitor"].get("bar_hsv_ranges")
        reference_size = self.cfg["health_monitor"].get("bar_reference_size")
        if not regions or not colors or not reference_size:
            return None

        frame_h, frame_w = img_frame.shape[:2]
        reference_w, reference_h = reference_size
        scale_x = frame_w / float(reference_w)
        scale_y = frame_h / float(reference_h)
        min_column_fraction = self.cfg["health_monitor"].get(
            "bar_min_column_fraction", 0.25)

        percentages = []
        detected_regions = []
        for name in ("hp", "mp", "exp"):
            x, y, width, height = regions[name]
            x = int(round(x * scale_x))
            y = int(round(y * scale_y))
            width = max(1, int(round(width * scale_x)))
            height = max(1, int(round(height * scale_y)))

            x1 = min(frame_w, x + width)
            y1 = min(frame_h, y + height)
            if x < 0 or y < 0 or x >= x1 or y >= y1:
                return (None, None, None)

            bar = img_frame[y:y1, x:x1]
            hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
            lower, upper = colors[name]
            mask = cv2.inRange(hsv, tuple(lower), tuple(upper))
            minimum_pixels = max(1, int(round(mask.shape[0] * min_column_fraction)))
            filled_columns = np.count_nonzero(
                np.count_nonzero(mask, axis=0) >= minimum_pixels)
            percentages.append(100.0 * filled_columns / mask.shape[1])
            detected_regions.append((x, y, x1 - x, y1 - y))

        self.loc_size_bars = detected_regions
        return tuple(percentages)

    def _monitor_loop(self):
        '''
        Main monitoring loop running in separate thread
        '''
        while not self.is_terminated:
            try:
                if not self.enabled:
                    self.limit_fps()
                    continue

                # Get current time
                t_cur = time.time()

                # Get current HP/MP ratios
                hp_percent, mp_percent, exp_percent = self.get_hp_mp_exp_percent()
                if hp_percent is not None:
                    # Check if HP bar has reduced
                    if self.hp_percent > hp_percent:
                        self.t_last_hp_reduce = t_cur
                    self.hp_percent = hp_percent
                if mp_percent is not None:
                    self.mp_percent = mp_percent
                if exp_percent is not None:
                    self.exp_percent = exp_percent

                hp_thres = self.cfg["health_monitor"]["add_hp_percent"]
                mp_thres = self.cfg["health_monitor"]["add_mp_percent"]
                hp_cd    = self.cfg["health_monitor"]["add_hp_cooldown"]
                mp_cd    = self.cfg["health_monitor"]["add_mp_cooldown"]
                watchdog_timeout = self.cfg["health_monitor"]["return_home_watch_dog_timeout"]

                # Check if need to heal (with cooldown)
                if self.cfg["health_monitor"]["force_heal"]:
                    # Ignore cooldown and force keycontroller to heal first
                    if self.hp_percent < hp_thres:
                        if not self.kb.is_need_force_heal:
                            logger.info(f"[Health Monitor] Force heal triggered, "
                                        f"HP: {self.hp_percent:.1f}%")
                        self.kb.is_need_force_heal = True
                    else:
                        self.kb.is_need_force_heal = False
                else:
                    if (self.hp_percent <= hp_thres and
                        t_cur - self.t_last_heal > hp_cd):
                        self._heal()
                        logger.info(f"[Health Monitor] Auto heal triggered, HP: {self.hp_percent:.1f}%")
                        self.t_last_heal = t_cur

                # Check if no HP potion and need to return home
                if self.cfg["health_monitor"]["return_home_if_no_potion"]:
                    if self.hp_percent >= hp_thres:
                        self.t_hp_watch_dog = t_cur # reset watchdog
                    else:
                        # If watchdog timeout, use homing scroll to return home
                        if t_cur - self.t_hp_watch_dog > watchdog_timeout:
                            logger.warning(f"[Health Monitor] HP({self.hp_percent:.1f}%) < {hp_thres:.1f}% "
                                           f"for {round(t_cur - self.t_hp_watch_dog, 2)} seconds.")
                            logger.warning(f"[Health Monitor] Return home because potion is used up.")
                            press_key(self.cfg["key"]["return_home"]) # Return home
                            self.is_terminated = True # Terminate Health monitor
                            self.kb.is_terminated = True # Terminate AutoBot

                # Check if need MP (with cooldown)
                if (self.mp_percent <= mp_thres and t_cur - self.t_last_mp > mp_cd):
                    self._add_mp()
                    self.t_last_mp = t_cur
                    logger.info(f"[Health Monitor] Auto MP triggered, MP: {self.mp_percent:.1f}%")

                # Sleep to avoid excessive CPU usage
                self.limit_fps()

            except Exception as e:
                logger.error(f"[Health Monitor] {e}")
                self.limit_fps()

    def _heal(self):
        '''
        Execute heal action
        '''
        try:
            press_key(self.cfg["key"]["add_hp"], 0.05)
        except Exception as e:
            logger.error(f"[Health Monitor] Heal action failed: {e}")

    def _add_mp(self):
        '''
        Execute MP recovery action
        '''
        try:
            press_key(self.cfg["key"]["add_mp"], 0.05)
        except Exception as e:
            logger.error(f"[Health Monitor] MP action failed: {e}")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
