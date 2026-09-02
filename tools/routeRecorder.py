'''
Auto generate route map 
'''
# Standard import
import time
import argparse
import sys
import os
import shutil

# CV import
import numpy as np
import cv2

# local import
import os as _os
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
_LOCAL_RUNTIME = _REPO_ROOT / ".yolo_runtime"
for _p in (
    _REPO_ROOT,
    _LOCAL_RUNTIME,
    _LOCAL_RUNTIME / "win32",
    _LOCAL_RUNTIME / "win32" / "lib",
):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_WIN32_DLL_HANDLE = None
if _os.name == "nt" and (_LOCAL_RUNTIME / "pywin32_system32").exists():
    _WIN32_DLL_HANDLE = _os.add_dll_directory(
        str(_LOCAL_RUNTIME / "pywin32_system32"))

from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.text_cn import put_text_cn
from src.utils.common import (
    find_pattern_sqdiff, draw_rectangle, screenshot,
    get_minimap_loc_size_compat, get_player_location_on_minimap,
    to_opencv_hsv, load_yaml, override_cfg, is_mac, load_image,
)
from src.input.KeyBoardListener import KeyBoardListener
from src.input.GameWindowCapturor import GameWindowCapturor


def _imwrite_cn(path, img):
    """Write an image to a possibly non-ASCII path.

    cv2.imwrite fails on paths containing non-ASCII characters in some
    builds (it uses a C-style fopen internally). Encode via cv2.imencode and
    write the bytes with Python's open(), which handles Unicode paths.
    Returns True on success.
    """
    try:
        ext = _Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


class RouteRecorder():
    '''
    Route recorder
    '''
    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 26
        text_y_start = 490
        dt_screenshot = time.time() - self.kb.t_func_key[1]
        dt_save_route = time.time() - self.kb.t_func_key[2]
        dt_save_map = time.time() - self.kb.t_func_key[3]
        # Recording status badge (big, colored)
        status_text = "● 录制中" if self.is_enable else "○ 已暂停"
        status_color = (0, 0, 255) if self.is_enable else (0, 160, 255)
        put_text_cn(
            self.img_frame_debug, status_text, (10, 40),
            1.2, status_color, 3, cv2.LINE_AA
        )
        # Map name badge
        if getattr(self, "map_name", ""):
            put_text_cn(
                self.img_frame_debug,
                f"地图: {self.map_name}",
                (10, 75), 0.8,
                (0, 255, 255), 2, cv2.LINE_AA
            )
        text_list = [
            f"FPS: {self.fps}",
            f"已记录: {self._route_pixel_count()} 个动作点",
            f"【F1】 {'暂停录制' if self.is_enable else '开始录制'}"
            + ("  - 已保存" if dt_save_route < 0.7 else ""),
            f"【F3】 保存当前路线"
            + ("  - 已保存!" if dt_save_route < 0.7 else ""),
            f"【F4】 保存地图底图"
            + ("  - 已保存!" if dt_save_map < 0.7 else ""),
            f"【F2】 截图",
            f"【Q】  退出录制",
            "",
            "操作说明: 在游戏里走动,经过绳子按↑、",
            "需要跳过的地方按空格(跳跃),走完按 F3 保存。",
            "",
            "提示: 走路时'已记录'数字会增加就说明录上了;",
            "原地不动数字不变是正常的。",
        ]
        for idx, text in enumerate(text_list):
            put_text_cn(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                0.62, (0, 255, 0),
                2, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=1
        )

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 70
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        if not getattr(self, "show_debug", False):
            return
        cv2.imshow("Game Window Debug",
                   self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap)
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0],
            self.loc_minimap_global[1] + self.loc_player_minimap[1]
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def replace_color_on_map(self, lower_hsv, upper_hsv, replace_color=(0, 0, 0)):
        '''
        Replace pixels in self.img_map that fall within the given HSV range
        and are part of a connected component with area > 15.
        '''
        hsv_map = cv2.cvtColor(self.img_map, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_map, to_opencv_hsv(lower_hsv), to_opencv_hsv(upper_hsv))

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10:
                component_mask = (labels == i)
                self.img_map[component_mask] = replace_color

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # The capture already returns the play-area frame at the configured
        # game size (GameWindowCapturor resizes the window). Use it as-is:
        # resizing here would shift the minimap region (config minimap.region
        # is in original-frame coordinates), breaking minimap/OCR alignment.
        return self.frame

    def __init__(self, args):
        '''
        Init MapleStoryBot
        '''
        self.args = args # User arguments
        self.idx_routes = 0 # Index of route map
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = True
        # Coordinate (top-left coordinate)
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_player_global_last = None # playeer location on global map last frame
        self.loc_player_global_prev = None # previous global position (walking detection)
        self.loc_player_global_smooth = None # EMA-smoothed global position
        self._walk_dir = 0 # last confirmed walking direction (-1/0/+1)
        self.minimap_nav = None # lazy MinimapNavigator (player detection)
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = None # minimap on game screen
        self.img_map = None # map
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_draw_blob = time.time() # Last draw blob timer
        self._log_tick = 0 # headless progress-log counter

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
        self.show_debug = bool(getattr(args, "show_debug", False))

        # Parse color_code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }
        self.color_code.update(color_code_up_down) # Combine both dictionaries

        self.fps_limit = self.cfg["system"]["fps_limit_route_recorder"]

        # Map directory: prefer the user-supplied name, else auto-detect the
        # map name from the minimap via OCR. Append mode is default so a
        # route can be extended across sessions.
        if args.new_map and args.new_map != 'new_map':
            self.map_name = args.new_map
        else:
            self.map_name = ""  # detected on the first frame
        self.map_dir = ""
        self._ensure_map_dir()

        # Load exist map
        if self.args.map != '':
            self.img_map = load_image(f"{self.args.map}")

        # Start keyboard listener thread
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

    def _ensure_map_dir(self, map_name=None):
        """Create/keep the minimaps/{map_name} directory for the current map."""
        if map_name is not None:
            self.map_name = map_name
        if not self.map_name:
            return
        self.map_dir = os.path.join("minimaps", self.map_name)
        os.makedirs(self.map_dir, exist_ok=True)
        logger.info(f"[routeRecorder] Map directory ready: {self.map_dir}")

    def detect_map_name(self, img_frame):
        """
        Auto-detect the map name from the minimap header via OCR. Returns the
        detected name or None.
        """
        try:
            from src.utils.minimap_ocr import get_minimap_map_name
            ocr_region = self.cfg["minimap"].get(
                "ocr_region", self.cfg["minimap"].get(
                    "region", [0, 0, 285, 245]))
            return get_minimap_map_name(img_frame, region=ocr_region)
        except Exception as e:
            logger.warning(f"[routeRecorder] map-name OCR failed: {e}")
            return None

    def _detect_player_on_minimap(self):
        """
        Locate the player's yellow arrow inside the minimap crop.

        The legacy get_player_location_on_minimap() was tuned for the
        international client (white-border minimap) and mis-detects UI buttons
        on the Shanda legacy client. We delegate to the auto-combat bot's
        MinimapNavigator, which uses bright-yellow blob detection PLUS a
        walkable-terrain constraint (the player stands on ground/rock), so the
        detection matches exactly what the bot sees. Returns (cx, cy) in
        minimap coordinates or None.
        """
        try:
            from tools.auto_combat import MinimapNavigator
            if self.minimap_nav is None:
                self.minimap_nav = MinimapNavigator(self.cfg)
            scan = self.minimap_nav.scan(self.img_frame, time.time())
            player_full = scan.get("player_xy")
            if player_full is None:
                return None
            region = self.cfg["minimap"].get("region", [0, 0, 285, 245])
            x0, y0 = int(region[0]), int(region[1])
            return (player_full[0] - x0, player_full[1] - y0)
        except Exception as exc:
            logger.warning(f"[routeRecorder] player detect failed: {exc}")
            return None

    def _count_existing_routes(self):
        """Number of route*.png already saved for the current map."""
        if not self.map_dir or not os.path.isdir(self.map_dir):
            return 0
        count = 0
        for name in os.listdir(self.map_dir):
            if name.startswith("route") and name.endswith(".png") \
                    and "rest" not in name:
                count += 1
        return count

    def _route_pixel_count(self):
        """Number of non-background pixels in the current route image.

        Walking / climbing / jumping color codes are drawn on top of the map
        base image, so counting pixels that differ from the base map tells us
        how much action has been recorded so far. The route image may be
        smaller than the map (the map grows as the player explores), so we
        compare against the map region matching the route image size.
        """
        if self.img_route is None or self.img_map is None:
            return 0
        h, w = self.img_route.shape[:2]
        map_region = self.img_map[:h, :w]
        if map_region.shape != self.img_route.shape:
            return 0
        diff = cv2.absdiff(self.img_route, map_region)
        return int(np.count_nonzero(np.any(diff > 20, axis=2)))

    def ensure_img_map_capacity(self, x, y, h, w):
        '''
        Ensure that self.img_map is large enough to contain the region defined by (x, y, h, w).
        Always add at least "map_padding" when expanding in any direction.
        '''
        map_h, map_w = self.img_map.shape[:2]
        pad = self.cfg["route_recoder"]["map_padding"]

        # Compute required expansion margins
        expand_top = pad - y if y < pad else 0
        expand_left = pad - x if x < pad else 0
        expand_bottom = y + h + pad - map_h if y + h + pad > map_h else 0
        expand_right = x + w + pad - map_w if x + w + pad > map_w else 0
        expand_top = max(0, expand_top)
        expand_left = max(0, expand_left)
        expand_bottom = max(0, expand_bottom)
        expand_right = max(0, expand_right)
        # If no expansion needed, return
        if expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
            return

        # Create new canvas and paste old image
        new_h = map_h + expand_top + expand_bottom
        new_w = map_w + expand_left + expand_right
        new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        new_map[expand_top:expand_top + map_h, expand_left:expand_left + map_w] = self.img_map
        self.img_map = new_map

        # Also update all global coordinates that depend on the map (optional)
        self.loc_minimap_global = (
            self.loc_minimap_global[0] + expand_left,
            self.loc_minimap_global[1] + expand_top
        )

    def remove_color_code_pixels(self, img):
        """
        Set all pixels in self.img_map to black if they match any color in color_code (assumed RGB).
        """
        for rgb in self.color_code.keys():
            bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB → BGR
            mask = np.all(img == bgr, axis=2)
            img[mask] = (0, 0, 0)
        return img

    def update_minimap(self):
        '''
        update_minimap
        '''

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        img_frame = self.get_img_frame()
        if img_frame is None:
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get minimap from game window
        if self.is_first_frame:
            region = self.cfg["minimap"].get("region")
            minimap_result = get_minimap_loc_size_compat(
                self.img_frame, fallback_region=region)
            if minimap_result is None:
                logger.error("Minimap not found in the game frame.")
                return -1

            # Auto-detect the map name from the minimap header and create the
            # per-map directory so route/map files land in the right folder.
            if not self.map_name:
                detected = self.detect_map_name(self.img_frame)
                if detected:
                    logger.info(f"[routeRecorder] Detected map: {detected}")
                    self._ensure_map_dir(detected)
                else:
                    logger.warning(
                        "[routeRecorder] Could not detect map name; "
                        "route will be saved under 'unknown_map'.")
                    self._ensure_map_dir("unknown_map")

            x, y, w, h = minimap_result
            # Discard 1 pixel boundary of the minimap when a border was found
            if region is None:
                x += 1
                y += 1
                w -= 2
                h -= 2
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y+h, x:x+w]
        else:
            x, y = self.loc_minimap
            h, w = self.img_minimap.shape[:2]
            self.img_minimap = self.img_frame[y:y+h, x:x+w]

        # Replace black pixels (0, 0, 0) with (1, 1, 1)
        black_mask = np.all(self.img_minimap == [0, 0, 0], axis=-1)
        self.img_minimap[black_mask] = [1, 1, 1]

        # Get player location on minimap (the same detection the first,
        # working recording used: minimap yellow dot lookup).
        loc_player_minimap = get_player_location_on_minimap(self.img_minimap)
        if loc_player_minimap:
            self.loc_player_minimap = loc_player_minimap

        # Update map
        if self.is_first_frame:
            # Reuse a previously saved map.png so a re-recording session
            # aligns with the existing route instead of rebuilding from the
            # current minimap crop (which would misalign coordinates).
            saved_map = None
            if self.map_dir:
                saved_map_path = os.path.join(self.map_dir, "map.png")
                if os.path.exists(saved_map_path):
                    saved_map = load_image(saved_map_path)
            if saved_map is not None:
                self.img_map = saved_map
                logger.info("[routeRecorder] Reused existing map.png "
                            "for alignment; F4 to overwrite it.")
            # copy minimap to map
            if self.img_map is None:
                self.img_map = self.img_minimap.copy()
                pad = self.cfg["route_recoder"]["map_padding"]
                self.img_map = cv2.copyMakeBorder(
                    self.img_map,
                    top=pad, bottom=pad, left=pad, right=pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )

            # Replace player "yellow" dot to black on map
            self.replace_color_on_map(
                (55, 40, 80),
                (60, 100, 100)
            )
            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 80, 80),
                                      (5, 100, 100))

            # Update route: start from the last saved route so re-recording
            # extends/overwrites it (F3 overwrites the file).
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.idx_routes = self._count_existing_routes()
            self.img_route_debug = self.img_route.copy()

        else:
            # Create mask where pixels are not black
            mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(np.uint8)
            mask = mask * 255

            # Perform template matching to find where the current minimap fits in the global map
            self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                self.img_map,
                self.img_minimap,
                mask=mask
            )
            x, y = self.loc_minimap_global
            h, w = self.img_minimap.shape[:2]
            # Ensure img_map is big enough to fit the newly explored region
            self.ensure_img_map_capacity(x, y, h, w)

            # Don't copy pixel near player
            player_yellow_dot_radius = 5
            px, py = self.loc_player_minimap
            h, w = self.img_minimap.shape[:2]
            x_min = max(0, px - player_yellow_dot_radius)
            x_max = min(w, px + player_yellow_dot_radius)
            y_min = max(0, py - player_yellow_dot_radius)
            y_max = min(h, py + player_yellow_dot_radius)
            # Apply the black color mask to mask player yellow dot
            self.img_minimap[y_min:y_max, x_min:x_max] = (0, 0, 0)

            # Update map
            if self.args.map == '':
                map_slice = self.img_map[y:y+h, x:x+w]
                black_mask = np.all(map_slice == [0, 0, 0], axis=2)
                map_slice[black_mask] = self.img_minimap[black_mask]

            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 78, 78),
                                      (5, 100, 100))

        if getattr(self, "show_debug", False):
            cv2.imshow("Map", self.img_map)
        self.img_route_debug = self.img_route.copy()

        # Get player location on global map.
        self.loc_player_global = self.get_player_location_on_global_map()

        # Track the previous smoothed position for walking-detection below
        # (short key taps are missed by the per-frame key snapshot).
        if self.loc_player_global_prev is None:
            self.loc_player_global_prev = self.loc_player_global

        # Determine which color code to use based on user input
        action = ""
        is_draw_blob = False
        key_press = self.kb.key_pressing
        if "space" in key_press:
            if "left" in key_press:
                action = "left none jump"
            elif "right" in key_press:
                action = "right none jump"
            elif "down" in key_press:
                action = "none down jump"
            else:
                action = "none none jump"
            is_draw_blob = True
        elif "e" in key_press: # Teleport skill
            if "left" in key_press:
                action = "left none teleport"
            elif "right" in key_press:
                action = "right none teleport"
            elif "down" in key_press:
                action = "none down teleport"
            elif "up" in key_press:
                action = "none up teleport"
            else:
                action = ""
            is_draw_blob = True
        elif "up" in key_press:
            action = "none up none"
        elif "down" in key_press:
            action = "none down none"
        elif "left" in key_press:
            action = "left none none"
        elif "right" in key_press:
            action = "right none none"
        else:
            action = ""
            # Short walking key taps are often shorter than one recorder
            # frame, so the key snapshot above misses them. Detect walking
            # from the player's displacement on the global map instead. To
            # avoid jitter drawing phantom lines, require a reasonably large
            # move and a direction consistent with the previous frame.
            if self.loc_player_global_prev is not None:
                dx = self.loc_player_global[0] - self.loc_player_global_prev[0]
                dy = self.loc_player_global[1] - self.loc_player_global_prev[1]
                dist = float(np.hypot(dx, dy))
                # Ignore tiny jitter and wild jumps (camera teleports).
                if 8.0 <= dist <= 80.0 and abs(dx) >= abs(dy) * 1.5:
                    action = ("right none none" if dx > 0
                              else "left none none")

        # Remember the current position for the next frame's displacement.
        self.loc_player_global_prev = self.loc_player_global

        # Check if need to change route
        if self.kb.is_pressed_func_key[2]: # 'F3' is pressed
            action = "none none goal"
            is_draw_blob = True
            self.kb.is_pressed_func_key[2] = False
        elif self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Update route image
        if self.is_enable and action != "":
            # Get color from action
            dict_action_to_color = {v: k for k, v in self.color_code.items()}
            color_rgb = dict_action_to_color.get(action, None)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw a line from the last position to the current one (if available)
            px, py = self.loc_player_global
            if is_draw_blob:
                dt = time.time() - self.t_last_draw_blob
                if dt > self.cfg["route_recoder"]["blob_cooldown"]:
                    # Draw a small filled circle at current position
                    cv2.circle(self.img_route,
                            (px, py),
                            radius=2,
                            color=color_bgr,
                            thickness=-1)  # filled circle
                    self.t_last_draw_blob = time.time()
                    self.loc_player_global_last = None
            else:
                if self.loc_player_global_last is None:
                    px_last, py_last = self.loc_player_global
                else:
                    px_last, py_last = self.loc_player_global_last
                cv2.line(self.img_route,
                        (px_last, py_last),
                        (px     , py),
                        color=color_bgr,
                        thickness=1)
                self.loc_player_global_last = self.loc_player_global

        # Save route image if goal is drawn. Single-route design: always
        # overwrite route1.png (the bot follows only the latest recording).
        if action == "none none goal":
            out_path = f"{self.map_dir}/route1.png"
            existed = os.path.exists(out_path)
            _imwrite_cn(out_path, self.img_route)
            self.img_route = self.img_map.copy()
            logger.info(f"Save route image to {out_path}"
                        f" ({'覆盖' if existed else '新增'})")

        # Save img_map to map.png
        if self.kb.is_pressed_func_key[3]: # 'F4' is pressed
            out_path = f"{self.map_dir}/map.png"
            _imwrite_cn(out_path, self.img_map)
            self.kb.is_pressed_func_key[3] = False
            logger.info(f"Save map image to {out_path}")

        #####################
        ### Debug Windows ###
        #####################
        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        # Resize img_route_debug for better visualization
        self.img_route_debug = cv2.resize(
                    self.img_route_debug, (0, 0),
                    fx=self.cfg["minimap"]["debug_window_upscale"],
                    fy=self.cfg["minimap"]["debug_window_upscale"],
                    interpolation=cv2.INTER_NEAREST)
        if getattr(self, "show_debug", False):
            cv2.imshow("Route Map Debug", self.img_route_debug)

        # Enable cached location since second frame
        self.is_first_frame = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument to specify map name
    parser.add_argument(
        '--new_map',
        type=str,
        default='new_map',
        help='Specify the new map name'
    )

    parser.add_argument(
        '--show-debug',
        action='store_true',
        help='Show the debug windows (Game Window Debug / Map / Route Map '
             'Debug). Off by default: the recorder runs headless and logs '
             'progress to the console, which avoids OpenCV window freezes '
             'when launched from a background process.'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--map',
        type=str,
        default='',
        help='use this map instead of creating a new one'
    )

    try:
        routeRecorder = RouteRecorder(parser.parse_args())
    except Exception as e:
        logger.error(f"RouteRecorder Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            routeRecorder.run_once()

            # Headless: exit on 'q' via the keyboard listener's quit key is
            # handled by KeyBoardListener; but keep a console-only friendly
            # progress log every ~2s so the user knows it is recording.
            if not routeRecorder.show_debug:
                routeRecorder._log_tick += 1
                if routeRecorder._log_tick % (routeRecorder.fps_limit * 2) == 0:
                    n = routeRecorder._route_pixel_count()
                    logger.info(
                        f"[routeRecorder] 录制中… 已记录 {n} 个动作点 "
                        f"(F1暂停/F3保存/F9或Ctrl+C退出)")
            else:
                # Exit if 'q' is pressed (only when a window is shown).
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / routeRecorder.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
