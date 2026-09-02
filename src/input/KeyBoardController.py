'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import threading
import time

# Library import
import pyautogui
from pynput import keyboard

# Local import
from src.utils.logger import logger
from src.utils.common import is_mac

if is_mac():
    import Quartz
else:
    import ctypes
    from ctypes import wintypes

    import pygetwindow as gw

    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    MAPVK_VK_TO_VSC = 0

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
    _user32.VkKeyScanW.restype = ctypes.c_short
    _user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
    _user32.MapVirtualKeyW.restype = wintypes.UINT
    _user32.keybd_event.argtypes = [
        wintypes.BYTE,
        wintypes.BYTE,
        wintypes.DWORD,
        wintypes.WPARAM,
    ]
    _user32.keybd_event.restype = None

    _WINDOWS_NAMED_KEYS = {
        "backspace": (0x08, 0x0E, False),
        "tab": (0x09, 0x0F, False),
        "enter": (0x0D, 0x1C, False),
        "shift": (0x10, 0x2A, False),
        "ctrl": (0x11, 0x1D, False),
        "alt": (0x12, 0x38, False),
        "esc": (0x1B, 0x01, False),
        "escape": (0x1B, 0x01, False),
        "space": (0x20, 0x39, False),
        "pageup": (0x21, 0x49, True),
        "pagedown": (0x22, 0x51, True),
        "end": (0x23, 0x4F, True),
        "home": (0x24, 0x47, True),
        "left": (0x25, 0x4B, True),
        "up": (0x26, 0x48, True),
        "right": (0x27, 0x4D, True),
        "down": (0x28, 0x50, True),
        "insert": (0x2D, 0x52, True),
        "delete": (0x2E, 0x53, True),
        "f1": (0x70, 0x3B, False),
        "f2": (0x71, 0x3C, False),
        "f3": (0x72, 0x3D, False),
        "f4": (0x73, 0x3E, False),
        "f5": (0x74, 0x3F, False),
        "f6": (0x75, 0x40, False),
        "f7": (0x76, 0x41, False),
        "f8": (0x77, 0x42, False),
        "f9": (0x78, 0x43, False),
        "f10": (0x79, 0x44, False),
        "f11": (0x7A, 0x57, False),
        "f12": (0x7B, 0x58, False),
    }

    _WINDOWS_MODIFIERS = {
        0x01: _WINDOWS_NAMED_KEYS["shift"],
        0x02: _WINDOWS_NAMED_KEYS["ctrl"],
        0x04: _WINDOWS_NAMED_KEYS["alt"],
    }

pyautogui.PAUSE = 0  # remove delay

def _resolve_windows_key(key):
    key_name = str(key).lower()
    if key_name in _WINDOWS_NAMED_KEYS:
        vk_code, scan_code, extended = _WINDOWS_NAMED_KEYS[key_name]
        return vk_code, scan_code, extended, 0

    if len(key_name) != 1:
        raise ValueError(f"Unsupported Windows key: {key}")

    key_data = _user32.VkKeyScanW(key_name)
    if key_data == -1:
        raise ValueError(f"Unable to map Windows key: {key}")

    vk_code = key_data & 0xFF
    modifiers = (key_data >> 8) & 0xFF
    scan_code = _user32.MapVirtualKeyW(vk_code, MAPVK_VK_TO_VSC) & 0xFF
    return vk_code, scan_code, False, modifiers

def _send_windows_key_event(vk_code, scan_code, extended=False, key_up=False):
    flags = KEYEVENTF_EXTENDEDKEY if extended else 0
    if key_up:
        flags |= KEYEVENTF_KEYUP
    _user32.keybd_event(vk_code, scan_code, flags, 0)

def _emit_windows_key(key, key_up=False):
    vk_code, scan_code, extended, modifiers = _resolve_windows_key(key)

    if not key_up:
        for modifier_flag, modifier_key in _WINDOWS_MODIFIERS.items():
            if modifiers & modifier_flag:
                _send_windows_key_event(*modifier_key)
        _send_windows_key_event(vk_code, scan_code, extended)
        return

    _send_windows_key_event(vk_code, scan_code, extended, key_up=True)
    for modifier_flag, modifier_key in reversed(_WINDOWS_MODIFIERS.items()):
        if modifiers & modifier_flag:
            _send_windows_key_event(*modifier_key, key_up=True)

def key_down(key):
    '''
    Press key down
    '''
    if not key:
        return

    if not is_mac():
        _emit_windows_key(key)
        return

    try:
        pyautogui.keyDown(key)
    except pyautogui.FailSafeException:
        logger.warning("[key_down] pyautogui failsafe triggered during key_down.")
        recover_mouse()

def key_up(key):
    '''
    Release key
    '''
    if not key:
        return

    if not is_mac():
        _emit_windows_key(key, key_up=True)
        return

    try:
        pyautogui.keyUp(key)
    except pyautogui.FailSafeException:
        logger.warning("[key_up] pyautogui failsafe triggered during key_up.")
        recover_mouse()

def recover_mouse():
    '''
    Move mouse back to center to avoid pyautogui failsafe
    '''
    pyautogui.FAILSAFE = False # Temp disasble failsafe to avoid nested exception

    screen_w, screen_h = pyautogui.size()
    pyautogui.moveTo(screen_w // 2, screen_h // 2)
    time.sleep(0.2) # Give it a moment to "cool down"

    pyautogui.FAILSAFE = True # Recover failsafe

def press_key(key, duration=0.05):
    '''
    Simulates a key press for a specified duration
    '''
    if key:
        key_down(key)
        time.sleep(duration)
        key_up(key)

class KeyBoardController():
    '''
    KeyBoardController
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""
        self.window_title = cfg["game_window"]["title"]
        self.fps = 0 # Frame per seconds
        # Timer
        self.t_last_up = 0.0
        self.t_last_down = 0.0
        self.t_last_toggle = 0.0
        self.t_last_screenshot = 0.0
        self.t_last_jump_down = 0.0
        self.t_last_run = time.time()
        self.t_last_skill = 0.0 # Last time character perform action(attack, cast spell, ...)
        self.t_last_buff_cast = [0] * len(self.cfg["buff_skill"]["keys"]) # Last time cast buff skill
        # Flags
        self.is_enable = True
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = self.cfg["system"]["key_debounce_interval"]
        self.fps_limit = self.cfg["system"]["fps_limit_keyboard_controller"]

        # use 'ctrl', 'alt' for mac, because it's hard to get around
        # macOS's security settings
        if is_mac():
            self.toggle_key = keyboard.Key.ctrl
            self.screenshot_key = keyboard.Key.alt
            self.terminate_key = keyboard.Key.esc
        else:
            self.toggle_key = keyboard.Key.f1
            self.screenshot_key = keyboard.Key.f2
            self.terminate_key = keyboard.Key.f12

        # set up attack key
        self.attack_key = ""
        if cfg["bot"]["attack"] == "aoe_skill":
            self.attack_key = cfg["key"]["aoe_skill"]
        elif cfg["bot"]["attack"] == "directional":
            self.attack_key = cfg["key"]["directional_attack"]
        else:
            raise ValueError(f"Unexpected attack type: {cfg['bot']['attack']}")

        # Start keyboard control thread
        threading.Thread(target=self.run, daemon=True).start()

        logger.info("[KeyBoardController] Init done")

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def disable(self):
        '''
        disable keyboard controlller
        '''
        self.is_enable = False

    def enable(self):
        '''
        enable keyboard controlller
        '''
        self.is_enable = True

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        self.cmd_left_right, self.cmd_up_down, self.cmd_action = new_command.split()

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
        if is_mac():
            active_window = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            for window in active_window:
                window_name = window.get(Quartz.kCGWindowName, '')
                if window_name and self.window_title in window_name:
                    return True
            return False
        else:
            try:
                active_window = gw.getActiveWindow()
                if not active_window:
                    return False
                return self.window_title in active_window.title
            except Exception as e:
                return False

    def release_all_key(self):
        '''
        Release all key
        '''
        key_up("left")
        key_up("right")
        key_up("up")
        key_up("down")
        # Also release attack keys to stop any ongoing attacks
        key_up(self.attack_key)

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

    def run(self):
        '''
        run
        '''
        while not self.is_terminated:
            # Check if game window is active
            if not self.is_enable or not self.is_game_window_active():
                self.limit_fps()
                continue

            # Buff skill
            for i, buff_skill_key in enumerate(self.cfg["buff_skill"]["keys"]):
                cooldown = self.cfg["buff_skill"]["cooldown"][i]
                if time.time() - self.t_last_buff_cast[i] >= cooldown and \
                    time.time() - self.t_last_skill > self.cfg["buff_skill"]["action_cooldown"]:
                    press_key(buff_skill_key)
                    logger.info(f"[Buff] Press buff skill key: '{buff_skill_key}' (cooldown: {cooldown}s)")
                    # Reset timers
                    self.t_last_buff_cast[i] = time.time()
                    self.t_last_skill = time.time()
                    break

            # Force Heal
            if self.is_need_force_heal:
                self.cmd_action = "add_hp"

            ##########################
            ### Left-Right Command ###
            ##########################
            if self.cmd_left_right == "left":
                key_up("right")
                key_down("left")
            elif self.cmd_left_right == "right":
                key_up("left")
                key_down("right")
            elif self.cmd_left_right == "stop":
                key_up("left")
                key_up("right")
            elif self.cmd_left_right == "none":
                if self.cmd_left_right_last != "none":
                    key_up("left")
                    key_up("right")
            else:
                logger.error("[KeyBoardController] Unsupported left-right command: "
                             f"{self.cmd_left_right}")
            self.cmd_left_right_last = self.cmd_left_right

            #######################
            ### Up-Down Command ###
            #######################
            if self.cmd_up_down == "up":
                key_up("down")
                key_down("up")
            elif self.cmd_up_down == "down":
                key_up("up")
                key_down("down")
            elif self.cmd_up_down == "stop":
                key_up("up")
                key_up("down")
            elif self.cmd_up_down == "none":
                if self.cmd_up_down_last != "none":
                    key_up("up")
                    key_up("down")
            else:
                logger.error("[KeyBoardController] Unsupported up-down command: "
                             f"{self.cmd_up_down}")
            self.cmd_up_down_last = self.cmd_up_down

            ######################
            ### Action Command ###
            ######################
            if self.cmd_action == "jump":
                press_key(self.cfg["key"]["jump"])
            elif self.cmd_action == "teleport":
                press_key(self.cfg["key"]["teleport"])
            elif self.cmd_action == "attack":
                press_key(self.attack_key)
                self.t_last_skill = time.time()
            elif self.cmd_action == "add_hp":
                press_key(self.cfg["key"]["add_hp"])
                self.cmd_action = "none"  # Reset command
            elif self.cmd_action == "add_mp":
                press_key(self.cfg["key"]["add_mp"])
                self.cmd_action = "none"  # Reset command
            elif self.cmd_action == "goal":
                pass
            elif self.cmd_action == "none":
                pass
            else:
                logger.error("[KeyBoardController] Unsupported action command: "
                             f"{self.cmd_action}")

            self.limit_fps()

        self.release_all_key() # Prevent key keep press down after termination

        logger.info("[KeyBoardController] terminated")
