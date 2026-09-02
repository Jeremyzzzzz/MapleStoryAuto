'''
KeyBoardListener for routeRecorder.py
'''
# Standard Import
import threading
import time

import pygetwindow as gw
from pynput import keyboard

# Local import
from src.utils.logger import logger

class KeyBoardListener():
    '''
    KeyBoardListener
    '''
    def __init__(self, cfg=None, is_autobot=True):
        self.cfg = cfg
        self.t_last_run = time.time()
        self.is_enable = True
        self.debounce_interval = 1 # second
        self.is_terminated = False
        self.fps = 0
        self.fps_limit = 30
        self.key_pressing = [] # record the key pressed by user
        self.is_pressed_func_key = [False]*12  # 'F1', 'F2', .... 'F12'

        # Timer
        self.t_func_key = [0]*12 # 'F1', 'F2', .... 'F12'

        # Keys
        self.movement_keys = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space"
        }
        self.func_keys = {
            getattr(keyboard.Key, f"f{i+1}"): i for i in range(12)
        }

        self.func_key_handlers = {
            f"f{i}": self.do_nothing for i in range(1, 13)
        }

        # Start keyboard control thread
        if is_autobot:
            threading.Thread(target=self.run_for_autobot, daemon=True).start()
        else:
            self.cfg = cfg
            self.window_title = cfg["game_window"]["title"]
            threading.Thread(target=self.run_for_route_recorder, daemon=True).start()

        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()

    def do_nothing(self):
        pass

    def register_func_key_handler(self, key: str, handler: callable):
        key = key.lower()
        if key in self.func_key_handlers:
            self.func_key_handlers[key] = handler
        else:
            logger.warning(f"[KeyBoardListener] '{key}' is not a supported function key.")

    def on_release(self, key):
        '''
        Handle key release events and update key_pressing list.
        '''
        try:
            # Regular keys (like 'a', '1', etc.)
            k = key.char.lower()
        except AttributeError:
            k = self.movement_keys.get(key, None)

        # Remove the key from key_pressing list if it's in there
        if k in self.key_pressing:
            self.key_pressing.remove(k)

    def on_press(self, key):
        '''
        Handle key press events.
        '''
        try:
            # Regular character keys (e.g., 'a', 'w', '1')
            k = key.char.lower()
        except AttributeError:
            # Handle F1, F2, F3, ... F12
            if key in self.func_keys:
                idx = self.func_keys[key]
                if time.time() - self.t_func_key[idx] > self.debounce_interval:
                    self.is_pressed_func_key[idx] = True # Polling
                    self.func_key_handlers.get(key.name.lower())()
                    self.t_func_key[idx] = time.time()

            k = self.movement_keys.get(key, None)

        if k and k not in self.key_pressing:
            self.key_pressing.append(k)

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        # self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

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

    def stop(self):
        '''
        Stop keyboard listener thread
        '''
        self.is_terminated = True

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
        active_window = gw.getActiveWindow()
        return active_window is not None and self.window_title in active_window.title

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

    def run_for_route_recorder(self):
        '''
        run
        '''
        while not self.is_terminated:
            # Check if game window is active
            if not self.is_enable or not self.is_game_window_active():
                self.limit_fps()
                continue

            self.limit_fps()

    def run_for_autobot(self):
        '''
        run
        '''
        while not self.is_terminated:
            self.limit_fps()

        logger.info("[KeyBoardListener] Terminated")
