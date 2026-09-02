# Standard Import
from argparse import Namespace
import sys

# Pyside
from PySide6.QtCore import Signal, QObject, QTimer

#  Local Import
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.logger import logger
from src.utils.common import load_yaml
from src.input.KeyBoardListener import KeyBoardListener

class AutoBotController(QObject):
    '''
    AutoBot Controller server as a middleman between engine and UI
    '''
    debug_image_signal = Signal(object)
    route_map_viz_signal = Signal(object)

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.ui = None

        # Init Auto Bot
        try:
            # Fake args to pass to AutoBot
            args = Namespace(
                disable_control=False,
                cfg="default",
                debug=False,
                record=False,
                is_ui=True,
                disable_viz=True,
                test_image='',
                init_state='',
            )
            self.auto_bot = MapleStoryAutoBot(args)
        except Exception as e:
            logger.error(f"MapleStoryAutoBot Init Failed: {e}")
            sys.exit(1)
        else:
            logger.info("MapleStoryAutoBot Init Successfully")

        # Update signal for debug window viz
        self.auto_bot.update_signals(self.debug_image_signal,
                                     self.route_map_viz_signal)

        # Monitor function keys
        self.kb_listener = KeyBoardListener(is_autobot=True)

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def update_signal(self, ui):
        '''
        Only called after UI init
        '''
        self.debug_image_signal.connect(ui.update_debug_canvas)
        self.route_map_viz_signal.connect(ui.update_route_map_canvas)
        # Register Function Key handler
        self.kb_listener.register_func_key_handler('f1', ui.button_start_pause.click)
        self.kb_listener.register_func_key_handler('f2', ui.button_screenshot.click)
        self.kb_listener.register_func_key_handler('f3', ui.button_record.click)
        self.kb_listener.register_func_key_handler('f12', lambda: ui.request_close.emit())

    def start_bot(self, cfg_path):
        '''
        Start the bot engine threads
        '''
        # Get config from ui
        cfg = load_yaml(cfg_path)

        # Auto bot load config
        if self.auto_bot.load_config(cfg) != 0:
            return -1 # Load fail

        # Start the bot engine
        try:
            self.auto_bot.start()
        except Exception as e:
            logger.error(f"[start_bot] {e}")
            return -1 # Start fail

        return 0 # start bot success

    def pause_bot(self):
        '''
        Gracefully pause in the engine
        '''
        self.auto_bot.pause()

    def take_screenshot(self):
        '''
        Called when user press screenshot button
        '''
        self.auto_bot.screenshot_img_frame()

    def start_recording(self):
        '''
        Called when user press start record button
        '''
        self.auto_bot.start_record()

    def stop_recording(self):
        '''
        Called when user press stop record button
        '''
        self.auto_bot.stop_record()

    def terminate_bot(self):
        '''
        Called when user stop bot or close UI
        '''
        # Terminate all bot threads
        self.auto_bot.terminate_threads()

    def enable_bot_viz(self):
        '''
        Called when user switch to viz tab
        '''
        self.auto_bot.enable_viz()

    def disable_bot_viz(self):
        '''
        Called when user switch from viz tab
        '''
        self.auto_bot.disable_viz()
