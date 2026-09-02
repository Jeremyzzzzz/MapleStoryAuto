'''
Execute this script:
python AutoDiceRoller.py --attribute 4,4,13,4 --cfg XXX
'''
# Standard import
import time
import argparse
import sys

# Library import
import numpy as np
import cv2

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import (
    find_pattern_sqdiff, screenshot, load_image,
    is_mac, override_cfg, load_yaml, click_in_game_window,
)
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardListener import KeyBoardListener

class AutoDiceRoller:
    '''
    AutoDiceRoller
    '''
    def __init__(self, args):
        '''
        Init AutoDiceRoller
        '''
        # self.cfg = Config # Configuration
        self.args = args # User arguments
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = True
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))

        # Set up fps limit
        self.fps_limit = self.cfg["system"]["fps_limit_auto_dice_roller"]

        # Load number image
        self.img_numbers = [
            load_image(f"numbers/{i}.png", cv2.IMREAD_GRAYSCALE)
            for i in range(4, 14)
        ]

        # Start keyboard listener thread
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
            self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        # Make sure resolution is as expected
        if self.cfg["game_window"]["size"] != self.frame.shape[:2]:
            text = (
                f"Unexpeted window size: {self.frame.shape[:2]} "
                f"(expect {self.cfg['game_window']['size']})"
            )
            logger.error(text)
            return

        # Resize raw frame to working size
        self.img_frame = cv2.resize(self.frame, WINDOW_WORKING_SIZE,
                                    interpolation=cv2.INTER_NEAREST)

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Enable cached location since second frame
        self.is_first_frame = False

        # Check if user want to disable dice rolling
        if self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        if self.is_enable and self.kb.is_game_window_active():
            loc_dice = (981, 445)
            loc_first_box = (890, 371)
            box_size = (22, 37) # (h ,w)
            box_y_interval = 25
            window_title = self.cfg["game_window"]["title"]

            # Parse the attribute number
            attibutes_info = []
            for i, attibute in enumerate(["STR", "DEX", "INT", "LUK"]):
                # Calculate the box position
                p0 = (loc_first_box[0], loc_first_box[1] + i * box_y_interval)
                p1 = (p0[0] + box_size[1], p0[1] + box_size[0])

                # Crop the box region from the image
                img_roi = self.img_frame_gray[p0[1]:p1[1], p0[0]:p1[0]]

                # Match with each number template (from 4 to 11)
                best_score = float('inf')
                best_digit = None
                for idx, img_number in enumerate(self.img_numbers, start=4):
                    _, score, _ = find_pattern_sqdiff(img_roi, img_number)
                    if score < best_score:
                        best_score = score
                        best_digit = idx
                logger.info(f"[{attibute}]: {best_digit} (score: {round(best_score, 2)})")
                attibutes_info.append((best_digit, best_score))

                # Draw box and put text on debug image
                cv2.rectangle(self.img_frame_debug, p0, p1, (0, 0, 255), 1)
                cv2.putText(
                    self.img_frame_debug,
                    f"{best_digit}",
                    (p0[0], p0[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # for val, score in attibutes_info:
            #     if score > 0.11:
            #         logger.warning(f"Stop! Unable to recognize number: {(val, score)})")
            #         self.is_enable = False

            # Check if is equal to target
            is_jackpot = True
            for i, (val, score) in enumerate(attibutes_info):
                target = self.args.attribute[i]
                if target is not None and target != val:
                    is_jackpot = False

            # Stop rolling dice if reach target
            if is_jackpot:
                self.is_enable = False
                logger.info("Hit Jackpot! Stop!")

            # Click to roll the dice or not
            if self.is_enable:
                click_in_game_window(window_title, loc_dice)
                logger.info("Roll the dice")

        # Show debug image on window
        self.update_img_frame_debug()

def parse_and_validate_attributes(attr_str):
    raw_values = attr_str.split(',')
    if len(raw_values) != 4:
        raise argparse.ArgumentTypeError("You must provide exactly 4 attributes: STR,DEX,INT,LUK")

    parsed = []
    total_known = 0
    unknown_count = 0

    for v in raw_values:
        if v.strip() == '?':
            parsed.append(None)
            unknown_count += 1
        else:
            try:
                val = int(v)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid attribute value: {v}")
            if not (4 <= val <= 13):
                raise argparse.ArgumentTypeError("Each attribute must be between 4 and 13.")
            parsed.append(val)
            total_known += val

    if unknown_count > 0 and (total_known > 25 or total_known + 4 * unknown_count > 25):
        raise argparse.ArgumentTypeError("Impossible to satisfy sum of 25 with current values.")

    return parsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--attribute',
        type=parse_and_validate_attributes,
        default=[4, 4, 13, 4],
        help='Assign the attributes in order: STR,DEX,INT,LUK.'
             'Each must be between 4-13 and total must sum to 25.'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    try:
        autoDiceRoller = AutoDiceRoller(parser.parse_args())
    except Exception as e:
        logger.error(f"AutoDiceRoller Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            autoDiceRoller.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / autoDiceRoller.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
