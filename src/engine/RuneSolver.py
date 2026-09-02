'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger
from src.utils.common import (find_pattern_sqdiff, draw_rectangle, screenshot,
    load_image, get_mask, nms_matches, to_opencv_hsv
)
from src.input.KeyBoardController import press_key

class RuneSolver:
    '''
    Init RuneSolver
    '''
    def __init__(self, cfg):
        self.cfg = cfg # Configuration
        # Image
        self.img_rune_warning = None
        self.img_runes = []
        self.img_arrows = {
            "left":
                [load_image("rune/arrow_left_1.png"),
                 load_image("rune/arrow_left_2.png"),
                 load_image("rune/arrow_left_3.png"),],
            "right":
                [load_image("rune/arrow_right_1.png"),
                 load_image("rune/arrow_right_2.png"),
                 load_image("rune/arrow_right_3.png"),],
            "up":
                [load_image("rune/arrow_up_1.png"),
                 load_image("rune/arrow_up_2.png"),
                 load_image("rune/arrow_up_3.png")],
            "down":
                [load_image("rune/arrow_down_1.png"),
                 load_image("rune/arrow_down_2.png"),
                 load_image("rune/arrow_down_3.png"),],
        }
        # Load rune images from rune/
        lang = cfg["system"]["language"]
        self.img_rune_warning = load_image(f"rune/rune_warning_{lang}.png",
                                           cv2.IMREAD_GRAYSCALE)
        self.img_rune_warning_mask = get_mask(load_image(f"rune/rune_warning_{lang}.png"), (0, 255, 0))

        self.img_runes = [load_image( "rune/rune_1.png"),
                          load_image(f"rune/rune_2_{lang}.png"),
                          load_image( "rune/rune_3.png"),]
        self.img_rune_enable = load_image(f"rune/rune_enable_{lang}.png",
                                          cv2.IMREAD_GRAYSCALE)
        # Coordinate
        self.loc_rune = None # rune location on game screen

    def reset(self):
        self.loc_rune = None

    def solve_rune(self, img, img_debug):
        '''
        Automatically solves the rune puzzle mini-game by recognizing directional arrows
        on the screen and simulating the correct key presses.

        This function:
        - Continuously checks whether the rune game is active.
        - For each of the 4 arrows:
            - Compares the cropped image with all known arrow templates.
            - Selects the best-matching arrow direction.
            - Simulates a key press corresponding to the detected arrow direction.
        - Repeats until the rune game ends.

        Note:
            - Skips key press simulation if `--disable_control` flag is set.

        Returns:
            None
        '''
        # Only the highlighted arrow will show on mask
        img_bin = self.arrow_hsv_binarized(img,
                                           self.cfg['rune_solver']['arrow_highlight_low_hsv'],
                                           self.cfg['rune_solver']['arrow_highlight_high_hsv'])
        # Debug img_bin
        # cv2.imshow("img_bin", img_bin)
        # cv2.waitKey(1)

        for arrow_idx in [0,1,2,3]:
            # Crop arrow detection box
            x0, y0 = self.cfg["rune_solver"]["arrow_box_coord"]
            x = x0 + self.cfg["rune_solver"]["arrow_box_interval"] * arrow_idx
            y = y0
            size = self.cfg["rune_solver"]["arrow_box_size"]

            # Detect circles
            img_roi = img_bin[y:y+size, x:x+size].astype(np.uint8)
            circles = cv2.HoughCircles(
                img_roi,
                method=cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=10,  # Minimum distance between centers
                param1=100,   # Edge detector high threshold
                param2=15,   # Circle detection threshold (smaller = more sensitive)
                minRadius=30,
                maxRadius=35,
            )

            if circles is not None: # Get the highlighted arrow
                circles = np.around(circles[0]).astype(int)  # Flatten and round
                # Circle Debug
                for (cx, cy, r) in circles:
                    # Offset back to original coordinates on full image
                    cv2.circle(img_debug, (x + cx, y + cy), r, (0, 0, 255), 2)

                # Loop through all possible arrows template and choose the most possible one
                best_score = float('inf')
                best_direction = ""
                for direction, arrow_list in self.img_arrows.items():
                    for img_arrow in arrow_list:
                        _, score, _ = find_pattern_sqdiff(
                                        img[y:y+size, x:x+size], img_arrow,
                                        mask=get_mask(img_arrow, (0, 255, 0)))
                        if score < best_score:
                            best_score = score
                            best_direction = direction
                logger.info(f"[RuneSolver] Highlighted arrow_{arrow_idx} is pointing {best_direction}"\
                            f"(score={round(best_score, 2)})")

                # Update img_frame_debug
                draw_rectangle(
                    img_debug, (x, y), (size, size),
                    (0, 0, 255), str(round(best_score, 2))
                )

                # For logging
                screenshot(img_debug, "solve_rune")

                # Press the key for 0.5 second
                press_key(best_direction, 0.5)

                time.sleep(0.5) # wait for the game to react

                return # Solve one arrow at a time

        logger.info("[RuneSolver] No arrows is highlighted, skip frame")

    def is_rune_enable(self, img, img_debug):
        '''
        Checks whether the rune enable message is appear on the game frame.

        This function:
        - Crops a specific region where the rune warning icon is expected.
        - Compare the cropped region against the known rune warning image template.
        - Returns True if rune warning template matched

        Returns:
            bool: True if the rune warning is detected, False otherwise.
        '''
        lang = self.cfg['system']['language']
        x0, y0 = self.cfg[f'rune_enable_msg_{lang}']["top_left"]
        x1, y1 = self.cfg[f'rune_enable_msg_{lang}']["bottom_right"]

        # Find the rune enable message
        _, score, _ = find_pattern_sqdiff(
                        img[y0:y1, x0:x1],
                        self.img_rune_enable)
        # Debug
        if self.cfg['rune_detect']['debug']:
            draw_rectangle(
                img_debug, (x0, y0), (y1-y0, x1-x0),
                (0, 0, 255), f"Rune Enable Msg({round(score, 2)})")

        if score < self.cfg[f'rune_enable_msg_{lang}']["diff_thres"]:
            logger.info(f"[RuneSolver] Detect rune enable message on screen with score({round(score, 2)})")
            return True
        else:
            return False

    def is_rune_warning(self, img, img_debug):
        '''
        Checks whether the rune warning icon is appear on the game frame.

        This function:
        - Crops a specific region where the rune warning icon is expected.
        - Compare the cropped region against the known rune warning image template.
        - Returns True if rune warning template matched

        Returns:
            bool: True if the rune warning is detected, False otherwise.
        '''
        lang = self.cfg['system']['language']
        x0, y0 = self.cfg[f'rune_warning_{lang}']["top_left"]
        x1, y1 = self.cfg[f'rune_warning_{lang}']["bottom_right"]

        # Detect rune warning
        _, score, _ = find_pattern_sqdiff(
                        img[y0:y1, x0:x1], self.img_rune_warning,
                        mask=self.img_rune_warning_mask)

        # Debug
        if self.cfg['rune_detect']['debug']:
            draw_rectangle(
                img_debug, (x0, y0), (y1-y0, x1-x0),
                (0, 0, 255), f"Rune Warning({round(score, 2)})")

        if score < self.cfg[f'rune_warning_{lang}']["diff_thres"]:
            logger.info(f"[RuneSolver] Detect rune warning on screen with score({score})")
            return True
        else:
            return False

    def update_rune_location(self, img, img_debug, loc_player):
        '''
        Checks if a rune icon is visible around the player's position.

        This function:
        - Uses template matching to detect the rune icon within this predefine box.

        Returns:
            nearest rune
        '''
        # Calculate bounding box
        h, w = img.shape[:2]
        h_rune_box = self.cfg["rune_detect"]["box_height"]
        w_rune_box = self.cfg["rune_detect"]["box_width"]
        x0 = max(0, loc_player[0] - w_rune_box // 2)
        y0 = max(0, loc_player[1] - h_rune_box)
        x1 = min(w, loc_player[0] + w_rune_box // 2)
        y1 = min(h, loc_player[1])

        # Debug
        draw_rectangle(
            img_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Rune Detection Range"
        )

        # Make sure ROI is large enough to hold a full rune
        max_rune_height = max(r.shape[0] for r in self.img_runes)
        max_rune_width  = max(r.shape[1] for r in self.img_runes)
        if (x1 - x0) < max_rune_width or (y1 - y0) < max_rune_height:
            return  # Skip check if box is out of range

        # Match each rune part separately
        matches = []
        for i, img_rune in enumerate(self.img_runes):
            mask = get_mask(img_rune, (0, 255, 0))
            loc, score, _ = find_pattern_sqdiff(img[y0:y1, x0:x1], img_rune, mask=mask)
            matches.append((i, loc, score, img_rune.shape))

        # # Matches box debug
        # for i, (part_idx, loc, score, shape) in enumerate(matches):
        #     draw_rectangle(
        #         self.img_frame_debug,
        #         (x0 + loc[0], y0 + loc[1]),
        #         shape,
        #         (255, 255, 0),
        #         f"{i},{round(score, 2)}",
        #         thickness=1,
        #         text_height=0.4
        #     )

        # Filter out good matches
        good_matches = [
            (i, loc, score, shape)
            for (i, loc, score, shape) in matches
            if score < self.cfg["rune_detect"]["diff_thres"]
        ]

        # Remove overlapping matches
        good_matches = nms_matches(good_matches)

        # Require at least 2 rune parts
        if len(good_matches) < 2:
            return

        # Horizontal: max distance between part's centers are small
        x_centers = [x0 + loc[0] + shape[1] // 2 for (_, loc, _, shape) in good_matches]
        if max(x_centers) - min(x_centers) > 10:
            return

        # Vertical: check if all Y's are strictly increasing
        ys = [y0 + loc[1] for (_, loc, _, _) in good_matches]
        if not all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)):
            return

        logger.info(f"[RuneSolver] Found rune parts near player with scores:"
                    f" {[round(s, 2) for (_, _, s, _) in matches]}")

        # Update rune location
        self.loc_rune = (int(sum(x_centers) / len(x_centers)),
                         int(sum(ys) / len(ys)))

        # Draw all parts on debug window
        for (i, loc, score, shape) in matches:
            draw_rectangle(
                img_debug,
                (x0 + loc[0], y0 + loc[1]),
                shape,
                (255, 0, 255),
                f"{i},{round(score, 2)}",
                text_height=0.5,
                thickness=1
            )

        # Draw rune location on debug window
        cv2.circle(img_debug, self.loc_rune,
                   radius=5, color=(0, 255, 255), thickness=-1)

        screenshot(img_debug, "rune_detected")

    def arrow_hsv_binarized(self, img, low_hsv, high_hsv):
        """
        Convert a BGR image to a binary mask using HSV thresholding.
        Handles hue wraparound (e.g., low_hsv > high_hsv).

        Args:
            img (np.ndarray): BGR image
            low_hsv (list/tuple): Lower HSV bound in standard format (0–360, 0–100, 0–100)
            high_hsv (list/tuple): Upper HSV bound in standard format (0–360, 0–100, 0–100)

        Returns:
            np.ndarray: Binary mask (0 or 255)
        """
        # Convert image to HSV (OpenCV format)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Check if hue range wraps around
        if low_hsv[0] > high_hsv[0]:
            # Wraparound: split into two ranges
            lower1 = to_opencv_hsv([0          , low_hsv[1] , low_hsv[2]])
            upper1 = to_opencv_hsv([high_hsv[0], high_hsv[1], high_hsv[2]])
            mask1 = cv2.inRange(img_hsv, lower1, upper1)

            lower2 = to_opencv_hsv([low_hsv[0] , low_hsv[1] , low_hsv[2]])
            upper2 = to_opencv_hsv([360        , high_hsv[1], high_hsv[2]])
            mask2 = cv2.inRange(img_hsv, lower2, upper2)

            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Normal range
            lower = to_opencv_hsv(low_hsv)
            upper = to_opencv_hsv(high_hsv)
            mask = cv2.inRange(img_hsv, lower, upper)

        return mask

    def is_in_rune_game(self, img, img_debug):
        '''
        Determines whether the rune puzzle game screen is currently active.

        This function:
        - Extracts the RoI where the first arrow of the rune puzzle appears.
        - Performs masked template matching with all known arrow templates.
        - Computes the best matching score across all arrow directions.
        - If the best match score is below a threshold,
          the rune puzzle is considered active.

        Returns:
            bool: True if the rune game is detected on screen, False otherwise.
        '''
        img_bin = self.arrow_hsv_binarized(img,
                                           self.cfg['rune_solver']['arrow_low_hsv'],
                                           self.cfg['rune_solver']['arrow_high_hsv'])
        # Debug img_bin
        # cv2.imshow("img_bin", img_bin)
        # cv2.waitKey(1)

        num_circles = 0
        for arrow_idx in [0,1,2,3]:
            # Crop arrow detection box
            x0, y0 = self.cfg["rune_solver"]["arrow_box_coord"]
            x = x0 + self.cfg["rune_solver"]["arrow_box_interval"] * arrow_idx
            y = y0
            size = self.cfg["rune_solver"]["arrow_box_size"]

            img_roi = img_bin[y:y+size, x:x+size].astype(np.uint8)
            # Detect circles
            circles = cv2.HoughCircles(
                img_roi,
                method=cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=10,  # Minimum distance between centers
                param1=100,   # Edge detector high threshold
                param2=15,   # Circle detection threshold (smaller = more sensitive)
                minRadius=30,
                maxRadius=35,
            )

            # Circle Debug
            if circles is not None:
                num_circles += 1
                circles = np.around(circles[0]).astype(int)  # Flatten and round
                for (cx, cy, r) in circles:
                    # Offset back to original coordinates on full image
                    cv2.circle(img_debug, (x + cx, y + cy), r, (0, 255, 0), 1)

        return num_circles >= 3


    def is_in_rune_game_legacy(self, img, img_debug):
        '''
        Determines whether the rune puzzle game screen is currently active.

        This function:
        - Extracts the RoI where the first arrow of the rune puzzle appears.
        - Performs masked template matching with all known arrow templates.
        - Computes the best matching score across all arrow directions.
        - If the best match score is below a threshold,
          the rune puzzle is considered active.

        Returns:
            bool: True if the rune game is detected on screen, False otherwise.
        '''
        # Crop arrow detection box
        x, y = self.cfg["rune_solver"]["arrow_box_coord"]
        size = self.cfg["rune_solver"]["arrow_box_size"]

        # Check if arrow appear on screen
        best_score = float('inf')
        for _, arrow_list in self.img_arrows.items():
            for img_arrow in arrow_list:
                _, score, _ = find_pattern_sqdiff(
                                img[y:y+size, x:x+size], img_arrow,
                                mask=get_mask(img_arrow, (0, 255, 0)))
                if score < best_score:
                    best_score = score

        draw_rectangle(
            img_debug, (x, y), (size, size),
            (0, 0, 255), str(round(best_score, 2))
        )

        if best_score < self.cfg["rune_solver"]["arrow_box_diff_thres"]:
            logger.info(f"[RuneSolver] Arrow detected with score({score})")
            return True
        else:
            return False
