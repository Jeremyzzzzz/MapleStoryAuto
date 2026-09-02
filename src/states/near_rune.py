import time

# Local import
from src.states.base_state import State
from src.utils.logger import logger
from src.input.KeyBoardController import press_key

class NearRuneState(State):
    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.bot = bot

    def on_enter(self):
        pass

    def on_exit(self):
        self.bot.rune_solver.reset()

    def check_transitions(self):
        if self.bot.rune_solver.is_in_rune_game(
            self.bot.img_frame, self.bot.img_frame_debug):
            # Check whether in arrow box mini game
            return "solving_rune"

        elif time.time() - self.bot.fsm.t_last_transition > \
            self.bot.cfg["rune_find"]["near_rune_duration"]:
            # Check if statue timeout
            return "finding_rune"

        else:
            return None

    def on_frame(self):
        # Update rune location on screen
        self.bot.rune_solver.update_rune_location(
                self.bot.img_frame,
                self.bot.img_frame_debug,
                self.bot.loc_player
        )

        dx = abs(self.bot.loc_player[0] - self.bot.rune_solver.loc_rune[0])
        dy = abs(self.bot.loc_player[1] - self.bot.rune_solver.loc_rune[1])
        logger.info(f"[NearRuneState] Player distance to rune: ({dx}, {dy})")

        # Check if close enough to trigger the rune
        if  dx < self.bot.cfg["rune_find"]["rune_trigger_distance_x"] and \
            dy < self.bot.cfg["rune_find"]["rune_trigger_distance_y"]:
            press_key("up", 0.02) # Attempt to trigger rune

        # Get commend from route map
        self.bot.update_cmd_by_route()

        # Check if reach goal on route map
        self.bot.check_reach_goal()

        # send command to keyboard controller
        self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                self.bot.cmd_move_y + ' ' + \
                                self.bot.cmd_action)
