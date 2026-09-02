# Local import
from src.states.base_state import State

class AuxiliaryState(State):
    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def check_transitions(self):
        return None

    def on_frame(self):
        pass
