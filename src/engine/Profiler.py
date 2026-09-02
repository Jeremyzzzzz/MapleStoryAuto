import time
from collections import defaultdict

# Local improt
from src.utils.logger import logger

class Profiler:
    '''
    Profiler
    '''
    def __init__(self, cfg):
        self.enable = cfg["profiler"]["enable"]
        self.reset()
        self.total_frames = 0
        # Timer
        self.t_start = time.time()
        self.t_last_mask = time.time()

        if self.enable:
            logger.info("Start Profiling...")

    def reset(self):
        '''
        Reset all profiling data
        '''
        self.start_time = time.time()
        self.times = defaultdict(float)       # total time per label
        self.counts = defaultdict(int)        # how many times each label is marked
        self.total_frames = 0
        self.t_start = self.t_last_mask = time.time()

    def start(self):
        '''
        Start timing for a new frame
        '''
        if not self.enable:
            return
        self.t_start = self.t_last_mask = time.time()
        self.total_frames += 1

    def mark(self, label):
        '''
        Mark time elapsed since last mark for a section
        '''
        if not self.enable:
            return
        now = time.time()
        elapsed = now - self.t_last_mask
        self.times[label] += elapsed
        self.counts[label] += 1
        self.t_last_mask = now

    def report(self):
        '''
        Report average time per section over all frames
        '''
        if not self.enable or self.total_frames == 0:
            return ""

        total_time = sum(self.times.values())
        report_lines = []

        for label, total_label_time in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            avg_time = total_label_time / self.total_frames
            percent = (total_label_time / total_time) * 100 if total_time > 0 else 0
            report_lines.append(f"{label:<20}: {avg_time:.4f}s avg ({percent:.1f}%)")

        avg_frame_time = total_time / self.total_frames
        total_duration = time.time() - self.start_time
        avg_fps = self.total_frames / total_duration if total_duration > 0 else 0

        report_lines.append(f"{'AVG FRAME TIME':<20}: {avg_frame_time:.4f}s over {self.total_frames} frames")
        report_lines.append(f"{'AVG FPS':<20}: {avg_fps:.2f}")

        return "\n".join(report_lines)
