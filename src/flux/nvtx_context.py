# In utils.py
from contextlib import contextmanager
import nvtx
import random
class _NVTXContextClass:
    enabled = False
    # Predefined NVTX colors
    COLORS = [
        "blue",    # 255
        "green",   # 65280
        "red",     # 16711680
        "yellow",  # 16776960
        "orange",  # 16753920
        "purple",  # 8388736
        "cyan",    # 65535
        "magenta", # 16711935
    ]
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @contextmanager
    def range(self, name, color=None):
        if self.enabled:
            if color is None:
                color = random.choice(self.COLORS)
            rng = nvtx.start_range(name, color=color)
            try:
                yield
            finally:
                nvtx.end_range(rng)
        else:
            yield

# Create global instance
NVTXContext = _NVTXContextClass()