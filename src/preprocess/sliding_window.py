import numpy as np


class SlidingWindowGenerator:
    def __init__(self, window_size=24, stride=2):
        self.window_size = window_size
        self.stride = stride

    def generate(self, sequence):
        """
        sequence: (T, D)
        return: (num_windows, window_size, D)
        """
        T, D = sequence.shape
        windows = []

        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(sequence[start:end])

        if not windows:
            return np.empty((0, self.window_size, D))

        return np.array(windows)