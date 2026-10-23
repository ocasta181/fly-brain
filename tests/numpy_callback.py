import numpy as np


class RestoreCallback:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        pass

    def __exit__(self, *exception):
        np.seterrcall(self.callback)
