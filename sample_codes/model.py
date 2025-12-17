import numpy as np

class Model:
    def fit(self, x):
        x = self._check(x)
        return np.mean(x)

    def _check(self, x):
        return np.array(x)

def main():
    m = Model()
    return m.fit([1, 2, 3])
