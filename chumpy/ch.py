import numpy as np

class Ch(np.ndarray):
    def __new__(cls, input_array=None, *args, **kwargs):
        if input_array is None:
            input_array = np.array([0.0])
        return np.asanyarray(input_array).view(cls)
    
    def __init__(self, *args, **kwargs):
        pass

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            # Fallback for standard numpy state
            try:
                super().__setstate__(state)
            except:
                pass

    @property
    def v(self):
        return np.array(self)

def depends_on(*args, **kwargs):
    return lambda x: x

class ndarray(Ch):
    pass
