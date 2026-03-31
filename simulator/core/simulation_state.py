import numpy as np

class SimulationState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def reset(self, rng: np.random.Generator):
        pass

    def step(self):
        pass
    
    

