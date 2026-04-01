from simulator.core.transforms import AngleTransformableBuffer


class SimulationState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.joints = AngleTransformableBuffer(
            size=(20, 8),
            scale=100.0
        )

    
