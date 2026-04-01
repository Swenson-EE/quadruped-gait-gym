from simulator.simulation.history import JointHistory

class SimulationState:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.joints = JointHistory()


