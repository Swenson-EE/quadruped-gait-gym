from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.bittle_sim import BittleSimulator
    

class Node:

    parent = None
    sim: "BittleSimulator" = None

    def __init__(self, parent):
        self.parent = parent
        self.sim = parent.sim if hasattr(parent, "sim") else parent

    def reset(self, rng):
        pass
    
    def step(self, rng):
        pass

