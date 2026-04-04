from simulator.core.subsystem.node import Node

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.bittle_sim import BittleSimulator


class Subsystem(Node):
    def __init__(self, parent):
        super().__init__(parent)
    
    def initialize(self):
        pass


    @property
    def model(self):
        return self.sim.model
    
    @property
    def data(self):
        return self.sim.data
    