from dataclasses import field
import numpy as np

from simulator.core.subsystem.module import Module
from simulator.core.registry import register_module
from simulator.modules import Physics

#import simulator.modules.physics.kinematics_systems as ks


@register_module(Physics)
class Kinematics(Module):
    """
    Handles info such as positions, orientations, and velocities
    """
    pass

