from simulator.core.subsystem.subsystem import Subsystem
from simulator.core.registry import register_module
from simulator.modules import Randomization

from scipy.spatial.transform import Rotation


@register_module(Randomization, position_range=(-0.1, 0.1), rotation_range=(-0.2, 0.2))
class InitialPoseRandomizer(Subsystem):
    """
    Randomizes the initial pose of the robot.
    position_range: (m)
    rotation_range: (rad)
    """
    def __init__(self, sim, position_range, rotation_range):
        super().__init__(sim)

        self.position_range = position_range
        self.rotation_range = rotation_range


    def reset_start(self, rng):
        rand_pos = self.random_position(rng)
        self.data.qpos[0:2] += rand_pos # only randomize x and y, leave z alone

        rand_quat = self.random_rotation(rng)
        self.data.qpos[3:7] += rand_quat

    def random_position(self, rng):
        return rng.uniform(*self.position_range, size=2)

    def random_rotation(self, rng):
        # random small roll, pitch, yaw (radians)
        rpy = rng.uniform(*self.rotation_range, size=3)
        
        q = Rotation.from_euler('xyz', rpy).as_quat()

        # apply to root orientation (assumes qpos[3:7] is quaternion)
        return [q[3], q[0], q[1], q[2]]
