from simulator.physics.core import RandomizationSubsystem
from scipy.spatial.transform import Rotation

class InitialPoseRandomizer(RandomizationSubsystem):
    """
    Randomizes the initial pose of the robot.
    position_range: (m)
    rotation_range: (rad)
    """

    def __init__(self, context, position_range=(-0.1, 0.1), rotation_range=(-0.5, 0.5)):
        super().__init__(context)

        self.position_range = position_range
        self.rotation_range = rotation_range

    def apply(self, rng):
        self.context.data.qpos[0:3] += self.random_position(rng)
        self.context.data.qpos[3:7] += self.random_rotation(rng)

    def random_position(self, rng):
        return rng.uniform(*self.position_range, size=3)

    def random_rotation(self, rng):
        # random small roll, pitch, yaw (radians)
        rpy = rng.uniform(*self.rotation_range, size=3)
        q = Rotation.from_euler('xyz', rpy).as_quat()

        # apply to root orientation (assumes qpos[3:7] is quaternion)
        return [q[3], q[0], q[1], q[2]]
