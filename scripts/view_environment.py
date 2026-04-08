import gymnasium as gym
import mujoco.viewer
import time
import numpy as np
import json

from environment.base_bittle_environment import BaseBittleEnvironment
from environment.continuous_bittle_environment import ContinuousBittleEnvironment

TEST_MODE = "manual"  # Set to "manual" for manual stepping, "auto" for automatic stepping
ENVIRONMENT = 'c'

step_flag = False
def key_callback(keycode):
    global step_flag
    if keycode == ord(' '):
        step_flag = True

if __name__ == "__main__":
    np.set_printoptions(linewidth=120)

    weights = {}
    with open("config/reward_weights.json") as file:
        weights = json.load(file)
        

    env: BaseBittleEnvironment = None
    match ENVIRONMENT.lower():
        case 'c':
            env = ContinuousBittleEnvironment(weights=weights)
        case _:
            print(f"Invalid environment type {ENVIRONMENT}")
            exit()

    obs, info = env.reset()
    
    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if TEST_MODE == 'auto' or (TEST_MODE == 'manual' and step_flag):
                step_flag = False

                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                viewer.sync()

                time.sleep(0.01)  # Adjust the sleep time as needed for smoother rendering

                if terminated or truncated:
                    obs, info = env.reset()

    env.close()
