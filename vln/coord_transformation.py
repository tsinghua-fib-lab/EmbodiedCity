import airsim
import numpy as np


# xyzw to roll, pitch, yaw
def quaternion2eularian_angles(quat):
    pry = airsim.to_eularian_angles(quat)    # p, r, y
    return np.array([pry[1], pry[0], pry[2]])


