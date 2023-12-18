import numpy as np
from utils import ExtendedKalmanFilter as EKF

# EKF DATA
"""
Require position data only
"""
ekf_pos = np.zeros((200, 61, 2))

# Parameter EKF:
noise_acc = np.load("data/noise_acc.npy")
noise_velo = np.load("data/noise_velo.npy")
noise_pos = np.load("data/noise_pos.npy")
dt = 1
# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0]) ** 2  # Observation x position covariance

for index in range(200):
    data_state_x = np.append(noise_pos[index, :, 0].reshape(-1, 1), noise_velo[index, :, 0].reshape(-1, 1), axis=1).reshape(-1, 2, 1)
    data_input_x = np.append(noise_velo[index, :, 0].reshape(-1, 1), noise_acc[index, :, 0].reshape(-1, 1), axis=1).reshape(-1, 2, 1)  

    ekf_x = EKF(data_state_x, data_input_x, R, Q, dt)
    xEst, z_x = ekf_x.ekf()
    
    data_state_y = np.append(noise_pos[index, :, 1].reshape(-1, 1), noise_velo[index, :, 1].reshape(-1, 1), axis=1).reshape(-1, 2, 1)
    data_input_y = np.append(noise_velo[index, :, 1].reshape(-1, 1), noise_acc[index, :, 1].reshape(-1, 1), axis=1).reshape(-1, 2, 1)  

    ekf_y = EKF(data_state_y, data_input_y, R, Q, dt)
    yEst, z_y = ekf_y.ekf()

    ekf_pos[index] = np.append(xEst[0, :61].reshape(-1, 1), yEst[0, :61].reshape(-1, 1), axis=1)

try:
    np.save("data/ekf_pos.npy", ekf_pos)
    print(f"Shape of ekf_pos: {ekf_pos.shape}")
    print("Success to create pos data for EKF signal")
except:
    raise("Error occurs on creating data for EKF signal")
