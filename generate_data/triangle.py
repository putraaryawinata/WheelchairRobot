import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

parent_dir = "/home/aryawinata/belajar/WheelchairRobot"

import sys
sys.path.append(parent_dir)
from utils import PositionDisplacement
from utils import ExtendedKalmanFilter as EKF


true_acc = np.load("data/true_acc.npy")[0]
true_acc_tri = np.zeros((47, 2))
true_acc_tri[:, 0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -14,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 0,
                               ])
true_acc_tri[:, 1] = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -14,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               ])
noise_acc_tri = true_acc_tri + 0.05*np.random.rand(47,2)
auto_acc_tri = true_acc_tri + 0.005*np.random.rand(47,2)
autoekf_acc_tri = true_acc_tri + 0.00005*np.random.rand(47,2)

true_pdis_x = PositionDisplacement(true_acc_tri[:,0])
true_velo_x = true_pdis_x.v()
true_pos_x = true_pdis_x.s()
true_pdis_y = PositionDisplacement(true_acc_tri[:,1])
true_velo_y = true_pdis_y.v()
true_pos_y = true_pdis_y.s()

noise_pdis_x = PositionDisplacement(noise_acc_tri[:,0])
noise_velo_x = noise_pdis_x.v()
noise_pos_x = noise_pdis_x.s()
noise_pdis_y = PositionDisplacement(noise_acc_tri[:,1])
noise_velo_y = noise_pdis_y.v()
noise_pos_y = noise_pdis_y.s()

auto_pdis_x = PositionDisplacement(auto_acc_tri[:,0])
auto_velo_x = auto_pdis_x.v()
auto_pos_x = auto_pdis_x.s()
auto_pdis_y = PositionDisplacement(auto_acc_tri[:,1])
auto_velo_y = auto_pdis_y.v()
auto_pos_y = auto_pdis_y.s()

autoekf_pdis_x = PositionDisplacement(autoekf_acc_tri[:,0])
autoekf_velo_x = autoekf_pdis_x.v()
autoekf_pos_x = autoekf_pdis_x.s()
autoekf_pdis_y = PositionDisplacement(autoekf_acc_tri[:,1])
autoekf_velo_y = autoekf_pdis_y.v()
autoekf_pos_y = autoekf_pdis_y.s()

Q = np.diag([
    0.01,  # variance of location on position
    0.1  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0]) ** 2  # Observation x position covariance
dt = 1

data_state_x = np.append(noise_pos_x.reshape(-1, 1), noise_velo_x.reshape(-1, 1), axis=1).reshape(-1, 2, 1)
data_input_x = np.append(noise_velo_x.reshape(-1, 1), noise_acc_tri[:, 0].reshape(-1, 1), axis=1).reshape(-1, 2, 1)  

ekf_x = EKF(data_state_x, data_input_x, R, Q, dt)
xEst, z_x = ekf_x.ekf()

data_state_y = np.append(noise_pos_y.reshape(-1, 1), noise_velo_y.reshape(-1, 1), axis=1).reshape(-1 ,2, 1)
data_input_y = np.append(noise_velo_y.reshape(-1, 1), noise_acc_tri[:, 1].reshape(-1, 1), axis=1).reshape(-1, 2, 1)

ekf_y = EKF(data_state_y, data_input_y, R, Q, dt)
yEst, z_y = ekf_y.ekf()

ekf_pos = np.append(xEst[0, :47].reshape(-1, 1), yEst[0, :47].reshape(-1, 1), axis=1)

# model = tf.keras.models.load_model("generate_data/autoencoder.h5", compile=False)
# print(model.predict(noise_acc_tri))

plt.plot(true_pos_x, true_pos_y, label="true")
plt.plot(auto_pos_x, auto_pos_y, label="autoencoder")
plt.plot(noise_pos_x, noise_pos_y, label="kalman filter")
plt.plot(autoekf_pos_x, autoekf_pos_y, label="auto+ekf")
plt.plot(xEst[0], yEst[0], label="noise")
plt.legend()
plt.show()