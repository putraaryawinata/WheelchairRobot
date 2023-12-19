import tensorflow as tf
import numpy as np

parent_dir = "/home/aryawinata/belajar/WheelchairRobot"

import sys
sys.path.append(parent_dir)
from metrics import R_squared
from utils import ExtendedKalmanFilter as EKF
from utils import PositionDisplacement

if __name__ == "__main__":
    # Preprocess Data
    parent_dir = "/home/aryawinata/belajar/WheelchairRobot"
    input_data = np.load(f"{parent_dir}/data/noise_acc.npy")
    gt_data = np.load(f"{parent_dir}/data/true_pos.npy")

    # Predict Data
    model = tf.keras.models.load_model("generate_data/ae_acc2acc.h5", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", R_squared])
    ae_acc = model.predict(input_data)

    # EKF Processing (comment this code if you are not using it!)
    ####################################################################################################################################
    ae_velo = np.zeros(ae_acc.shape)
    ae_pos = np.zeros(ae_acc.shape)
    ekf_ae_pos = np.zeros(ae_acc.shape)
    for index in range(len(ae_acc)):
        ## Define position and velocity data for ae_acc
        pdis_ae_x = PositionDisplacement(ae_acc[index, :, 0])
        ae_velo[index,:,0] = pdis_ae_x.v()
        ae_pos[index,:,0] = pdis_ae_x.s()

        pdis_ae_y = PositionDisplacement(ae_acc[index,:, 1])
        ae_velo[index,:,1] = pdis_ae_y.v()
        ae_pos[index,:,1] = pdis_ae_y.s()
        
        ## Covariance for EKF simulation
        Q = np.diag([
            0.1,  # variance of location on x-axis
            1.0  # variance of velocity
        ]) ** 2  # predict state covariance
        R = np.diag([1.0]) ** 2  # Observation x position covariance
        dt = 1

        data_state_x = np.append(ae_pos[index, :, 0].reshape(-1, 1), ae_velo[index, :, 0].reshape(-1, 1), axis=1).reshape(-1, 2, 1)
        data_input_x = np.append(ae_velo[index, :, 0].reshape(-1, 1), ae_acc[index, :, 0].reshape(-1, 1), axis=1).reshape(-1, 2, 1)  

        ekf_x = EKF(data_state_x, data_input_x, R, Q, dt)
        xEst, z_x = ekf_x.ekf()
        
        data_state_y = np.append(ae_pos[index, :, 1].reshape(-1, 1), ae_velo[index, :, 1].reshape(-1, 1), axis=1).reshape(-1, 2, 1)
        data_input_y = np.append(ae_velo[index, :, 1].reshape(-1, 1), ae_acc[index, :, 1].reshape(-1, 1), axis=1).reshape(-1, 2, 1)  

        ekf_y = EKF(data_state_y, data_input_y, R, Q, dt)
        yEst, z_y = ekf_y.ekf()

        ekf_ae_pos[index] = np.append(xEst[0, :61].reshape(-1, 1), yEst[0, :61].reshape(-1, 1), axis=1)
    ####################################################################################################################################

    try:
        np.save(f"{parent_dir}/data/ekf_auto_pos.npy", ekf_ae_pos)
        print("Success to create pos data for autoencoder+ekf signal")
    except:
        raise("Error occurs on creating data for autoencoder+ekf signal")

