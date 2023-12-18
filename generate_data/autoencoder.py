import tensorflow as tf
import numpy as np

parent_dir = "/home/aryawinata/belajar/WheelchairRobot"

import sys
sys.path.append(parent_dir)
from metrics import R_squared

if __name__ == "__main__":
    # Preprocess Data
    parent_dir = "/home/aryawinata/belajar/WheelchairRobot"
    input_data = np.load(f"{parent_dir}/data/noise_acc.npy")
    gt_data = np.load(f"{parent_dir}/data/true_pos.npy")

    # Predict Data
    model = tf.keras.models.load_model("autoencoder.h5", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", R_squared])
    ae_pos = model.predict(input_data)
    try:
        np.save(f"{parent_dir}/data/auto_pos.npy", ae_pos)
        print("Success to create pos data for autoencoder signal")
    except:
        raise("Error occurs on creating data for autoencoder signal")

