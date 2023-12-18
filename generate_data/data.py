import numpy as np
import os

parent_dir = "/home/aryawinata/belajar/WheelchairRobot"

import sys
sys.path.append(parent_dir)
from utils import PositionDisplacement

# NOISE DATA
"""
Require: acceleration, velocity, and position data
"""
noise_acc = np.zeros((200, 61, 2))
noise_velo = np.zeros((200, 61, 2))
noise_pos = np.zeros((200, 61, 2))
for i in range(100):
    # data_1e-1
    ## define acceleration
    x = np.load(f"{parent_dir}/data/data_1e-1/data_x_{i}.npy")
    y = np.load(f"{parent_dir}/data/data_1e-1/data_y_{i}.npy")

    ## determine position and velocity:
    pdis_x = PositionDisplacement(x)
    pos_x = pdis_x.s()
    velo_x = pdis_x.v()
    pdis_y = PositionDisplacement(y)
    pos_y = pdis_y.s()
    velo_y = pdis_y.v()

    ## create dataset
    acc_arr = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    noise_acc[i] = acc_arr
    acc_velo = np.append(velo_x.reshape(-1, 1), velo_y.reshape(-1, 1), axis=1)
    noise_velo[i] = acc_velo
    acc_pos = np.append(pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), axis=1)
    noise_pos[i] = acc_pos

    # data_1e-1
    ## define acceleration
    x = np.load(f"{parent_dir}/data/data_5e-1/data_x_{i}.npy")
    y = np.load(f"{parent_dir}/data/data_5e-1/data_y_{i}.npy")

    ## determine position:
    pdis_x = PositionDisplacement(x)
    pos_x = pdis_x.s()
    velo_x = pdis_x.v()
    pdis_y = PositionDisplacement(y)
    pos_y = pdis_y.s()
    velo_y = pdis_y.v()

    ## create dataset
    acc_arr = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    noise_acc[i+100] = acc_arr
    acc_velo = np.append(velo_x.reshape(-1, 1), velo_y.reshape(-1, 1), axis=1)
    noise_velo[i+100] = acc_velo
    acc_pos = np.append(pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), axis=1)
    noise_pos[i+100] = acc_pos

try:
    np.save(f"{parent_dir}/data/noise_acc.npy", noise_acc)
    print(f"Shape of noise_acc: {noise_acc.shape}")
    np.save(f"{parent_dir}/data/noise_velo.npy", noise_velo)
    print(f"Shape of noise_velo: {noise_acc.shape}")
    np.save(f"{parent_dir}/data/noise_pos.npy", noise_pos)
    print(f"Shape of noise_pos: {noise_acc.shape}")
    print("Success to create acc, velo, and pos data for noise signal")
except:
    raise("Error occurs on creating data for noise signal")

# GROUND TRUTH DATA
"""
Require: acceleration and position data
"""
true_acc = np.zeros((200, 61, 2))
true_pos = np.zeros((200, 61, 2))

## define acceleration
x = np.load(f"{parent_dir}/data/data_true.npy")[:, 0]
y = np.load(f"{parent_dir}/data/data_true.npy")[:, 1]

## determine position:
pdis_x = PositionDisplacement(x)
pos_x = pdis_x.s()
pdis_y = PositionDisplacement(y)
pos_y = pdis_y.s()

for i in range(200):
    true_acc[i] = np.load(f"{parent_dir}/data/data_true.npy")
    true_pos[i] = np.append(pos_x.reshape(-1, 1), pos_y.reshape(-1, 1), axis=1)

try:
    np.save(f"{parent_dir}/data/true_acc.npy", true_acc)
    print(f"Shape of true_acc: {true_acc.shape}")
    np.save(f"{parent_dir}/data/true_pos.npy", true_pos)
    print(f"Shape of true_pos: {true_pos.shape}")
    print("Success to create acc and pos data for ground truth signal")
except:
    raise("Error occurs on creating data for ground truth signal")