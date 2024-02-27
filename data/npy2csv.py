import numpy as np

name = ["ekf_auto_pos", "ekf_pos", "noise_pos", "auto_pos"]
new_name = ["noise_pos", "ekf_pos", "auto_pos", "ekf_auto_pos"]
for i in range(len(name)):
    arr = np.load(f"{name[i]}.npy")
    np.savetxt(f"{new_name[i]}.csv", arr[0], delimiter=",")