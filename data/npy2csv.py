import numpy as np

name = ["ekf_auto_pos", "ekf_pos", "noise_pos", "auto_pos"]
for n in name:
    arr = np.load(f"{n}.npy")
    np.savetxt(f"{n}.csv", arr[0], delimiter=",")