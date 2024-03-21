import numpy as np

name = ["ekf_auto_pos", "ekf_pos", "noise_pos", "auto_pos"]
new_name = ["noise_pos", "ekf_pos", "auto_pos", "ekf_auto_pos"]
for index in range(200):
    for i in range(len(name)):
        arr = np.load(f"{name[i]}.npy")
        np.savetxt(f"all/{new_name[i]}_{index}.csv", arr[index], delimiter=",")

# arr = np.load(f"svm_y.npy")
# print(arr.shape)
# np.savetxt(f"svm_y.csv", arr, delimiter=",")