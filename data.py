import numpy as np


dataset = np.zeros((200, 2, 61))
for i in range(100):
    # data_1e-1
    x = np.load(f"data/data_1e-1/data_x_{i}.npy")
    y = np.load(f"data/data_1e-1/data_y_{i}.npy")
    arr = np.array([x, y])
    dataset[i] = arr

    # data_5e-1
    x = np.load(f"data/data_5e-1/data_x_{i}.npy")
    y = np.load(f"data/data_5e-1/data_y_{i}.npy")
    arr = np.array([x, y])
    dataset[i+100] = arr

np.save("dataset.npy", dataset)
