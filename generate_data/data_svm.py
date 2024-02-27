import numpy as np

var_data = 2
mean = 0

even_plane = np.append(np.append(3*np.random.rand(1000,1) - 1.5*np.ones((1000,1)), 3*np.random.rand(1000,1) - 1.5*np.ones((1000,1)), axis=1),
                       3*np.random.rand(1000,1) + 8.3*np.ones((1000,1)), axis=1)
# even_plane = np.append(even_plane, np.zeros((1000,1)), axis=1)
# np.save("data/even_plane.npy", even_plane)

fwd_lean_plane = np.append(np.append(3*np.random.rand(1000,1) + 1.3*np.ones((1000,1)), 3*np.random.rand(1000,1) - 1.5*np.ones((1000,1)), axis=1),
                           3*np.random.rand(1000,1) + 7.5*np.ones((1000,1)), axis=1)
# fwd_lean_plane = np.append(fwd_lean_plane, np.ones((1000,1)), axis=1)
# np.save("data/fwd_lean_plane.npy", fwd_lean_plane)

bwd_lean_plane = np.append(np.append(3*np.random.rand(1000,1) - 1.3*np.ones((1000,1)), 3*np.random.rand(1000,1) - 1.5*np.ones((1000,1)), axis=1),
                           3*np.random.rand(1000,1) + 7.3*np.ones((1000,1)), axis=1)
# bwd_lean_plane = np.append(bwd_lean_plane, 2*np.ones((1000,1)), axis=1)
# np.save("data/bwd_lean_plane.npy", bwd_lean_plane)

X = np.append(np.append(even_plane, fwd_lean_plane, axis=0), bwd_lean_plane, axis=0)
y = np.append(np.append(np.zeros((1000, 1)), np.ones((1000, 1)), axis=0), 2*np.ones((1000, 1)), axis=0)
np.save("data/svm_x.npy", X)
np.save("data/svm_y.npy", y)
