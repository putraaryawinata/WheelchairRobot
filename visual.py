import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from utils import Animation

def shift(pos, shift=15):
    new_pos = np.zeros((pos.shape))
    new_pos = pos
    new_pos[:, 0] = pos[:, 0] + shift
    return new_pos

true_pos = shift(np.load("data/true_pos.npy")[0])
auto_pos = shift(np.load("data/noise_pos.npy")[0])
ekf_pos = shift(np.load("data/ekf_pos.npy")[0])
noise_pos = shift(np.load("data/ekf_auto_pos.npy")[0])
ekf_auto_pos = shift(np.load("data/auto_pos.npy")[0])

plt.plot(true_pos[:, 0], true_pos[:, 1], label="true")
plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], label="kalman filter")
plt.plot(noise_pos[:, 0], noise_pos[:, 1], label="noise")
plt.legend()
plt.savefig("kf_plot.png")

plt.clf()

plt.plot(true_pos[:, 0], true_pos[:, 1], label="true")
plt.plot(auto_pos[:, 0], auto_pos[:, 1], label="autoencoder")
plt.plot(noise_pos[:, 0], noise_pos[:, 1], label="noise")
plt.legend()
plt.savefig("auto_plot.png")

plt.clf()

plt.plot(true_pos[:, 0], true_pos[:, 1], label="true")
plt.plot(auto_pos[:, 0], auto_pos[:, 1], label="autoencoder")
plt.plot(ekf_pos[:, 0], ekf_pos[:, 1], label="kalman filter")
plt.plot(ekf_auto_pos[:, 0], ekf_auto_pos[:, 1], label="auto+ekf")
plt.plot(noise_pos[:, 0], noise_pos[:, 1], label="noise")
plt.legend()
plt.savefig("autoekf_plot.png")

plt.clf()

# for index in range(1):
#     true_pos = np.load("data/true_pos.npy")[index]
#     noise_pos = np.load("data/noise_pos.npy")[index]
#     ekf_pos = np.load("data/ekf_pos.npy")[index]
#     auto_pos = np.load("data/auto_pos.npy")[index]
#     ekf_auto_pos = np.load("data/ekf_auto_pos.npy")[index]

#     fig = plt.figure()
#     plt.xlim(-100, 200)
#     plt.ylim(-100, 200)
#     graph, = plt.plot([], [], 'o')
#     graph1, = plt.plot([], [], 'o')
#     graph2, = plt.plot([], [], 'o')
#     graph3, = plt.plot([], [], 'o')
#     graph4, = plt.plot([], [], 'o')
    
#     def animate(i):
#         graph.set_data(true_pos[:i+1, 0], true_pos[:i+1, 1])
#         graph1.set_data(noise_pos[:i+1, 0], noise_pos[:i+1, 1])
#         graph2.set_data(ekf_pos[:i+1, 0], ekf_pos[:i+1, 1])
#         graph3.set_data(auto_pos[:i+1, 0], auto_pos[:i+1, 1])
#         graph4.set_data(ekf_auto_pos[:i+1, 0], ekf_auto_pos[:i+1, 1])

#         return [graph, graph1, graph2, graph3, graph4]

#     ani = animation.FuncAnimation(fig, animate, frames=61, interval=200)
#     ani.save(f'visual/mpu_{index}.gif', writer = 'ffmpeg', fps = 5)
#     plt.clf()

#     print(f"Success animating data at {index}")
