import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PositionDisplacement:
    def __init__(self, acc_arr, dt=1):
        self.acc_arr = acc_arr
        self.dt = dt

    def vt(self, a_now, a_prev, v_prev, init=False):
        """
        This function defines the velocity respect to an axis at a certain time
        """
        # print(f"a_now: {a_now}, a_prev: {a_prev}, v_prev: {v_prev}")
        return v_prev+0.5*self.dt*(a_now+a_prev)

    def st(self, v_now, v_prev, s_prev, init=False):
        """
        This function defines the position respect to an axis at a certain time
        """
        return s_prev+0.5*self.dt*(v_now+v_prev)

    def v(self):
        v_arr = [0]
        for i in range(len(self.acc_arr)):
            if i == 0:
                continue
            v_i = self.vt(self.acc_arr[i], self.acc_arr[i-1], v_arr[i-1])
            v_arr.append(round(v_i, 10))

        return np.array(v_arr)

    def s(self):
        s_arr = [0]
        v_arr = self.v()
        for i in range(len(self.acc_arr)):
            if i == 0:
                continue
            s_i = self.st(v_arr[i], v_arr[i-1], s_arr[i-1])
            s_arr.append(round(s_i, 10))

        return np.array(s_arr)

class ExtendedKalmanFilter:
    # RUN THIS
    def __init__(self, data_state, data_input, R, Q, dt=1, show_animation=False):
        self.data_state = data_state
        self.data_input = data_input
        self.R = R
        self.Q = Q
        self.dt = dt
        self.show_animation = show_animation
    

    def motion_model(self, X, U, dt):
        # x.shape = (2, 1)
        F = np.array([[1.0, 0,],
                    [0, 1.0,]])

        B = np.array([[dt, 0],
                    [0, dt]])

        X = F @ X + B @ U
        return X

    def observation_model(self, x):
        H = np.array([
            [1, 0],
            [0, 1]
        ])
        z = H @ x
        return z

    def jacob_f(self, x, u, dt):
        # x_t = x_{t-1} + v*dt
        # v_t = v_{t-1} + a*dt
        # hence:
        # dx/dv = dt
        # dx/da = 0
        # dv/dv = 1
        # dv/da = dt
        jF = np.array([
            [dt, 0],
            [1, dt],
        ])
        return jF

    def jacob_h(self):
        # Jacobian of Observation Model
        jH = np.array([
            [1, 0],
            [0, 1]
        ])
        return jH

    def ekf_estimation(self, xEst, PEst, z, u, R, Q, dt):
        #  Predict
        xPred = self.motion_model(xEst, u, dt)
        jF = self.jacob_f(xEst, u, dt)
        PPred = jF @ PEst @ jF.T + Q

        #  Update
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    def ekf(self):
        time = 0.0

        # State Vector [x v]
        xEst = self.data_state[0]
        PEst = np.eye(2)

        # xDR = np.zeros((4, 1))  # Dead reckoning

        # history
        hxEst = xEst
        hz = np.zeros((2, 1))

        for i in range(len(self.data_input)):
            time += 1
            z, ud = self.data_input[i], self.data_input[i]

            xEst, PEst = self.ekf_estimation(xEst, PEst, z, ud, self.R, self.Q, self.dt)

            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hz = np.hstack((hz, z))

            if self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(hz[0, :], hz[1, :], ".g")
                plt.plot(hxEst[0, :].flatten(),
                        hxEst[1, :].flatten(), "-r", label="est")
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                plt.pause(0.001)
        return hxEst, hz

class Animation:
    def __init__(self, *args):
        """
        Input: must be a pair of x and y array, have a dimension of (n, 2)
        """
        self.n = args[0].shape[0]
        self.x = np.zeros((len(args), self.n, 2))
        self.y = np.zeros((len(args), self.n, 2))
        self.graph = []
        for index, arr in enumerate(args):
            self.x[index] = arr[:, 0]
            self.y[index] = arr[:, 1]
            graph_index, = plt.plot([], [], 'o', label=f"graph {index+1}")
            self.graph.append(graph_index)
        
        self.xlim = [np.min(self.x), np.max(self.x)]
        self.ylim = [np.min(self.y), np.max(self.y)]
    
    def animate(self, i):
        for index in range(self.graph):
            self.graph[index].set_data(self.x[index,:i], self.y[index,:i])
        return self.graph
    
    def build_animation(self, save_name="anim", ext="gif", fps=5):
         fig = plt.figure()
         plt.xlim(self.xlim[0], self.xlim[1])
         plt.ylim(self.ylim[0], self.ylim[1])

         ani = animation.FuncAnimation(fig, self.animate, frames=self.n, interval=200)
         ani.save(f"{save_name}.{ext}", writer='ffmpeg', fps=fps)