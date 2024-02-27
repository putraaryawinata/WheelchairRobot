import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

true_pos = np.load("data/true_pos.npy")[0]
noise_pos = np.load("data/noise_pos.npy")[0]
ekf_pos = np.load("data/ekf_pos.npy")[0]
auto_pos = np.load("data/auto_pos.npy")[0]
ekf_auto_pos = np.load("data/ekf_auto_pos.npy")[0]

def error_analysis():
    ea = pd.DataFrame()
    ea["Algoritma"] = ["Tanpa algoritma (controlled)", "Hanya Kalman Filter (KF)", "Hanya Autoencoder", "Autoencoder dengan KF"]
    ea["Root Mean Squared Error (m) - sumbu-x"] = [np.mean(abs(true_pos-ekf_auto_pos), axis=0)[0], np.mean(abs(true_pos-ekf_pos), axis=0)[0],
                                                   np.mean(abs(true_pos-noise_pos), axis=0)[0], np.mean(abs(true_pos-auto_pos), axis=0)[0],
                                                   ]
    ea["Root Mean Squared Error (m) - sumbu-y"] = [np.mean(abs(true_pos-ekf_auto_pos), axis=0)[1], np.mean(abs(true_pos-ekf_pos), axis=0)[1],
                                                   np.mean(abs(true_pos-noise_pos), axis=0)[1], np.mean(abs(true_pos-auto_pos), axis=0)[1],
                                                   ]
    ea["Root Mean Squared Error (m) - overall"] = [np.mean(abs(true_pos-ekf_auto_pos)), np.mean(abs(true_pos-ekf_pos)),
                                                   np.mean(abs(true_pos-noise_pos)), np.mean(abs(true_pos-auto_pos)),
                                                   ]
    ea["Accuracy (%)"] = [r2_score(true_pos, ekf_auto_pos), r2_score(true_pos, ekf_pos),
                          r2_score(true_pos, noise_pos), r2_score(true_pos, auto_pos),
                          ]
    return ea
print(error_analysis())
error_analysis().to_csv("Tabel_Analisis_Error.csv")
# print(np.mean(abs(true_pos-noise_pos)))
# print(np.mean(abs(true_pos-ekf_pos), axis=0))
# print(np.mean(abs(true_pos-auto_pos), axis=0))
# print(np.mean(abs(true_pos-ekf_auto_pos), axis=0))
