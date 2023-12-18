import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input, Flatten

from model import model_autoencoder
from metrics import R_squared
from callbacks import early_stopping

if __name__ == "__main__":
    print("Training starts!")

    # Data Preprocessing
    input_data = np.load("data/noise_acc.npy")
    gt_data = np.load("data/true_pos.npy")

    x_train, x_test, y_train, y_test = train_test_split(input_data, gt_data, test_size=0.2, random_state=42)
    input_shape = x_train.shape[1:]

    # Build Model
    model = model_autoencoder(input_shape, ext_lstm_units=16, int_lstm_units=32, metrics=["mae", R_squared])
    model.summary()
    history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), callbacks=[early_stopping])
    print(f"Training stops!")
    model.save("autoencoder.h5")
    