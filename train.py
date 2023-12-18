import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input, Flatten

if __name__ == "__main__":
    print("Training starts!")