import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

def model_autoencoder(input_shape, ext_lstm_units, int_lstm_units, metrics=["mae"]):
    input_layer = Input(shape=input_shape, name="input")
    encoder = LSTM(ext_lstm_units, activation='relu', return_sequences=True, name="first encoder")(input_layer)
    encoder_1 = LSTM(int_lstm_units, activation='relu', return_sequences=True, name="second encoder")(encoder)
    repeat = RepeatVector(input_shape[0], name="repeat")(encoder_1)
    decoder = LSTM(8, activation='relu', return_sequences=True, name="first decoder")(repeat)
    decoder_1 = LSTM(int_lstm_units, activation='relu', return_sequences=True, name="second decoder")(decoder)
    output_layer = TimeDistributed(Dense(input_shape[1]), name="output")(decoder_1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=metrics)
    return model