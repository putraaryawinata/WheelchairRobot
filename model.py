import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, \
    Input, Flatten, Conv1D, MaxPool1D, UpSampling1D, Reshape

def model_autoencoder(input_shape, ext_lstm_units, int_lstm_units, metrics=["mae"]):
    input_layer = Input(shape=input_shape)
    encoder = Conv1D(ext_lstm_units, kernel_size=3, padding="same", activation="relu")(input_layer)
    mp = MaxPool1D()(encoder)
    encoder_1 = Conv1D(int_lstm_units, kernel_size=3, padding="same", activation="relu")(mp)
    mp_1 = MaxPool1D()(encoder_1)

    decoder = Conv1D(int_lstm_units, kernel_size=3, padding="same", activation="relu")(mp_1)
    us = UpSampling1D()(decoder)
    decoder_1 = encoder = Conv1D(ext_lstm_units, kernel_size=3, padding="same", activation="relu")(us)
    us_1 = UpSampling1D()(decoder_1)
    flatten = Flatten()(us_1)
    fc = Dense(122, activation="relu")(flatten)
    fc_1 = Dense(122, activation="linear")(fc)

    output_layer = Reshape((61, 2))(fc_1)

    ######

    # input_layer = Input(shape=input_shape)
    # encoder = LSTM(ext_lstm_units, activation='relu', return_sequences=True)(input_layer)
    # encoder_1 = LSTM(int_lstm_units, activation='relu', return_sequences=True)(encoder)
    # # flatten = Flatten()(encoder_1)
    # output_layer = Dense(input_shape[1])(encoder_1)

    ######
    
    # input_layer = Input(shape=input_shape, name="input")
    # encoder = LSTM(ext_lstm_units, activation='relu', return_sequences=True, name="first_encoder")(input_layer)
    # encoder_1 = LSTM(int_lstm_units, activation='relu', return_sequences=True, name="second_encoder")(encoder)
    # encoder_2 = LSTM(int_lstm_units, activation='relu', name="third_encoder")(encoder_1)
    # repeat = RepeatVector(input_shape[0], name="repeat")(encoder_2)
    # decoder = LSTM(int_lstm_units, activation='relu', return_sequences=True, name="first_decoder")(repeat)
    # decoder_1 = LSTM(int_lstm_units, activation='relu', return_sequences=True, name="second_decoder")(decoder)
    # decoder_2 = LSTM(ext_lstm_units, activation='relu', return_sequences=True, name="third_decoder")(decoder_1)
    # output_layer = TimeDistributed(Dense(input_shape[1]), name="output")(decoder_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=metrics)
    return model