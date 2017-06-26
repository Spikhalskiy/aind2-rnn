import numpy as np

import string
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    last_window_start_index = len(series) - window_size
    X = [series[start_index:start_index+window_size] for start_index in range(last_window_start_index)]
    first_result_pos = window_size
    y = [series[result_index] for result_index in range(first_result_pos, len(series))]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text


    # remove as many non-english characters and character sequences as you can
    correct_punctuation = set([' ', "'", "-", '!', ',', '.', ':', ';', '?'])
    correct_symbols = correct_punctuation.union(string.ascii_lowercase)
    incorrect_symbols = list(correct_symbols.symmetric_difference(set(text)))
    print("Incorrect symbols that will be deleted", incorrect_symbols)
    for incorrect_symbol in incorrect_symbols:
        text = text.replace(incorrect_symbol, ' ')
    text = text.replace('--', '-')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    last_window_start_index = len(text) - window_size
    inputs = [text[start_index:start_index+window_size] for start_index in range(0, last_window_start_index, step_size)]
    first_result_pos = window_size
    outputs = [text[result_index] for result_index in range(first_result_pos, len(text), step_size)]
    
    return inputs, outputs

### TODO: build the required RNN model: a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, categories_number):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, categories_number)))
    model.add(Dense(categories_number))
    model.add(Activation('softmax'))
    return model
