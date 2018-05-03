
import numpy as np
from keras.models import Model
from keras.constraints import max_norm
from keras.layers import Dense, Conv1D, MaxPooling1D, Concatenate, Input, Dropout,Flatten


def build_model(max_sequence_length,num_classes, embedding_size, filter_sizes, num_filters, dropout_rate,l2_reg):

    inputs = Input(shape = (max_sequence_length,embedding_size,))

    # Construction of Convolutional layer
    convs = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters = num_filters,
                      kernel_size = filter_size,
                      activation = "relu",
                      padding = "valid")(inputs)
        pooled = MaxPooling1D(pool_size =max_sequence_length-filter_size+1,
                              strides = 1,
                              padding = "valid")(conv)
        convs.append(pooled)

    concatenate_layer = Concatenate(axis = 1)(convs)

    flat_layer = Flatten()(concatenate_layer)
    
    # Dropout

    drop_layer = Dropout(dropout_rate)(flat_layer)
    
    # Fully-connected layer

    predictions = Dense(num_classes, 
						activation = "softmax",
						kernel_constraint=max_norm(l2_reg))(drop_layer)

    model = Model(inputs = inputs, outputs = predictions)
    return model

