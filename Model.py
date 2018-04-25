# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:44:46 2018
@author: Batuhan
"""
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Concatenate, Input, Dropout, Masking, Reshape,Flatten
import numpy as np


def build_model(max_sequence_length,num_classes, embedding_size, filter_sizes, num_filters, l2_reg):

    inputs = Input(shape = (max_sequence_length,embedding_size,))
#    input_sentence = Reshape(target_shape = (max_sequence_length, embedding_size))(inputs)
#    mask = Masking(mask_value = float('nan'),
#                   input_shape = (max_sequence_length, embedding_size))(input_sentence)
#    sequence_length = K.shape(mask).eval(session = session)[0]
        
#    input_sentence = Reshape(target_shape = (max_sequence_length, embedding_size))(inputs)

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

#    concatenate_layer = Concatenate(axis = 0)(convs)

    flat_layer = Flatten()(pooled)
    
    # Dropout

    drop_layer = Dropout(0.5)(flat_layer)
    

    # Fully-connected layer

    predictions = Dense(num_classes, activation = "softmax")(drop_layer)

    model = Model(inputs = inputs, outputs = predictions)
    return model

#tf_session = K.get_session()
#model1 = build_model(200,2, 300,[2,3,4], 100, 0)
#model1.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#model1.summary()
