# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:44:46 2018

@author: Batuhan
"""

from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Concatenate, Input, Dropout



def build_model(sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg):
    
    inputs = Input(shape = (sequence_length, embedding_size))
    
    # Construction of Convolutional layer
    convs = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters = num_filters, kernel_size = filter_size, activation = "relu")(inputs)
        pooled = MaxPooling1D(filter_size)(conv)
        convs.append(pooled)
        
    concatenate_layer = Concatenate(axis = 1)(convs)

    # Dropout
    
    drop_layer = Dropout(0.5)(concatenate_layer)
    
    # Fully-connected layer
    
    predictions = Dense(num_classes, activation = "softmax")(drop_layer)
    
    model = Model(inputs = inputs, outputs = predictions)
    return model

model = build_model(10, 3, 3, 20,[3,2], 2, 2)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()