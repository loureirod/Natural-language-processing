# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:04:23 2018

@author: Batuhan
"""

import tensorflow as tf

class Build_model(object):
    
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg):
        
        # Create a convolution layer based on word2vec representation of words
        
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length, embedding_size, 1], name= "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name= "input_y")
        
        #L2 regularization loss
        
        l2_loss = tf.constant(0.0)
        
        
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)
        outputs = []
        
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-%s" % filter_size):
                # Convolutional layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID",name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")
                
                # Maxpool on a window of size ksize
                
                pooled = tf.nn.max_pool(h, ksize = [1, sequence_length - filter_size + 1, 1, 1], strides = [1, 1, 1, 1], padding = "VALID", name = "maxpool")
                outputs.append(pooled)
                
                total_num_filters = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(outputs, 3)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_num_filters])
                
                # Dropout
                
                with tf.name_scope("dropout"):
                    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
                
                # Fully-connected layer
                
                with tf.name_scope("output"):
                    W = tf.get_variable( "w", shape = [total_num_filters, num_classes], initializer = tf.contrib.layers.xavier_initializer())
                    b = tf.variable(tf.constant(0.1, shape = [num_classes]), name= "b")
                    
                    # Compute the l2 loss for the l2 regularization
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    
                    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                    self.predictions = tf.argmax(self.scores, 1, name="predictions")
                    
                with tf.name_scope("loss"):
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                    self.loss = tf.reduce_mean(losses) + l2_reg * l2_loss
                        
                    # Accuracy
                with tf.name_scope("accuracy"):
                    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                    