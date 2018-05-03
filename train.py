#! /usr/bin/env python

import tensorflow as tf
import argparse
import numpy as np
import pre_processing
import Model

from keras.backend import set_session

args = argparse.ArgumentParser()

# Model Hyperparameters
args.add_argument("--max_review_length", type = int, default = 2746) # "Maximum number of word in a review"
args.add_argument("--embedding_dim", type = int, default = 300) # "Dimensionality of character embedding"
args.add_argument("--filter_sizes", type = str, default = "3,4,5")# "Comma-separated filter sizes"
args.add_argument("--number_filters", type = int,default = 100) # "Number of filters per filter size"
args.add_argument("--dropout_rate", type = float, default = 0.5) # "Dropout keep probability"
args.add_argument("--l2_reg", type = float, default = 3) # "L2 regularization lambda"

# Training parameters
args.add_argument("--batch_size", type=int, default=50) #Mini-batch size used for training
args.add_argument("--number_epochs", type = int, default = 10) # "Number of training epochs"
args.add_argument("--evaluate_every", type = int, default = 1) # "Evaluate model on dev set after this many steps"

FLAGS, unparsed = args.parse_known_args()


#%% Data Preparation
# ==================================================

def reshape_data(x_text,y_text):
    nb_reviews = len(y_text); max_length_text = FLAGS.max_review_length ; embbeding_size  = 300
    x = np.zeros((nb_reviews,max_length_text,embbeding_size))
    y = np.zeros((nb_reviews,2))
    for review in range(nb_reviews):
        length_text = np.shape(x_text.at[review,'Messages'])[0]
        for word in range(length_text):
                x[review][word] = x_text.at[review,'Messages'][word]
        if y_text.at[review,0] == 0:
            y[review] = np.array([1,0])    
        else:
            y[review] = np.array([0,1])
    return x,y



# Load data
print("Loading data...")
x_text, y_text = pre_processing.load_dataset("./MR")
nb_reviews = len(y_text) ; index_shuffle = np.random.permutation(nb_reviews)

print("Data sampling...")
x_shuffle = x_text.iloc[index_shuffle].reset_index(drop=True)
y_shuffle = y_text.iloc[index_shuffle].reset_index(drop=True)

del x_text,y_text

print("Reshaping data...")
x_train, y_train = reshape_data(x_shuffle[:600] ,y_shuffle[:600])
x_dev, y_dev = reshape_data(x_shuffle[1700:2000].reset_index(drop=True), y_shuffle[1700:2000].reset_index(drop=True))

del x_shuffle,y_shuffle

#%% Training
# ==================================================
print('Training... \n')
with tf.Graph().as_default():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) #run cnn on GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session()) 
      
    cnn = Model.build_model(max_sequence_length=FLAGS.max_review_length,
            num_classes=2,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.number_filters,
            dropout_rate = FLAGS.dropout_rate,
            l2_reg=FLAGS.l2_reg)
    
    cnn.compile(optimizer = "adadelta",loss = "categorical_crossentropy",metrics = ['accuracy']) 

    for i in range(FLAGS.number_epochs//FLAGS.evaluate_every):
        print('TRAIN epoch {}'.format((i+1)*FLAGS.evaluate_every)) 
        
        cnn.fit(x_train,y_train,
                epochs = FLAGS.evaluate_every,
                batch_size = FLAGS.batch_size,
                shuffle = True,
                verbose = 1)
        
        print('Evaluating on dev set')
        l,g = cnn.evaluate(x_dev,y_dev)
                           #,batch_size = FLAGS.batch_size)
        print("Loss on dev set at epoch {} : loss = {:g}, accuracy = {:g} \n".format((i+1)*FLAGS.evaluate_every,l,g))
        
    cnn.save("cnn_model.h5",overwrite = True)

    
            
    