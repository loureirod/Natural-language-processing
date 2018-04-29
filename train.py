#! /usr/bin/env python

import tensorflow as tf
import argparse
import numpy as np
import time
import pre_processing
import Model
#from tensorflow.contrib import learn

#from keras import backend as K
from keras.backend import set_session

args = argparse.ArgumentParser()

args.add_argument("--dev_sample_percentage", type=float, default=.1)
#args.add_argument("--positive_data_file" ,type = str, default = "./data/rt-polaritydata/pos")# "Data source for the positive data.")
#args.add_argument("--negative_data_file" ,type = str, default = "./data/rt-polaritydata/neg") #"Data source for the negative data."


# Model Hyperparameters
args.add_argument("--embedding_dim", type = int, default = 300) # "Dimensionality of character embedding (default: 128)")
args.add_argument("--filter_sizes", type = str, default = "3,4,5")# "Comma-separated filter sizes (default: '3,4,5')")
args.add_argument("--num_filters", type = int,default = 100) # "Number of filters per filter size (default: 128)")
args.add_argument("--dropout_keep_prob", type = float, default = 0.5) # "Dropout keep probability (default: 0.5)")
args.add_argument("--l2_reg_lambda", type = float, default = 3) # "L2 regularization lambda (default: 0.0)")

# Training parameters
args.add_argument("--batch_size", type=int, default=50)
args.add_argument("--num_epochs", type = int, default = 10) # "Number of training epochs (default: 200)")
args.add_argument("--evaluate_every", type = int, default = 1) # "Evaluate model on dev set after this many steps (default: 100)")
args.add_argument("--checkpoint_every", type = int, default = 10) # "Save model after this many steps (default: 100)")
args.add_argument("--num_checkpoints", type = int, default = 5) # "Number of checkpoints to store (default: 5)")

## Misc Parameters
#args.add_argument("--allow_soft_placement", type = bool, default = True) # "Allow device soft device placement")
#args.add_argument("--log_device_placement", type = bool, default = False) # "Log placement of ops on devices")

FLAGS, unparsed = args.parse_known_args()


#%% Data Preparation
# ==================================================
def reshape_data(x_text,y_text):
    nb_reviews = len(y_text); max_length_text = 2746 ; embbeding_size  = 300
    x = np.zeros((nb_reviews,max_length_text,embbeding_size)) 
    y = np.zeros((nb_reviews,2))
    for review in range(nb_reviews):
        length_text = np.shape(x_text.at[review,'Messages'])[0]
        for word in range(length_text):
            x[review] = x_text.at[review,'Messages'][word]
        if y_text.at[review,0] == 0:
            y[review] = np.array([1,0])    
        else:
            y[review] = np.array([0,1])
    return x,y

print("Loading data...")
x_text, y_text = pre_processing.load_dataset("./MR")

print("Reshaping data...")
x_train, y_train = reshape_data(x_text[:1000] ,y_text[:1000])
#x_train2, y_train2 = reshape_data(x_shuffle[500:1000] ,y_shuffle[500:1000])
x_dev, y_dev = reshape_data(x_text[1000:1200].reset_index(drop=True), y_text[1000:1200].reset_index(drop=True))



'''
# Load data
print("Loading data...")
x_text, y_text = pre_processing.load_dataset("./MR")
nb_reviews = len(x_text) ; index_shuffle = np.random.permutation(nb_reviews)

print("Reshaping data...")
x_shuffle = x_text.iloc[index_shuffle].reset_index(drop=True)
y_shuffle = y_text.iloc[index_shuffle].reset_index(drop=True)
x_train, y_train = reshape_data(x_shuffle[:1000] ,y_shuffle[:1000])
#x_train2, y_train2 = reshape_data(x_shuffle[500:1000] ,y_shuffle[500:1000])
x_dev, y_dev = reshape_data(x_shuffle[1000:1200].reset_index(drop=True), y_shuffle[1000:1200].reset_index(drop=True))
'''

#%% Training
# ==================================================

with tf.Graph().as_default():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) #run cnn on GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session()) 
#    session = K.get_session()
      
    cnn = Model.build_model(max_sequence_length=2746,
            num_classes=2,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg=FLAGS.l2_reg_lambda)
    
#    cnn.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics=['accuracy'])
    cnn.compile(optimizer = "adadelta",loss = "categorical_crossentropy",metrics = ['accuracy']) 
    
    global_start_time = time.time()
#    print('Loading dev set')
#    x_dev,y_dev = reshape_data(x_text,y_text,0)
    for i in range(FLAGS.num_epochs//FLAGS.evaluate_every):
#        print('Loading new train set')
#        x_train,y_train = reshape_data(x_text,y_text,i+1)
        print('TRAIN epoch {}'.format((i+1)*FLAGS.evaluate_every)) 
        start_time = global_start_time
        cnn.fit(x_train,y_train,
                epochs = FLAGS.evaluate_every,
                batch_size = FLAGS.batch_size,
                shuffle = True,
                verbose = 1)
    #                initial_epoch = (i+1)*FLAGS.evaluate_every)
        end_time = time.time()
        print('Evaluate on dev set')
        l,g = cnn.evaluate(x_dev,y_dev)
        print("Time to train: {}, Loss on dev set at epoch {} : loss = {:g}, accuracy = {:g}".format(end_time - start_time,round((i+1)*FLAGS.evaluate_every,2),l,g))
        
        cnn.save("cnn_model.h5",overwrite = True)

    
            
    