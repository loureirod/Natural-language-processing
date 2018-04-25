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
args.add_argument("--filter_sizes", type = str, default = "3")# "Comma-separated filter sizes (default: '3,4,5')")
args.add_argument("--num_filters", type = int,default = 100) # "Number of filters per filter size (default: 128)")
args.add_argument("--dropout_keep_prob", type = float, default = 0.5) # "Dropout keep probability (default: 0.5)")
args.add_argument("--l2_reg_lambda", type = float, default = 0.0) # "L2 regularization lambda (default: 0.0)")

# Training parameters
args.add_argument("--batch_size", type=int, default=20)
args.add_argument("--num_epochs", type = int, default = 50) # "Number of training epochs (default: 200)")
args.add_argument("--evaluate_every", type = int, default = 10) # "Evaluate model on dev set after this many steps (default: 100)")
args.add_argument("--checkpoint_every", type = int, default = 10) # "Save model after this many steps (default: 100)")
args.add_argument("--num_checkpoints", type = int, default = 5) # "Number of checkpoints to store (default: 5)")

## Misc Parameters
#args.add_argument("--allow_soft_placement", type = bool, default = True) # "Allow device soft device placement")
#args.add_argument("--log_device_placement", type = bool, default = False) # "Log placement of ops on devices")

FLAGS, unparsed = args.parse_known_args()


#%% Data Preparation
# ==================================================
def reshape_data(x,y):
    nb_reviews,max_text_length = x.shape ; embedding_size = x[0][0].shape[0]
    max_text_length = 2000
    x_new = np.zeros((nb_reviews,max_text_length,embedding_size))
    y_new = np.zeros((nb_reviews,1,1))
    for review in range(nb_reviews):
        for word in range(max_text_length):
            x_new[review][word] = x[review][word]
        y_new[review] = np.ones((1,1))
    x_new = np.reshape(x_new,(nb_reviews,max_text_length,embedding_size,1))
    y_new = np.reshape(y_new,(nb_reviews,1,1,1))
    return x_new, y_new

    
# Load data
print("Loading data...")
x_text, y_text = pre_processing.load_dataset("./MR")
dev_sample_index = int(0.1*len(y_text))

x_train = np.zeros((200,2000,300)) ; x_dev = np.zeros((200,2000,300))
y_train = np.zeros((200,2)) ; y_dev = np.zeros((200,2)) 
for review in range(200):
    for word in range(2000):
        x_train[review][word] = x_text[word][10*review]
        x_dev[review][word] = x_text[word][5+10*review]
    if y_text[0][10*review] == 0:
        y_train[review] = np.array([1,0])
    else:
        y_train[review] = np.array([0,1])
    if y_text[0][5+10*review] == 0:
        y_train[review] = np.array([1,0])
    else:
        y_train[review] = np.array([0,1])



'''
#x_train, x_dev = x_text[:dev_sample_index][:2000], x_text[dev_sample_index:2*dev_sample_index][:2000]
#y_train, y_dev = y_text[:dev_sample_index], y_text[dev_sample_index:2*dev_sample_index]

x_train,y_train = reshape_data(x_train,y_train)
x_dev,y_dev = reshape_data(x_dev,y_dev)
# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))

#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
#y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]


#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
'''

#%% Training
# ==================================================

with tf.Graph().as_default():
#    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) #run cnn on GPU
#    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session()) 
#    session = K.get_session()
      
    cnn = Model.build_model(max_sequence_length=2000,
            num_classes=2,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg=FLAGS.l2_reg_lambda)
    
#    cnn.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics=['accuracy'])
    cnn.compile(optimizer = "adadelta",loss = "categorical_crossentropy",metrics = ['accuracy']) 
    
    global_start_time = time.time()
    for i in range(FLAGS.evaluate_every):
        start_time = global_start_time
        cnn.fit(x_train,y_train,
                epochs = FLAGS.num_epochs//FLAGS.evaluate_every,
                batch_size = FLAGS.batch_size,
                shuffle = True,
                verbose = 1)
        end_time = time.time()
        eval_batch = cnn.evaluate(x_dev,y_dev)
#               batch_size = None,
#               verbose = 0,
#               steps = y_dev.shape[1]//FLAGS.batch_size)
        print("Time to train: {}, Loss on dev set at epoch {} : {}".format(end_time - start_time,i*FLAGS.evaluate_every,eval_batch))
        
        cnn.save("cnn_model.h5",overwrite = True)

    
            
    