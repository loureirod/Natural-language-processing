import os,sys
import re
import pandas as pd
import numpy as np
import gensim.models as gm


def split_text(text):
    '''Split words and english short forms '''

    text = re.sub(r"\'s", " \'s", text) 
    text = re.sub(r"\'ve", " \'ve", text) 
    text = re.sub(r"n\'t", " n\'t", text) 
    text = re.sub(r"\'re", " \'re", text) 
    text = re.sub(r"\'d", " \'d", text) 
    text = re.sub(r"\'ll", " \'ll", text) 
    text = re.sub(r",", " , ", text) 
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " \( ", text) 
    text = re.sub(r"\)", " \) ", text) 
    text = re.sub(r"\?", " \? ", text)

    text = text.lower()

    return text.split()

def load_word2vec_file(word2vec_file):

    print("Loading " + word2vec_file + " ... It might take some minutes ... ")
    model = gm.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

    return model

def extract_words(negative_path,positive_path):

    negative_files = os.listdir(negative_path)
    positive_files = os.listdir(positive_path)

    labels = [0]*len(negative_files) + [1]*len(positive_files)

    all_texts = []

    texts_max_lenght = 0


    for file_path in negative_files:
        text_file = open(negative_path + "/" + file_path,mode='r')  
        words = split_text( text_file.read() )
        all_texts.append( words )
        texts_max_lenght = max(texts_max_lenght,len(words))
        text_file.close()

    for file_path in positive_files:
        text_file = open(positive_path + "/" + file_path,mode='r')  
        words = split_text( text_file.read() )
        all_texts.append( words )
        texts_max_lenght = max(texts_max_lenght,len(words))        
        text_file.close()

    return all_texts,labels,texts_max_lenght

def text2vec(model,all_texts,texts_max_lenght):
    '''Return dataset and unknow words'''
    I = len(all_texts)
    J = texts_max_lenght
    K = 300

    J_test = 0

    dataset = pd.DataFrame(index=range(0,I),columns=[k for k in range(0,J)])
    
    unkown_words = set()

    print("Building dataset...")   

    for i,text in enumerate(all_texts):
        for j,word in enumerate(text):
            try:
                dataset[i][j] = model.word_vec(word)
            except KeyError:
                dataset[i][j] = np.random.uniform(-1,1,size=300)
                unkown_words.add(word)

        for k in range(j,J):
            dataset[i][k] = np.zeros(300)

        if i%100 == 0:
            print(i)

    print("J:"+str(J))        

    return dataset,unkown_words

def vec2csv(dataset,labels,path):
    '''Path should be a path to a folder. Filenames will be dataset.pkl and labels.pkl'''

    dataset.to_pickle(path + "/dataset.pkl")
    labels = pd.DataFrame(labels)
    labels.to_pickle(path + "/labels.pkl")

def load_dataset(path):
    '''Returns (dataset,labels) where dataset is a set of vector representations of stored in path'''
    
    dataset = pd.read_pickle(path + "/dataset.pkl")
    labels = pd.read_pickle(path + "/labels.pkl")

    return dataset, labels

def cosine_similarity(vect1,vect2):
    return np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))

def vector_query(model,word):
    return model.word_vec(word)

if __name__=='__main__':

    # Compute vector representation of words and store it on HDD

    word2vec_file = "word2vec/GoogleNews-vectors-negative300.bin"

    negative_path = "MR/reviews/neg"
    positive_path = "MR/reviews/pos"

    all_texts,labels,texts_max_lenght = extract_words(negative_path,positive_path)

    model = load_word2vec_file(word2vec_file)

    dataset,unkown_words = text2vec(model,all_texts,texts_max_lenght)

    choice = input("Do you want to store dataset and labels on HDD ? [yes/no]")

    if choice == "yes":
        print("Storing on hdd ...")
        vec2csv(dataset,labels,"MR")
