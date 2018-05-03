# Natural-language-processing
Modelling and sentence classification with convolutional neural networks.

This project is based on the work of Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188) and the one of Yoon Kim [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882v2)

## Pre-processing

Vector description of words is based on the algorithm [word2vec](https://code.google.com/archive/p/word2vec/). In this project pre-trained words are loaded from word2vec applied on part of Google News dataset (about 100 billion words). The pre-trained words are  available here: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

Our model is trained on the Movie Review dataset that can be downloaded [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz). This dataset separates movie reviews in two categories: positive and negative reviews 

Since the Machine Learning algorithm learns from vector representations of words the dataset should be converted into a set of vectors. To perform that, __pre_processing.py__ should be executed alone only a single time to convert the dataset using word2vec algorithm and store the result in the files dataset.pkl and labels.pkl . 
Note that the following code from __pre_processing.py__ should be adapted to your configuration:
```python
if __name__=='__main__':

    # Compute vector representation of words and store it on HDD

    word2vec_file = "word2vec/GoogleNews-vectors-negative300.bin"

    negative_path = "MR/reviews/neg"
    positive_path = "MR/reviews/pos"
```

After that you juste have to run the following code everytime you want to load the new dataset :

```python
import pre_processing as pre

dataset,labels = pre.load_dataset("path/to/folder")
```

## Training

The script train.py is in charge of the training. It imports pre_processing.py and Model.py and train the cnn model on the training set. In order to execute this code, make sure you have enough RAM, otherwise you will get a Memory Error while running :
```python
print("Reshaping data...")
x_train, y_train = reshape_data(x_shuffle[:600] ,y_shuffle[:600])
```
If there is still a Memory Error, please try the following code :
```
import gc

gc.collect()
```


