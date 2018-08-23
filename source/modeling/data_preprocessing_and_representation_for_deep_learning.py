__author__ = 'VaradaKolhatkar'

#from tflearn.datasets import imdb
import numpy as np
import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk import sent_tokenize
import tflearn
import sys, os, argparse, pickle, random, smart_open
from tflearn.data_utils import to_categorical, pad_sequences

import sys
sys.path.append('../../')
from config import Config
import pickle as pkl

def get_glove_dictionary():
    '''
    :param glove_dict_path:
    :return:
    '''
    import smart_open
    f = smart_open.smart_open(Config.GLOVE_DICTIONARY_PATH,'rb')
    index = 0
    dictionary = {}
    for line in f:
        try: 
            dictionary[line.strip().decode("utf-8")] = index
            index+=1
        except:
            print('Could not convert the string: ', line)
    return dictionary


# A global variable for the dictionary, which will be used in many functions. 
dictionary = get_glove_dictionary()
    
    
def tokenize(comment):
    tokenized = " ".join(sent_tokenize(" ".join(word_tokenize(comment))))
    return tokenized


def text_2_dict_ids_list(text):
    return [dictionary[w] if w in dictionary else 1 for w in tokenize(text).strip().split()]


def create_numeric_representation_of_text_and_labels(data_path = Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_12000,
                                                    text_col = 'comment_text', 
                                                    target_col = 'constructive'):
    '''
    '''
    df = pd.read_csv(data_path)
    df[text_col + '_dict_ids'] = df[text_col].apply(text_2_dict_ids_list)                
    df_pos_subset = df[df[target_col] == 1]
    df_neg_subset = df[df[target_col] == 0]
    x_pos = df_pos_subset[text_col + '_dict_ids'].tolist()        
    x_neg = df_neg_subset[text_col + '_dict_ids'].tolist()
    y_pos = [1] * len(x_pos)
    y_neg = [0] * len(x_neg)        
    x = x_pos + x_neg
    y = y_pos + y_neg       
    return (x, y)


def split_and_sort_data(train_set, n_words=100000, valid_portion=0.1,
                        maxlen=None,
                        sort_by_len=True):
    '''Loads the dataset
    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    '''

    #############
    # LOAD DATA #
    #############

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    #test_set_x, test_set_y = test_set

    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    #test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        #sorted_index = len_argsort(test_set_x)
        #test_set_x = [test_set_x[i] for i in sorted_index]
        #test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    #test = (test_set_x, test_set_y)

    return train, valid #, test


def pad_vectors(X, maxlen=200, value=0.0): 
    # Data preprocessing
    # Sequence padding
    X = pad_sequences(X, maxlen=maxlen, value=value)
    return X


def binarize_labels(Y, nb_classes=2): 
    # Converting labels to binary vectors        
    Y = to_categorical(Y, nb_classes=2)
    return Y        


def get_preprocessed_and_padded_train_validation_splits(train_set):
    train, validation = split_and_sort_data(train_set=train_set, n_words=50000, valid_portion=0.1)
    trainX, trainY = train
    validationX, validationY = validation
    #testX, testY = test

    print('trainX[-1] after load data: ', trainX[1000])


    print('Train_len: ', len(trainX), len(trainY))
    print('Validation_len: ', len(validationX), len(validationY))
    #print('Test len: ', len(testX), len(testY))
    #print('test data: ', testX[0:10])

    trainX = pad_vectors(trainX)        
    trainY = binarize_labels(trainY)

    validationX = pad_vectors(validationX)        
    validationY = binarize_labels(validationY)        

    print('len of trainX, validationX', len(trainX), ' ', len(validationX))
    return trainX, trainY, validationX, validationY


def prepare_test_example(test_comment):
    comment_tokenized = tokenize(test_comment)
    mapped_comment = []
    for w in comment_tokenized.split():
        w = w.lower()
        w = w.strip()
        w = w.encode()
        if w in dictionary: 
            mapped_comment.append(dictionary[w])
        else:
            mapped_comment.append(1)

    testX = [mapped_comment]
    testX = pad_vectors(testX)
    return testX         
                
if __name__=="__main__":
    train_set = create_numeric_representation_of_text_and_labels()
    trainX, trainY, validationX, validationY = get_preprocessed_and_padded_train_validation_splits(train_set)
    
    