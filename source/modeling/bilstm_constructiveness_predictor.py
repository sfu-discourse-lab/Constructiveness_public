__author__ = 'VaradaKolhatkar'

"""
Training bidirectional LSTM for constructiveness. 

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""

#from __future__ import division, print_function, absolute_import

import tflearn
import sys, os, argparse, pickle, random, smart_open
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk import sent_tokenize
tf.logging.set_verbosity(tf.logging.INFO)
ROOT = '/home/ling-discourse-lab/Varada/'

class Constructiveness_biLSTM():
    def __init__(self, model_path, input_data_path=None, mode='test'):        
        self.model_path = model_path
        self.model = None
        self.net = None    
        self.initialize()        
        self.glove_path = ROOT + 'data/glove/glove.840B.vocab'    
        gf = smart_open.smart_open(self.glove_path, 'rb')
        self.word_to_index = {}
        self.index_to_word = {}
        index = 0 
        self.words = []
        for line in gf:
            line = line.strip()
            #line = line.decode("utf-8", errors='ignore')
            self.word_to_index[line] = index
            self.index_to_word[index] = line
            self.words.append(line)
            index+=1
        if mode == 'train':
            self.input_data_path = input_data_path
            self.train_bilstm()
        elif mode == 'test':
            #print('network built')
            self.model.load(self.model_path)

    def build_network(self, ilearning_rate=0.001):
        '''
        :param ilearning_rate:
        :return:
        Build network
        '''

        #tf.reset_default_graph()
        net = input_data(shape=[None, 200])
        net = embedding(net, input_dim=51887, output_dim=200, trainable=False, name="EmbeddingLayer")

        #net = embedding(net, input_dim=20000, output_dim=128, trainable=False, weights_init=W,
        #                        name="EmbeddingLayer")
        #net = tflearn.embedding(net, input_dim=20000, output_dim=128, trainable=False, weights_init = W, name="EmbeddingLayer")
        net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
        net = dropout(net, 0.5)
        net = fully_connected(net, 2, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy'
                         , learning_rate=ilearning_rate)

        return net

    def initialize(self):
        self.net = self.build_network()
        self.model = tflearn.DNN(self.net, clip_gradients=0., tensorboard_verbose=0)

    def tokenize(self, comment):
        tokenized = " ".join(sent_tokenize(" ".join(word_tokenize(comment))))
        return tokenized

    def indices_to_sentences(self, index_list):
        '''
        :param index_list:
        :return:
        '''
        comment_list = [self.index_to_word[idx].decode("utf-8") for idx in index_list]
        return " ".join(comment_list)

    def prepare_test_example(self, test_comment):
        comment_tokenized = self.tokenize(test_comment)
        mapped_comment = []
        for w in comment_tokenized.split():
            w = w.lower()
            w = w.strip()
            w = w.encode()
            if w in self.word_to_index: 
                mapped_comment.append(self.word_to_index[w])
            else:
                mapped_comment.append(1)
                
        testX = [mapped_comment]
        testX = pad_sequences(testX, maxlen=200, value=0.)
        return testX 
    
    def prepare_data(self):
        train, validation, test = imdb.load_data(path=self.input_data_path, n_words=50000, valid_portion=0.1)
        trainX, trainY = train
        validationX, validationY = validation
        testX, testY = test
        
        print('trainX[-1] after load data: ', trainX[1000])
        print(self.indices_to_sentences(trainX[1000]))

        print('Train_len: ', len(trainX), len(trainY))
        print('Validation_len: ', len(validationX), len(validationY))
        print('Test len: ', len(testX), len(testY))
        print('test data: ', testX[0:10])
        
        # Data preprocessing
        # Sequence padding
        trainX = pad_sequences(trainX, maxlen=200, value=0.)
        testX = pad_sequences(testX, maxlen=200, value=0.)
        validationX = pad_sequences(validationX, maxlen=200, value=0.)

        print(trainX[0:10])
        print('len of trainX, validationX, and testX', len(trainX), ' ', len(validationX), ' ', len(testX))
        
        # Converting labels to binary vectors        
        trainY = to_categorical(trainY, nb_classes=2)
        validationY = to_categorical(validationY, nb_classes=2)
        testY = to_categorical(testY, nb_classes=2)
        print('len of trainY, validationY, and testY', len(trainY), ' ', len(validationY), ' ', len(testY))
        return trainX, trainY, testX, testY, validationX, validationY        

    def train(self, trainX, trainY, ibatch_size=512):
        # Training
        gf = smart_open.smart_open(ROOT + 'data/glove/glove.pickle', 'rb')
        glove_embeddings = pickle.load(gf)
        shortened_embeddings = np.zeros((51887,200))
        index = 0
        for item in glove_embeddings:
            shortened_embeddings[index, :] = item[:200]
            index += 1

        # Retrieve embedding layer weights (only a single weight matrix, so index is 0)
        embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
        print('default embeddings: ', embeddingWeights[0])

        self.model.set_weights(embeddingWeights, shortened_embeddings)

        self.model.fit(trainX, trainY, validation_set=0.1, n_epoch=10, show_metric=True, batch_size=ibatch_size)
        self.model.save(self.model_path)
        print('Model saved at the following location: ', self.model_path)
        return self.model

    def train_bilstm(self):
        trainX, trainY, testX, testY, validationX, validationY = self.prepare_data()    
        #model_path = (model_path + '_' + str(hyperparameters['BATCH_SIZE']) + \
        #              '_' + str(hyperparameters['LEARNING_RATE']) + '.tflearn')
        self.model = self.train(trainX, trainY)
        
    def predict_bilstm(self, test_comment):
        #print('model loaded')
        test_example = self.prepare_test_example(test_comment)
        #print('test example prepaped', test_example)
        prediction = self.model.predict(test_example)
        #prediction = predict_test_example(model, test_example)
        prediction_winner = 'Non constructive' if prediction[0][0] > prediction[0][1] else 'Constructive'
        print(prediction, '=>', prediction_winner)
        return prediction_winner

    def getHyperparameters(self, tune=False):
        if tune:
            hyperparams=({
            'BATCH_SIZE':random.choice([128,256,512]),
            'LEARNING_RATE':random.choice([0.001, 0.01, 0.0001]),
            #'LEARNING_RATE':random.uniform(0.001, 0.09),
            'OPTIMIZER':random.choice(["SGD", "Adam", "Adagrad"])
            })
        else:
            hyperparams = ({
            'BATCH_SIZE':512,
            'LEARNING_RATE':0.001,
            'OPTIMIZER':"Adam"
            })

        return hyperparams        
    
if __name__=="__main__":
    ROOT = '/home/ling-discourse-lab/Varada/'
    bilstm_model_path = ROOT + 'output/intermediate_output/models/NYT_picks_train_SFU_test.tflearn'
    input_data_path = ROOT + 'output/intermediate_output/pickle_files/NYT_YNC_plain_train_SFU_test.pkl'
    
    # train mode
    # bilstm = Constructiveness_biLSTM(bilstm_model_path, input_data_path, 'train')
    
    # test mode
    bilstm = Constructiveness_biLSTM(bilstm_model_path)    
    comment1 = "Harper promised 15% discount of 1000 dollars spent on children's sports activities. Mulcair promises 15 dollar a day child care. I will go with Mulcair."
    comment2 = "It will collect dust on a shelf somewhere."
    comment3 = "Very well said... and thank you for shouting it from the rooftops. The morning my children and I woke up to the horrible reality of Trump winning, my 14 year-old daughter asked me how it happened how a woman could be hated so much that a racist, sexist bully could be the preferable choice over Ms. Clinton. Barely able to explain it to myself, I struggled to find the words I needed to tell her that her voice, her vote, her existence mattered in the face of such insanity taking place in the U.S. Days later, I'm still trying to find the answer and I appreciate your take on it. If Trump was a protest vote, then clearly Americans chose to protest civility, inclusion, reason, kindness and collaboration. More than ever before in my life (and yes, I was born in the States but have lived here since I was 10), I'm grateful to be a Canadian, but I'm not sure I will ever be able to explain this presidency to my daughter or my two sons. One thing for sure, with strong, intelligent women like you and so many others willing to speak up and lean in, my daughter will have role models to follow. Thank you."
    print('Comment: ', comment1)
    print('Prediction: ', bilstm.predict_bilstm(comment1))
    print('-----------')
    print('Comment: ', comment2)
    print('Prediction: ', bilstm.predict_bilstm(comment2))
    print('-----------')
    print('Comment: ', comment3)
    print('Prediction: ', bilstm.predict_bilstm(comment3))
    print('-----------')
    