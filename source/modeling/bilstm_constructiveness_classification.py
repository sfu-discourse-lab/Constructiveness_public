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
#from tflearn.datasets import imdb
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
import sys
sys.path.append('../../')
from config import Config
from deep_learning_data_preprocessing_and_representation import *
import pickle as pkl

class BiLSTMConstructivenessClassifier():
    def __init__(self, train_data_path = Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_12000, 
                       test_data_path = Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_1000, 
                       mode = 'train',
                       model_path = Config.BILSTM_MODEL_PATH):        

        self.model = None
        self.net = None
        self.initialize()
        self.word_to_index = {}
        self.index_to_word = {}
        index = 0 
        self.words = []
        '''for line in self.dictionary:
            line = line.strip()
            #line = line.decode("utf-8", errors='ignore')
            self.word_to_index[line] = index
            self.index_to_word[index] = line
            self.words.append(line)
            index+=1
           '''
        if mode == 'train': 
            train_set = create_numeric_representation_of_text_and_labels()
            self.train_bilstm(train_set)
        else:
            self.model.load(model_path)                       
            
        #print('Test Accuracy', self.model.evaluate(X=x_test, Y=y_test))
        #accuracy = self.predict(self.model, x_test, y_test)
        #print('Accuracy: ', accuracy)
       
    def my_evaluate(self, predictions, gold):
        '''
        :param predictions:
        :param gold:
        :return:
        '''
        count = 0
        correct = 0
        total = 0
        gold_data = []
        preds = []
        print('#GOLD: ', len(gold))
        print('#PREDICTIONS: ', len(predictions))
        for i in range(len(gold)):
            gold_winner = 0 if gold[i][0] > gold[i][1] else 1
            gold_data.append(gold_winner)
            prediction_winner = 0 if predictions[i][0] > predictions[i][1] else 1
            preds.append(prediction_winner)
            if gold_winner == prediction_winner:
                correct += 1
            total += 1

        print('Correct/Total ', correct, '/', total)
        accuracy = (correct * 100) / float(total)
        print('Accuracy: ', accuracy)

        print('LSTM correct prediction: {:4.2f}'.format(np.mean(preds == gold_data)))
        print(metrics.classification_report(gold_data, preds, target_names=['non-constructive', 'constructive']))

        return accuracy

    
    def predict(self, model, testX, testY):
        print(model.evaluate(testX, testY))

        predictions = model.predict(testX)
        count = 0
        for prediction in predictions:
            print(prediction, '=>', testY[count])
            count += 1
        accuracy = my_evaluate(predictions, testY)
        return accuracy
        

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


    def indices_to_sentences(self, index_list):
        '''
        :param index_list:
        :return:
        '''
        comment_list = [self.index_to_word[idx].decode("utf-8") for idx in index_list]
        return " ".join(comment_list)

    def predict_single_example(self, test_comment):
        #print('model loaded')
        test_example = prepare_test_example(test_comment)
        #print('test example prepaped', test_example)
        prediction = self.model.predict(test_example)
        #prediction = predict_test_example(model, test_example)
        prediction_winner = 'Non constructive' if prediction[0][0] > prediction[0][1] else 'Constructive'
        print(prediction, '=>', prediction_winner)
        return prediction_winner    
    
    def prepare_test_example(self, test_comment):
        comment_tokenized = tokenize(test_comment)
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
    

    def train(self, trainX, trainY, model_path = Config.MODEL_PATH + 'SOCC_bilstm.tflearn', 
              ibatch_size=512):
        # Training
        gf = smart_open.smart_open(Config.GLOVE_EMBEDDINGS_PATH, 'rb')
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
        self.model.save(model_path)
        print('Model saved at the following location: ', model_path)
        return self.model

    
    def train_bilstm(self, train_set):
        trainX, trainY, validationX, validationY = get_preprocessed_and_padded_train_validation_splits(train_set)
        #trainX, trainY, testX, testY, validationX, validationY = self.get_train_test_validation_data()
        # self.prepare_train_validation_data()    
        #model_path = (model_path + '_' + str(hyperparameters['BATCH_SIZE']) + \
        #              '_' + str(hyperparameters['LEARNING_RATE']) + '.tflearn')
        self.model = self.train(trainX, trainY)
        print('Validation Accuracy', self.model.evaluate(X=validationX, Y=validationY))
        #print('Test Accuracy', self.model.evaluate(X=testX, Y=testY))

        
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
    #bilstm_model_path = Config.MODEL_PATH + 'SOCC_annotated.tflearn'
    input_data_path = Config.TRAIN_PATH + 'SOCC_NYT_picks_constructive_YNACC_non_constructive.csv'
    bilstm = BiLSTMConstructivenessClassifier(input_data_path)
    # train mode

    sys.exit(0)

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
    