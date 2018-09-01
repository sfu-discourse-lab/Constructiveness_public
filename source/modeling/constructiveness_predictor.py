__author__ = 'VaradaKolhatkar'

import pandas as pd
import numpy as np
import argparse
from sklearn.externals import joblib
import sys
sys.path.append('../../')
from config import Config
sys.path.append(Config.PROJECT_HOME + 'source/feature_extraction/')

import feature_extractor
from feature_extractor import FeatureExtractor
from svm_constructiveness_classification import ConstructivenessClassifier
from bilstm_constructiveness_classification import BiLSTMConstructivenessClassifier

import nltk 
from nltk import word_tokenize
from nltk import sent_tokenize


class ConstructivenessPredictor():
    def __init__(self, svm_model_path = Config.SVM_MODEL_PATH,
                 bilstm_model_path = Config.BILSTM_MODEL_PATH):
        '''
        :param model_path: str (model path)

        Description: This class assumes that you have a feature-based trained model. It returns the prediction of the
        given example based on the trained model.
        '''
        # load svm model
        self.svm_pipeline = joblib.load(svm_model_path)
        
        # load bilstm model
        self.bilstm_classifier = BiLSTMConstructivenessClassifier(mode = 'test', model_path = bilstm_model_path)


    def predict_svm(self, example):
        '''
        :param example: str (example comment)
        :return: str (constructiveness prediction for the example)

        Description:
        Given a comment example, example, this class method returns whether the comment
        is constructive or not based on the trained model for constructiveness.
        '''

        # Build a feature vector for the example
        example_df = pd.DataFrame.from_dict({'pp_comment_text': [example], 'constructive':['?']})
        print(example_df)
        fe = FeatureExtractor(example_df)
        fe.extract_features()
        feats_df = fe.get_features_df()

        # Get the prediction score and find the winner
        prediction = self.svm_pipeline.predict(feats_df)[0]
        prediction_winner = 'Non-constructive' if prediction == 0 else 'Constructive'

        return prediction_winner.upper()
    
    def predict_bilstm(self, example):
        '''
        '''
        prediction_winner = self.bilstm_classifier.predict_single_example(example)
        return prediction_winner.upper()
        

if __name__ == "__main__":
    example1 = r'Allowing mercenaries to run the war is a truly frightening development. Contractors should only be used where the US Army truly lacks resources or expertise. If the Afghan government has any sensible people in charge who care for their country, they should vigorously protest the decision to hand the war effort over to mercenaries. This is a sure way to increase the moral hazards a thousand fold, hide war crimes, and increase corruption beyond even the high levels that exist today.'
    example2 = r'This is rubbish!!!'
    predictor = ConstructivenessPredictor()

    prediction = predictor.predict_svm(example1)
    print("Comment: ", example1)
    print('SVM prediction: ', prediction)

    prediction = predictor.predict_bilstm(example1)
    print("Comment: ", example1)
    print('BILSTM prediction: ', prediction)

    prediction = predictor.predict_svm(example2)
    print("Comment: ", example2)
    print('SVM prediction: ', prediction)

    prediction = predictor.predict_bilstm(example2)
    print("Comment: ", example2)
    print('BILSTM prediction: ', prediction)

    
    