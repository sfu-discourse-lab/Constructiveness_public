__author__ = 'VaradaKolhatkar'

import pandas as pd
import argparse
from sklearn.externals import joblib
from feature_extractor import FeatureExtractor

class ConstructivenessPredictor():
    def __init__(self, model_path):
        '''
        :param model_path: str (model path)

        Description: This class assumes that you have a feature-based trained model. It returns the prediction of the
        given example based on the trained model.
        '''
        self.model_path = model_path
        self.fe = FeatureExtractor(None)

        # load model
        self.pipeline = joblib.load(model_path)
        self.cols = (['pp_comment_text', 'constructive', 'Has_conjunction_or_connectives',
                      'Has_stance_adverbials',
                      'Has_reasoning_verbs', 'Has_modals', 'Has_shell_nouns', 'Len', 'Average_word_length',
                      'Redability', 'PersonalEXP', 'Named_entity_count', 'nSents', 'Avg_words_per_sent'])
        
        
    def predict_svm(self, example):
        '''
        :param example: str (example comment)
        :return: str (constructiveness prediction for the example)

        Description:
        Given a comment example, example, this class method returns whether the comment
        is constructive or not based on the trained model for constructiveness.
        '''

        # Build a feature vector for the example
        feat_vector = self.fe.extract_feature_vector_for_unknow_example(example)

        # read the feature vector as a dataframe and select relevant columns from the dataframe
        df = pd.DataFrame([feat_vector], columns=self.cols)

        # Get the prediction score and find the winner
        prediction = self.pipeline.predict(df)[0]
        prediction_winner = 'Non-constructive' if prediction == 0 else 'Constructive'

        return prediction_winner.upper()

def get_arguments():
    parser = argparse.ArgumentParser(description='Classify constructive comments')

    parser.add_argument('--model_path', '-m', type=str, dest='model_path', action='store',
                        default= '/Users/vkolhatk/Data/Constructiveness/results/svm_results/models/model.pkl',
                        help="the input dir containing data")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    example1 = r'Allowing mercenaries to run the war is a truly frightening development. Contractors should only be used where the US Army truly lacks resources or expertise. If the Afghan government has any sensible people in charge who care for their country, they should vigorously protest the decision to hand the war effort over to mercenaries. This is a sure way to increase the moral hazards a thousand fold, hide war crimes, and increase corruption beyond even the high levels that exist today.'

    csvm = ConstructivenessPredictor(args.model_path)

    prediction = csvm.predict_svm(example1)
    print("Comment: ", example1)
    print('Prediction: ', prediction)