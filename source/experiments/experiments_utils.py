import pandas as pd
import sys
sys.path.append('../modeling/')
from svm_constructiveness_classification import ConstructivenessClassifier
sys.path.append('../../')
from config import Config

def print_data_size(feats_df):
    counts_dict = feats_df['constructive'].value_counts().to_dict()
    counts_dict['Constructive'] = counts_dict.pop(1)
    counts_dict['Non constructive'] = counts_dict.pop(0)        
    print('Size of the training data: ', feats_df.shape[0], 
          '\tConstructive (', counts_dict['Constructive'], ')', 
          '\tNon constructive (', counts_dict['Non constructive'],')'
         )

def wrong_predictions_from_cross_validation(feats_df,
                                           feature_set, 
                                           n = 10):
    svm_classifier = ConstructivenessClassifier(feats_df)    
    return svm_classifier.get_wrong_predictions_from_nfold_cross_validation(feature_set, n = n)
    

def run_training_size_experiments(feats_df, feature_set):
    print_data_size(feats_df)
    svm_classifier = ConstructivenessClassifier(feats_df)
    svm_classifier.create_learning_curves(feature_set)
    

def run_cross_validation_experiments(feats_df, 
                                     feature_set, 
                                     n = 10):  
    
    print_data_size(feats_df)
    svm_classifier = ConstructivenessClassifier(feats_df)
    return svm_classifier.run_nfold_cross_validation(feature_set = feature_set, n = n)


def find_best_feature_subset(feats_df, 
                       feature_set, 
                       n = 10):  
    print_data_size(feats_df)
    svm_classifier = ConstructivenessClassifier(feats_df)
    return svm_classifier.get_best_feature_subset(feature_set = feature_set, n = n)


def grid_search_experiments(feats_df, 
                            feature_set, 
                            n = 8):    
    svm_classifier = ConstructivenessClassifier(feats_df)
    svm_classifier.grid_search(feature_set)
    
def get_corr_df(df, model_cols, annotation_cols):
    model_df = df[model_cols]
    both_df = df[model_cols + annotation_cols]
    corr_df = both_df.corr(method = 'pearson')[annotation_cols]
    return corr_df    
    

if __name__ == '__main__':
    training_feats_file = Config.ALL_FEATURES_FILE_PATH
    training_feats_df = pd.read_csv(training_feats_file)
    SOCC_df = training_feats_df[training_feats_df['source'].isin(['SOCC'])]
    feature_set = ['text_feats', 
                   'length_feats',
                   'argumentation_feats',
                   'COMMENTIQ_feats',
                   'named_entity_feats',
                   'constructiveness_chars_feats',
                   'non_constructiveness_chars_feats',
                   'toxicity_chars_feats',
                   'perspective_content_value_feats',
                   'perspective_aggressiveness_feats',
                   'perspecitive_toxicity_feats'    
                  ]

    #grid_search_experiments(SOCC_df, feature_set)
    #print('The best feature combinations: ', find_best_feature_subset(SOCC_df, feature_set))           
   

