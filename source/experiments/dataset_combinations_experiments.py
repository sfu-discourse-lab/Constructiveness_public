import argparse
import pandas as pd
import sys
sys.path.append('../modeling/')
from svm_constructiveness_classification import ConstructivenessClassifier
sys.path.append('../../')
from config import Config

def get_arguments():
    parser = argparse.ArgumentParser(description='Classify constructive comments')
    parser.add_argument('--train_dataset_path', '-tr', type=str, dest='train_dataset_path', action='store',
                        default= Config.TRAIN_PATH + 'SOCC_nyt_ync_features.csv',
                        help="The input CSV containing features")

    parser.add_argument('--model_path', '-m', type=str, dest='model_file_path', action='store',
                        default= Config.MODEL_PATH + 'svm_model.pkl',
                        help="the path of the file to save the model")

    parser.add_argument('--test_dataset_path', '-te', type=str, dest='test_dataset_path', action='store',
                        default= Config.TEST_PATH + 'features.csv',
                        help="The test dataset path for constructive and non-constructive comments")

    args = parser.parse_args()
    return args


def run_cross_validation_experiments(feats_df, n = 10):

    svm_classifier = ConstructivenessClassifier(feats_df)
    return svm_classifier.run_nfold_cross_validation(n)


def run_data_subset_experiments(training_feats_df):
    '''
    '''        
    data_sources = ['SOCC', 
                    'NYTPicks+YNACC', 
                    'SOCC+NYTPicks+YNACC', 
                    'SOCC+NYTPicks']        
    # Cross validation experiments
    print('\n===============================\n')
    print('Cross validation experiments: ')
    print('\n===============================\n')
    
    for data_source in data_sources:
        print('Data source: ', data_source) 
        sources = data_source.split('+')
        
        if data_source.startswith('SOCC+NYTPicks'):
            # sample negative examples from SOCC and the same number of +ve examples from NYTPicks
            subset_df = training_feats_df[training_feats_df['source'].isin(sources)]
            SOCC_neg_df = subset_df[(subset_df['source'] == 'SOCC') & (subset_df['constructive'] == 0)]
            NYTPicks_df = subset_df[(subset_df['source'] == 'NYTPicks')]                        
            NYTPicks_df_sample = NYTPicks_df.sample(n = SOCC_neg_df.shape[0])
            train_df = pd.concat([SOCC_neg_df, NYTPicks_df_sample])    
        else:             
            train_df = training_feats_df[training_feats_df['source'].isin(sources)]        
            
        print('Size of the training data: ', train_df.shape[0])
        print('Cross-validation results: ', run_cross_validation_experiments(train_df))
        print('\n----------------------------\n')            
        
    
if __name__ == "__main__":
    args = get_arguments()
    print(args)

    # Training data with features csv
    #svm_classifier = ConstructivenessClassifier(args.train_dataset_path, args.test_dataset_path, args.features)
    training_feats_df = pd.read_csv(args.train_dataset_path)
    run_data_subset_experiments(training_feats_df)

    #svm_classifier = ConstructivenessClassifier(args.train_dataset_path, args.test_dataset_path, args.features)
    #svm_classifier.run_nfold_cross_validation(training_feats_df, n=10)
    #svm_classifier.grid_search()
    #svm_classifier.train_classifier(args.model_file_path)
    #svm_classifier.test_classifier()
    #svm_classifier.run_svm_cross_validation()
    #svm_classifier.run_svm_with_csv_features()
    #Results with word count features
    #print('Results with word features:')
    #svm_classifier.run_svm_with_word_count_features()
