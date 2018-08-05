__author__      = "Varada Kolhatkar"
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.append('COMMENTIQ_code_subset/')
import commentIQ_features
from spacy_features import CommentLevelFeatures

ROOT = '/Users/vkolhatk/dev/Constructiveness/'

class FeatureExtractor():
    '''
    A feature extractor for feature-based models.
    Extract comment features and write csvs with features
    '''

    def __init__(self, data_csv, comment_column = 'pp_comment_text', label_column='constructive'):
        '''
        '''
        # Read all files
        self.conjuctions_and_connectives = self.file2list(ROOT + 'resources/connectives.txt')
        self.stance_adverbials = self.file2list(ROOT + 'resources/stance_adverbials.txt')
        self.reasoning_verbs = self.file2list(ROOT + 'resources/reasoning_verbs.txt')
        self.root_clauses = self.file2list(ROOT + 'resources/root_clauses.txt')
        self.shell_nouns = self.file2list(ROOT + 'resources/shell_nouns.txt')
        self.modals = self.file2list(ROOT + 'resources/modals.txt')
        self.data_csv = data_csv
        self.df = pd.read_csv(data_csv)
        self.comment_col = comment_column
        self.features_data = []
        self.cols = []

    def file2list(self, file_name):
        '''
        :param file_name: String. Path of a filename
        :return: list

        Description: Given the file_name path, this function returns the values in the file as a list.

        '''
        L = open(file_name).readlines()
        L = [s.strip() for s in L]
        return L

    def has_conjunctions_and_connectives(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.conjuctions_and_connectives):
            #print('Comment with conjunctions and connectives: ', comment)
            return 1

        return 0

    def has_reasoning_verbs(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.reasoning_verbs):
            #print('Comment with reasoning verbs: ', comment)
            return 1

        return 0

    def get_length(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        return len(comment.split())

    def get_average_word_length(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: float
        '''
        return round(np.mean([len(word) for word in comment.split()]),3)

    def has_modals(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.modals):
            #print('Comment with modals: ', comment)
            return 1
        return 0

    def has_shell_nouns(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.shell_nouns):
            #print('Comment with shell nouns: ', comment)
            return 1

        return 0

    def has_stance_adverbials(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(rc in comment for rc in self.stance_adverbials):
            #print('Comment with stance adverbials: ', comment)
            return 1
        return 0


    def has_root_clauses(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(rc in comment for rc in self.root_clauses):
            #print('Comment with root clauses: ', comment)
            return 1
        return 0

    def get_comment_level_features(self, comment):
        '''
        :param comment: String. Comment to extract features.
        :return: int, int, float
        '''
        cf = CommentLevelFeatures(comment)

        # Named entity counts
        ner_count = len(cf.get_named_entities())

        # Number of sentences
        nsentscount = cf.get_sentence_counts()

        # Average nwords per sentence
        anwords = cf.average_nwords_per_sentence()

        return ner_count, nsentscount, anwords

    def extract_features(self, output_csv):
        '''
        :param output_csv: String. The CSV path to write feature vectors
        :return: None

        Description: Given the output CSV file path, output_csv, this function extracts features and writes
        them in output_csv.
        '''

        self.df['has_conjunctions_and_connectives'] = self.df[self.comment_col].apply\
            (self.has_conjunctions_and_connectives)

        self.df['has_stance_adverbials'] = self.df[self.comment_col].apply(
            self.has_stance_adverbials)

        self.df['has_reasoning_verbs'] = self.df[self.comment_col].apply(
            self.has_reasoning_verbs)

        self.df['has_modals'] = self.df[self.comment_col].apply(
            self.has_modals)

        self.df['has_shell_nouns'] = self.df[self.comment_col].apply(
            self.has_shell_nouns)

        self.df['length'] = self.df[self.comment_col].apply(
            self.get_length)

        self.df['average_word_length'] = self.df[self.comment_col].apply(
            self.get_average_word_length)

        self.df['readability_score'] = self.df[self.comment_col].apply(
            commentIQ_features.calcReadability)

        self.df['personal_exp_score'] = self.df[self.comment_col].apply(
            commentIQ_features.calcPersonalXPScores)

        self.df['named_entity_count'], self.df['nSents'], self.df['Avg_words_per_sent'] = \
            zip(*self.df[self.comment_col].apply(self.get_comment_level_features))

        cols = (['pp_comment_text', 'constructive', 'has_conjunctions_and_connectives',
                     'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
                     'length', 'average_word_length', 'readability_score', 'personal_exp_score',
                     'named_entity_count', 'nSents', 'Avg_words_per_sent'])

        self.df.to_csv(output_csv, columns = cols, index = False)
        print('Features CSV written: ', output_csv)

def get_arguments():
    parser = argparse.ArgumentParser(description='SFU Sentiment Calculator')
    parser.add_argument('--train_dataset_path', '-tr', type=str, dest='train_data_path', action='store',
                        default = '/Users/vkolhatk/Data/Constructiveness/data/train/NYT_picks_constructive_YNACC_non_constructive.csv',
                        help="The training data csv")

    parser.add_argument('--test_dataset_path', '-te', type=str, dest='test_data_path', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/test/SOCC_constructiveness.csv',
                        help="The test dataset path for constructive and non-constructive comments")

    parser.add_argument('--train_features_csv', '-trf', type=str, dest='train_features_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/train/features.csv',
                        help="The file containing comments and extracted features for training data")

    parser.add_argument('--test_features_csv', '-tef', type=str, dest='test_features_csv', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/test/features.csv',
                        help="The file containing comments and extracted features for test data")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)

    #fe_train = FeatureExtractor(args.train_data_path)
    #fe_train.extract_features(args.train_features_csv)

    fe_test = FeatureExtractor(args.test_data_path)
    fe_test.extract_features(args.test_features_csv)



