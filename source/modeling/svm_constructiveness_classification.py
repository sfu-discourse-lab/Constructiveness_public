from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn import metrics
import numpy as np
import pandas as pd
import argparse, glob, codecs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from feature_pipelines import build_feature_pipelines_and_unions

from itertools import chain, combinations

class ConstructivenessClassifier():
    '''

    '''
    def __init__(self, df_train, features = None, df_test = None,
                 comments_col = 'pp_comment_text',
                 target_col = 'constructive'):
        '''
        :param train_features_csv: (str) The CSV path containing training data features
        :param test_features_csv: (str) The CSV path containing test data features
        :param features: (list) The features to consider in the model
        :param comments_col: (str) The name of the column containing comments text
        :param target_col: (str) The name of the column containing the target variable
        :param classifier: (str) The name of the sklearn classifier
        '''

        #self.df_train = pd.read_csv(train_features_csv, header=0)
        #self.df_test = pd.read_csv(test_features_csv, header=0)

        self.df_train = df_train
        self.df_test = df_test

        self.comments_col = comments_col

        self.features = features
        self.pipeline = None

        self.X1 = self.df_train[self.df_train.columns.drop(target_col)]
        #print('CSV columns: ', self.X1.columns)
        self.y1 = self.df_train[target_col]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X1, self.y1, test_size=0.9, random_state=0)

        #self.X_test = self.df_test[self.df_test.columns.drop(target_col)]
        #self.y_test = self.df_test[target_col]
        # Best parameters found
        # {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
        self.tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        self.scores = ['precision', 'recall']
        self.CV_P_R = []

    def grid_search(self, feature_set):
        '''
        :return:
        '''
        feats = build_feature_pipelines_and_unions(feature_set)
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(),
                               self.tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            #clf.fit(self.X_train, self.y_train)

            pipeline = Pipeline([
                ('features', feats),
                ('classifier', clf),
            ])

            pipeline.fit(self.X1, self.y1)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self.y_valid, pipeline.predict(self.X_valid)
            print(classification_report(y_true, y_pred))
            print()

    def build_classifier_pipeline(self, feature_set, 
                                  classifier = SGDClassifier(loss='hinge', 
                                                             penalty='l2', 
                                                             alpha=1e-3, 
                                                             max_iter=50, 
                                                             tol=1e-4, 
                                                             random_state=42)):
                                  #SVC(C=100, gamma=0.001, kernel='rbf')):
        '''
        :param feature_set:
        :param classifier:
        :return:
        '''
        print('Classifier: ', classifier)
        print('Feature set: ', feature_set)
        feats = build_feature_pipelines_and_unions(feature_set)
        pipeline = Pipeline([
            ('features',feats),
            ('classifier', classifier),
        ])
        return pipeline

    
    def train_classifier(self,
                         model_path,
                         #classifier,
                         feature_set):
        '''
        :return:
        '''
        self.pipeline = self.build_classifier_pipeline(feature_set = feature_set)
        self.pipeline.fit(self.X1, self.y1)
        joblib.dump(self.pipeline, model_path)
        s = pickle.dumps(self.pipeline)
        print('Model trained and pickled in file: ', model_path)

        
    def get_wrong_predictions_from_nfold_cross_validation(self, feature_set, n=5):
        '''
        :param n:
        :return:
        '''
        pipeline = self.build_classifier_pipeline(feature_set)
        predicted = cross_val_predict(pipeline, self.X1, self.y1, cv=n)
        return predicted, self.X1, self.y1 
        
        
    def run_nfold_cross_validation(self, feature_set, n=5, scoring = 'f1'):
        '''
        :param n:
        :return:
        '''
        print('Cross validation folds: ', n)
        pipeline = self.build_classifier_pipeline(feature_set)
        scores = cross_val_score(pipeline, self.X1, self.y1, cv=n, scoring=scoring)        
        results = {}
        results['scores'] = scores
        results['mean_score'] = np.mean(scores)
        results['variance'] = np.var(scores)
        return results 
    

    def get_best_feature_subset(self, feature_set, n=8):                
        print('Cross validation folds: ', n)
        n_features = len(feature_set)
        subsets = (combinations(feature_set, k) for k in range(1,len(feature_set)))
        
        best_score = -np.inf
        best_subset = None
        
        for subset in subsets:
            for ss in subset: 
                results = self.run_nfold_cross_validation(feature_set = ss)
                #print('Subset: ', ss)
                print('Results: ')
                for (key, val) in results.items():
                    print(key, ' : ', val)
                print('---------\n')
                if results['mean_score'] > best_score:
                    best_score, best_subset = results['mean_score'], ss

        return best_subset, best_score
    
    
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt


    def create_learning_curves(self, feature_set):
        estimator = self.build_classifier_pipeline(feature_set = feature_set)
        title = "Learning Curves with classifier: "

        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        print('plot learning curves:  ')
        self.plot_learning_curve(estimator, title, self.X1, self.y1, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        print('Show the plots')
        plt.show()        
        

if __name__=="__main__":
    import sys
    sys.path.append('../../')
    from config import Config
    training_feats_file = Config.FEATURES_FILE_PATH #'SOCC_nyt_ync_features.csv'
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
                   'perspective_aggressiveness_feats'
                   'perspecitive_toxicity_feats'    
               ]
    
    svm_classifier = ConstructivenessClassifier(SOCC_df)
    svm_classifier.train_classifier(model_path = Config.MODEL_PATH + 'svm_model_new.pkl', feature_set = feature_set)
    
