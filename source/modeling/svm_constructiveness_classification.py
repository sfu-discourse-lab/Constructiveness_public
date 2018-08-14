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

from sklearn import metrics
import numpy as np
import pandas as pd
import argparse, glob, codecs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from feature_pipelines import build_feature_pipelines_and_unions


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
            self.X1, self.y1, test_size=0.2, random_state=0)

        #self.X_test = self.df_test[self.df_test.columns.drop(target_col)]
        #self.y_test = self.df_test[target_col]
        # Best parameters found
        # {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
        self.tuned_parameters = [#{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             #'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        self.scores = ['precision', 'recall']
        self.CV_P_R = []

    def grid_search(self):
        '''
        :return:
        '''
        feats = build_feature_pipelines_and_unions()
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(),
                               self.tuned_parameters, cv=3,
                               scoring='%s_macro' % score)
            #clf.fit(self.X_train, self.y_train)

            pipeline = Pipeline([
                ('features', feats),
                ('classifier', clf),
            ])

            pipeline.fit(self.X_train, self.y_train)

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

    def build_classifier_pipeline(self,
                                  classifier = SGDClassifier(loss='hinge', 
                                                             penalty='l2', 
                                                             alpha=1e-3, 
                                                             max_iter=20, 
                                                             tol=1e-4, 
                                                             random_state=42),
                                  
                                  #=SVC(C=1000, gamma=0.001, kernel='rbf')
                                  feature_set = []):
        '''
        :param feature_set:
        :param classifier:
        :return:
        '''
        feats = build_feature_pipelines_and_unions()
        pipeline = Pipeline([
            ('features',feats),
            #('feature_selection', SelectFromModel(SVC(kernel = 'linear'))),
            #('feature_selection', RFE(estimator=SVC(kernel='linear'), n_features_to_select=1, step=1)),
            #('feature_selection', RFECV(estimator=SVC(kernel='linear'), step=1, cv=StratifiedKFold(2),
            #  scoring='accuracy')),
            #('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
            ('classifier', classifier),
        ])
        return pipeline

    def train_classifier(self,
                         model_path,
                         #classifier,
                         feature_set = []):
        '''
        :return:
        '''
        self.pipeline = self.build_classifier_pipeline()
        self.pipeline.fit(self.X1, self.y1)
        joblib.dump(self.pipeline, model_path)
        s = pickle.dumps(self.pipeline)
        print('Model trained and pickled in file: ', model_path)
        #svm_predictor = pickle.loads(s)
        #preds = svm_predictor.predict(self.X_test)
        #print('Mean accuracy: ', np.mean(preds == self.y_test))
        #print('SVM correct prediction: {:4.2f}'.format(np.mean(preds == self.y_test)))
        #print(metrics.classification_report(self.y_test, preds, target_names=['non-constructive', 'constructive']))#

        
    def run_nfold_cross_validation(self, n=5):
        '''
        :param n:
        :return:
        '''
        pipeline = self.build_classifier_pipeline()
        scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=n, scoring='f1_micro')
        results = {}
        results['scores'] = scores
        results['mean_score'] = np.mean(scores)
        results['variance'] = np.var(scores)
        return results 
    

    def test_classifier(self,
                        model_path=''):
        '''
        :return:
        '''
        te_targets = self.df_test['constructive'].tolist()
        predicted = self.pipeline.predict(self.df_test)
        print(predicted)

        print('SVM correct prediction: {:4.2f}'.format(np.mean(predicted == te_targets)))
        print(metrics.classification_report(te_targets, predicted, target_names=['non-constructive', 'constructive']))

        
    def plot_coefficients(self, classifier, feature_names, top_features=20):
        import matplotlib.pyplot as plt
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()



