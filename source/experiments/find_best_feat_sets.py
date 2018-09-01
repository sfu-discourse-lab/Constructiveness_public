import pandas as pd
import sys
sys.path.append('../../')
from config import Config
from experiments_utils import *

training_feats_file = Config.ALL_FEATURES_FILE_PATH
training_feats_df = pd.read_csv(training_feats_file)
SOCC_df = training_feats_df[training_feats_df['source'].isin(['SOCC'])]

print('The best feature combinations: ', find_best_feature_subset(SOCC_df, Config.FEATURE_SETS))           
