class Config:
    PROJECT_NAME = 'Constructiveness'
    NYT_API_KEY = 'XXX'
    
    HOME = '/home/vkolhatk/'
    PROJECT_HOME = HOME + 'dev/Constructiveness_public/'
    DATA_HOME = HOME + 'data/Constructiveness_public/'
    RESOURCES_HOME = PROJECT_HOME + 'resources/'
    
    # SOCC related paths 
    SOCC_ANNOTATED_PATH = DATA_HOME + 'SOCC/'
    SOCC_ANNOTATED_CONSTRUCTIVENESS_1000 = SOCC_ANNOTATED_PATH + 'SFU_constructiveness_toxicity_corpus.csv'
    SOCC_ANNOTATED_CONSTRUCTIVENESS_12000 = SOCC_ANNOTATED_PATH + 'constructiveness_and_toxicity_annotations_batches_1_to_12.csv'
    SOCC_ANNOTATED_WITH_PERSPECTIVE_SCORES = SOCC_ANNOTATED_PATH + 'annotated_SOCC_with_perspective_scores_corrected.csv'

        
    # NYT PICKS paths
    NYT_PICKS_PATH = DATA_HOME + 'NYT_comments_csvs/'
    NYT_PICKS_SFU = NYT_PICKS_PATH + 'NYT_comments_all_June_6_editor_picks.csv'
    NYT_PICKS_COMMENTIQ = NYT_PICKS_PATH + 'editorsselections-10-2014.csv'
    
    # Yahoo News Annotated Comments Corpus paths
    YNACC_PATH = DATA_HOME + 'ydata-ynacc-v1_0/'    
    YNACC_EXPERT_ANNOTATIONS = YNACC_PATH + 'ydata-ynacc-v1_0_expert_annotations.tsv'
    YNACC_MTURK_ANNOTATIONS = YNACC_PATH + 'ydata-ynacc-v1_0_turk_annotations.tsv'
    
    # Train data files
    TRAIN_PATH = DATA_HOME + 'train/'        
    
    # Model paths
    MODEL_PATH = DATA_HOME + 'models/'
    SVM_MODEL_PATH = MODEL_PATH + 'svm_model_new.pkl'
    BILSTM_MODEL_PATH = MODEL_PATH + 'SOCC_bilstm.tflearn'
    
    # Test data files
    TEST_PATH = DATA_HOME + 'test/'

    # Glove path
    GLOVE_DICTIONARY_PATH = HOME + 'data/glove/glove.840B.vocab'
    GLOVE_EMBEDDINGS_PATH = HOME + 'data/glove/glove.pickle'
    
    # Pickle paths
    SOCC_PICKLE_PATH = TRAIN_PATH + ''
    
    # Web interface settings
    PORT = 9898
    HOST = 'localhost'
    FEEDBACK_CSV_DIR = DATA_HOME + 'feedback/'
    FEEDBACK_CSV_PATH = FEEDBACK_CSV_DIR + 'feedback.csv'
    
    
    
    