class Config:
    PROJECT_NAME = 'Constructiveness'
    NYT_API_KEY = 'XXX'
    
    HOME = '/home/vkolhatk/'
    PROJECT_HOME = HOME + 'dev/Constructiveness_public/'
    DATA_HOME = HOME + 'data/Constructiveness_public/'
    RESOURCES_HOME = PROJECT_HOME + 'resources/'
    
    # SOCC related paths 
    SOCC = HOME + '/data/SOCC/'
    SOCC_RAW = HOME + 'data/SOCC/raw/'
    SOCC_COMMENTS_PATH = SOCC_RAW + 'gnm_comments.csv'
    SOCC_ARTICLES_PATH = SOCC_RAW + 'gnm_articles.csv'
    SOCC_COMMENT_THREADS_PATH = SOCC_RAW + 'gnm_comment_threads.csv'
    
    SOCC_ANNOTATED_PATH = DATA_HOME + 'SOCC/'    
    SOCC_ANNOTATED_CONSTRUCTIVENESS_1000 = SOCC_ANNOTATED_PATH + 'SFU_constructiveness_toxicity_corpus.csv'
    SOCC_ANNOTATED_CONSTRUCTIVENESS_12000 = SOCC_ANNOTATED_PATH + 'constructiveness_and_toxicity_annotations_batches_1_to_12.csv'
    SOCC_ANNOTATED_WITH_PERSPECTIVE_SCORES = SOCC_ANNOTATED_PATH + 'annotated_SOCC_with_perspective_scores_revised.csv'
    SOCC_COMMENTS_WITH_PERSPECTIVE_SCORES = SOCC_ANNOTATED_PATH + 'gnm_comments_scored_2.csv'
        
    # NYT PICKS paths
    NYT_PICKS_PATH = DATA_HOME + 'NYT_comments_csvs/'
    NYT_PICKS_SFU = NYT_PICKS_PATH + 'NYT_comments_all_June_6_editor_picks.csv'
    NYT_PICKS_COMMENTIQ = NYT_PICKS_PATH + 'editorsselections-10-2014.csv'
    
    # Yahoo News Annotated Comments Corpus paths
    YNACC_PATH = DATA_HOME + 'ydata-ynacc-v1_0/'    
    YNACC_EXPERT_ANNOTATIONS = YNACC_PATH + 'ydata-ynacc-v1_0_expert_annotations.tsv'
    YNACC_MTURK_ANNOTATIONS = YNACC_PATH + 'ydata-ynacc-v1_0_turk_annotations.tsv'
    
    # Training data files
    TRAIN_PATH = DATA_HOME + 'train/'
    #ALL_FEATURES_FILE_PATH = TRAIN_PATH + 'old_SOCC_SOCC_nyt_ync_all_features.csv'   
    ALL_DATASETS_ALL_FEATURES_FILE_PATH = TRAIN_PATH + 'all_datasets_all_feats.csv'
    ALL_FEATURES_FILE_PATH = TRAIN_PATH + 'socc_both_nyt_ynacc_feats.csv'
    ALL_SOCC_FEATURES_FILE_PATH = TRAIN_PATH + 'SOCC_all_feats.csv'
    CTC_FEATURES_FILE_PATH = TRAIN_PATH + 'CTC_with_all_feats.csv'
    #'all_SOCC_constructiveness_feats.csv'
    
    # Model paths
    MODEL_PATH = DATA_HOME + 'models/'
    SVM_MODEL_PATH = MODEL_PATH + 'svm_model_new.pkl'
    BILSTM_MODEL_PATH = MODEL_PATH + 'SOCC_bilstm.tflearn'

    # Feature sets
    ALL_FEATURES = ['ngram_feats',
                    'tfidf_feats',
                    'pos_feats',
                    # constructiveness_chars_feats
                    'specific_points',
                    'dialogue',
                    'no_con',
                    'evidence',
                    'personal_story',
                    'solution',
                    # non_constructiveness_chars_feats                       
                    'no_respect',
                    'no_non_con',
                    'provocative',
                    'sarcastic',
                    'non_relevant',
                    'unsubstantial',
                    # toxicity_chars_feats
                    'personal_attack',
                    'teasing',
                    'no_toxic',
                    'abusive',
                    'embarrassment',
                    'inflammatory',
                    # argumentation_feats
                    'has_conjunctions_and_connectives',
                    'has_stance_adverbials',
                    'has_reasoning_verbs',
                    'has_modals',
                    'has_shell_nouns',
                    # length_feats
                    'length',
                    'average_word_length',
                    'nSents',
                    'avg_words_per_sent',
                    # COMMENTIQ_feats
                    'readability_score',
                    'personal_exp_score',
                    # named_entity_feats
                    'named_entity_count',
                    # perspecitive_toxicity_feats                        
                    'SEVERE_TOXICITY:probability',
                    'SEXUALLY_EXPLICIT:probability',
                    'TOXICITY:probability', 
                    'TOXICITY_IDENTITY_HATE:probability',
                    'TOXICITY_INSULT:probability',
                    'TOXICITY_OBSCENE:probability',
                    'TOXICITY_THREAT:probability', 
                    'INFLAMMATORY:probability',
                    'LIKELY_TO_REJECT:probability',
                    # perspective_aggressiveness_feats                                                
                    'ATTACK_ON_AUTHOR:probability',
                    'ATTACK_ON_COMMENTER:probability', 
                    'ATTACK_ON_PUBLISHER:probability',
                    # perspective_content_value_feats
                    'INCOHERENT:probability',                        
                    'OFF_TOPIC:probability',
                    'SPAM:probability',
                    'UNSUBSTANTIAL:probability']    
    
    FEATURE_SETS = ['ngram_feats', 
                    'tfidf_feats',
                    'pos_feats',
                    'length_feats',
                    'argumentation_feats',
                    'text_quality_feats',
                    'named_entity_feats',
                    #'constructiveness_chars_feats',
                    #'non_constructiveness_chars_feats',
                    #'toxicity_chars_feats',
                    'perspective_content_value_feats',
                    'perspective_aggressiveness_feats',
                    'perspecitive_toxicity_feats'
                   ]
    
    # Results paths
    RESULTS_PATH = DATA_HOME + 'results/'
    FIGURES_PATH = RESULTS_PATH + 'figures/'
    
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
    
    
    
    