from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from my_transformers import TextSelector, NumberSelector
from sklearn.preprocessing import StandardScaler
NGRAM_MIN = 1
NGRAM_MAX = 4

def text_feats_pipeline():
    '''
    :return:
    '''
    # Create feature pipelines and then feature unions
    # ngram and tf-idf features
    text = Pipeline([
        ('selector', TextSelector(key='pp_comment_text')),
        ('ngrams', CountVectorizer(ngram_range=(NGRAM_MIN,NGRAM_MAX))),
        ('tfidf', TfidfTransformer())
        # ('tfidf', TfidfVectorizer(stop_words='english'))
    ])
    return text

def length_feats_pipeline():
    '''
    #:return:
    '''
    # Length features
    length = Pipeline([
        ('selector', NumberSelector(key='length')),
        ('standard', StandardScaler())
    ])

    average_word_length = Pipeline([
        ('selector', NumberSelector(key='average_word_length')),
        ('standard', StandardScaler())
    ])

    nSents = Pipeline([
        ('selector', NumberSelector(key='nSents')),
        ('standard', StandardScaler())
    ])

    avg_words_per_sent = Pipeline([
        ('selector', NumberSelector(key='avg_words_per_sent')),
        ('standard', StandardScaler())
    ])

    length_feats = FeatureUnion([
        ('length', length),
        ('average_word_length', average_word_length),
        ('nSents', nSents),
        ('avg_words_per_sent', avg_words_per_sent)
    ])

    return length_feats

def argumentation_feats_pipeline():
    '''
    #:return:
    '''
    has_conjunctions_and_connectives = Pipeline([
        ('selector', NumberSelector(key='has_conjunctions_and_connectives')),
        ('standard', StandardScaler())
    ])

    has_stance_adverbials = Pipeline([
        ('selector', NumberSelector(key='has_stance_adverbials')),
        ('standard', StandardScaler())
    ])

    has_reasoning_verbs = Pipeline([
        ('selector', NumberSelector(key='has_reasoning_verbs')),
        ('standard', StandardScaler())
    ])

    has_modals = Pipeline([
        ('selector', NumberSelector(key='has_modals')),
        ('standard', StandardScaler())
    ])

    has_shell_nouns = Pipeline([
        ('selector', NumberSelector(key='has_shell_nouns')),
        ('standard', StandardScaler())
    ])
    argumentation_feats = FeatureUnion([
            ('has_conjunctions_and_connectives', has_conjunctions_and_connectives),
            ('has_stance_adverbials', has_stance_adverbials),
            ('has_reasoning_verbs', has_reasoning_verbs),
            ('has_modals', has_modals),
            ('has_shell_nouns', has_shell_nouns)
            ])
    return argumentation_feats

def COMMENTIQ_feats_pipeline():
    '''
    :return:
    '''
    readability_score = Pipeline([
        ('selector', NumberSelector(key='readability_score')),
        ('standard', StandardScaler())
    ])

    personal_exp_score = Pipeline([
        ('selector', NumberSelector(key='personal_exp_score')),
        ('standard', StandardScaler())
    ])
    COMMENTIQ_feats = FeatureUnion([
        ('readability_score', readability_score),
        ('personal_exp_score', personal_exp_score),
    ])
    return COMMENTIQ_feats

def named_entity_feats_pipeline():
    '''
    :return:
    '''
    named_entity_count = Pipeline([
        ('selector', NumberSelector(key='named_entity_count')),
        ('standard', StandardScaler())
    ])

    return named_entity_count

def constructiveness_chars_feats_pipeline():
    '''
    :return:
    '''

    specific_points = Pipeline([
        ('selector', NumberSelector(key='specific_points')),
        ('standard', StandardScaler())
    ])

    dialogue = Pipeline([
        ('selector', NumberSelector(key='dialogue')),
        ('standard', StandardScaler())
    ])

    no_con = Pipeline([
        ('selector', NumberSelector(key='no_con')),
        ('standard', StandardScaler())
    ])

    evidence = Pipeline([
        ('selector', NumberSelector(key='evidence')),
        ('standard', StandardScaler())
    ])

    personal_story = Pipeline([
        ('selector', NumberSelector(key='personal_story')),
        ('standard', StandardScaler())
    ])

    solution = Pipeline([
        ('selector', NumberSelector(key='solution')),
        ('standard', StandardScaler())
    ])

    constructiveness_chars_feats = FeatureUnion([
        ('specific_points', specific_points),
        ('dialogue', dialogue),
        ('no_con', no_con),
        ('evidence', evidence),
        ('personal_story', personal_story),
        ('solution', solution)
    ])

    return constructiveness_chars_feats

def non_constructiveness_chars_feats_pipeline():
    '''
    :return:
    '''
    no_respect = Pipeline([
        ('selector', NumberSelector(key='no_respect')),
        ('standard', StandardScaler())
    ])

    no_non_con = Pipeline([
        ('selector', NumberSelector(key='no_non_con')),
        ('standard', StandardScaler())
    ])

    sarcastic = Pipeline([
        ('selector', NumberSelector(key='sarcastic')),
        ('standard', StandardScaler())
    ])

    non_relevant = Pipeline([
        ('selector', NumberSelector(key='non_relevant')),
        ('standard', StandardScaler())
    ])

    unsubstantial = Pipeline([
        ('selector', NumberSelector(key='unsubstantial')),
        ('standard', StandardScaler())
    ])
    non_constructiveness_chars_feats = FeatureUnion([
        ('no_respect', no_respect),
        ('no_non_con', no_non_con),
        ('sarcastic', sarcastic),
        ('non_relevant', non_relevant),
        ('unsubstantial', unsubstantial)
    ])

    return non_constructiveness_chars_feats

def toxicity_chars_feats_pipeline():
    '''
    :return:
    '''
    personal_attack = Pipeline([
        ('selector', NumberSelector(key='personal_attack')),
        ('standard', StandardScaler())
    ])

    teasing = Pipeline([
        ('selector', NumberSelector(key='teasing')),
        ('standard', StandardScaler())
    ])

    no_toxic = Pipeline([
        ('selector', NumberSelector(key='no_toxic')),
        ('standard', StandardScaler())
    ])

    abusive = Pipeline([
        ('selector', NumberSelector(key='abusive')),
        ('standard', StandardScaler())
    ])

    embarrassment = Pipeline([
        ('selector', NumberSelector(key='embarrassment')),
        ('standard', StandardScaler())
    ])

    inflammatory = Pipeline([
        ('selector', NumberSelector(key='inflammatory')),
        ('standard', StandardScaler())
    ])

    toxicity_chars_feats = FeatureUnion([
        ('personal_attack', personal_attack),
        ('teasing', teasing),
        ('no_toxic', no_toxic),
        ('abusive', abusive),
        ('embarrassment', embarrassment),
        ('inflammatory', inflammatory)
    ])
    return toxicity_chars_feats

def perspective_content_value_feats_pipeline():
    '''
    :return:
    '''
    INCOHERENT_probability = Pipeline([
        ('selector', NumberSelector(key='INCOHERENT:probability')),
        ('standard', StandardScaler())
    ])

    #OFF_TOPIC_probability = Pipeline([
    #    ('selector', NumberSelector(key='OFF_TOPIC_probability')),
    #    ('standard', StandardScaler())
    #])

    SPAM_probability = Pipeline([
        ('selector', NumberSelector(key='SPAM:probability')),
        ('standard', StandardScaler())
    ])

    UNSUBSTANTIAL_probability = Pipeline([
        ('selector', NumberSelector(key='UNSUBSTANTIAL:probability')),
        ('standard', StandardScaler())
    ])
    
    #LIKELY_TO_REJECT_probability = Pipeline([
    #    ('selector', NumberSelector(key='LIKELY_TO_REJECT:probability')),
    #    ('standard', StandardScaler())
    #])
    
    perspective_content_value_feats = FeatureUnion([
        ('INCOHERENT_probability', INCOHERENT_probability),
        ('SPAM_probability', SPAM_probability),
        ('UNSUBSTANTIAL_probability', UNSUBSTANTIAL_probability)
    ])
    return perspective_content_value_feats


def perspective_aggressiveness_feats_pipeline():
    '''
    '''
    ATTACK_ON_AUTHOR_probability = Pipeline([
        ('selector', NumberSelector(key='ATTACK_ON_AUTHOR:probability')),
        ('standard', StandardScaler())
    ])

    ATTACK_ON_COMMENTER_probability = Pipeline([
        ('selector', NumberSelector(key='ATTACK_ON_COMMENTER:probability')),
        ('standard', StandardScaler())
    ])

    ATTACK_ON_PUBLISHER_probability = Pipeline([
        ('selector', NumberSelector(key='ATTACK_ON_PUBLISHER:probability')),
        ('standard', StandardScaler())
    ])
    
    perspective_aggressiveness_feats = FeatureUnion([
        ('ATTACK_ON_AUTHOR_probability', ATTACK_ON_AUTHOR_probability),
        ('ATTACK_ON_COMMENTER_probability', ATTACK_ON_COMMENTER_probability),    
        ('ATTACK_ON_PUBLISHER_probability', ATTACK_ON_PUBLISHER_probability)                
    ])
    return perspective_aggressiveness_feats

def perspecitive_toxicity_feats_pipeline():
    '''
    '''
    SEVERE_TOXICITY_probability = Pipeline([
        ('selector', NumberSelector(key='SEVERE_TOXICITY:probability')),
        ('standard', StandardScaler())
    ])

    SEXUALLY_EXPLICIT_probability = Pipeline([
        ('selector', NumberSelector(key='SEXUALLY_EXPLICIT:probability')),
        ('standard', StandardScaler())
    ])

    TOXICITY_probability = Pipeline([
        ('selector', NumberSelector(key='TOXICITY:probability')),
        ('standard', StandardScaler())
    ])

    TOXICITY_IDENTITY_HATE_probability = Pipeline([
        ('selector', NumberSelector(key='TOXICITY_IDENTITY_HATE:probability')),
        ('standard', StandardScaler())
    ])

    TOXICITY_INSULT_probability = Pipeline([
        ('selector', NumberSelector(key='TOXICITY_INSULT:probability')),
        ('standard', StandardScaler())
    ])

    TOXICITY_OBSCENE_probability = Pipeline([
        ('selector', NumberSelector(key='TOXICITY_OBSCENE:probability')),
        ('standard', StandardScaler())
    ])

    TOXICITY_THREAT_probability = Pipeline([
        ('selector', NumberSelector(key='TOXICITY_THREAT:probability')),
        ('standard', StandardScaler())
    ])

    INFLAMMATORY_probability = Pipeline([
        ('selector', NumberSelector(key='INFLAMMATORY:probability')),
        ('standard', StandardScaler())
    ])

    OBSCENE_probability = Pipeline([
        ('selector', NumberSelector(key='OBSCENE:probability')),
        ('standard', StandardScaler())
    ])
    
    perspecitive_toxicity_feats = FeatureUnion([
        ('SEVERE_TOXICITY_probability', SEVERE_TOXICITY_probability),
        ('SEXUALLY_EXPLICIT_probability', SEXUALLY_EXPLICIT_probability),    
        ('TOXICITY_probability', TOXICITY_probability),
        ('TOXICITY_IDENTITY_HATE_probability', TOXICITY_IDENTITY_HATE_probability),
        ('TOXICITY_INSULT_probability', TOXICITY_INSULT_probability),
        ('TOXICITY_OBSCENE_probability', TOXICITY_OBSCENE_probability),
        ('TOXICITY_THREAT_probability', TOXICITY_THREAT_probability),
        ('INFLAMMATORY_probability', INFLAMMATORY_probability),
        ('OBSCENE_probability', OBSCENE_probability)        
    ])
    return perspecitive_toxicity_feats


def build_feature_pipelines_and_unions(feature_set = ['text_feats', 'length_feats',
                                                     'argumentation_feats',
                                                     'COMMENTIQ_feats',
                                                     'named_entity_feats',
                                                     'constructiveness_chars_feats',
                                                     'non_constructiveness_chars_feats',
                                                     'toxicity_chars_feats', 
                                                     'perspective_content_value_feats', 
                                                     'perspective_aggressiveness_feats',
                                                     'perspecitive_toxicity_feats']):
    '''
    :return: re
    '''
    text_feats = text_feats_pipeline()
    length_feats = length_feats_pipeline()
    argumentation_feats = argumentation_feats_pipeline()
    COMMENTIQ_feats = COMMENTIQ_feats_pipeline()
    named_entity_feats = named_entity_feats_pipeline()
    constructiveness_chars_feats = constructiveness_chars_feats_pipeline()
    non_constructiveness_chars_feats = non_constructiveness_chars_feats_pipeline()
    toxicity_chars_feats = toxicity_chars_feats_pipeline()
    perspective_content_value_feats = perspective_content_value_feats_pipeline()
    perspective_aggressiveness_feats = perspective_aggressiveness_feats_pipeline()
    perspecitive_toxicity_feats = perspecitive_toxicity_feats_pipeline()
    
    feat_sets_dict = {'text_feats': text_feats,
                      'length_feats': length_feats,
                      'argumentation_feats': argumentation_feats,
                      'COMMENTIQ_feats': COMMENTIQ_feats,
                      'named_entity_feats': named_entity_feats,
                      'constructiveness_chars_feats': constructiveness_chars_feats,
                      'non_constructiveness_chars_feats': non_constructiveness_chars_feats,
                      'toxicity_chars_feats': toxicity_chars_feats,
                      'perspective_content_value_feats': perspective_content_value_feats,
                      'perspective_aggressiveness_feats': perspective_aggressiveness_feats, 
                      'perspecitive_toxicity_feats': perspecitive_toxicity_feats
                     }
    
    feat_tuples = [(feat, feat_sets_dict[feat]) for feat in feature_set]
    feats = FeatureUnion(feat_tuples)

    return feats

if __name__ == "__main__":
    feats = build_feature_pipelines_and_unions()
    print(feats)
