from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    'num_leaves': randint(20, 100), #
    'max_depth': randint(3, 50), #
    'learning_rate': uniform(0.01, 0.3), #
    'n_estimators': randint(100, 1000), #
    'boosting_type': ['gbdt', 'dart', 'goss'], #
}


RANDOM_SEARCH_PARAMS = {
    'n_iter':10,
    'cv':5,
    'verbose':2,
    'random_state':42,
    'scoring':'accuracy',
    'n_jobs':-1
}

