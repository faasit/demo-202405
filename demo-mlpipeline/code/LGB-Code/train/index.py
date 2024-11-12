import lightgbm as lgb
import numpy as np
import random
import time
from threading import Thread
from faasit_runtime import function, FaasitRuntime,create_handler
from typing import List, Tuple, Dict, Any
import logging

@function
def train_entry(frt: FaasitRuntime):
    params = frt.input()
    start = time.time()
    ia = params['input']['train_pca_transform']
    oa = params['output']['model']
    param = {
        'feature_fraction': 1.0,
        'max_depth': 8,
        'num_of_trees': 30,
        'chance': 1
    }

    store = frt.storage
    input = store.get(ia, timeout = 5,src_stage='stage0')
    assert(input is not None)

    end_input = time.time()

    time_summary = {}
    
    param['feature_fraction'] = round(random.random() / 2 + 0.5, 1)
    param['chance'] = round(random.random() / 2 + 0.5, 1)
    
    summary = train(frt, param['feature_fraction'], param['max_depth'], 
          param['num_of_trees'], param['chance'], input, oa)

    end = time.time()
        
    return {
        'input_time': end_input - start,
        'compute_time': summary['compute_time'],
        'output_time': summary['output_time'],
        'total_time': end - start, 
        'avg_acc': summary['acc'],
    }

def train(frt: FaasitRuntime, feature_fraction, 
          max_depth, num_of_trees, chance, input, oa) -> Dict[str, Any]:

    start = time.time()
    start_download = time.time()
    end_download = time.time()
    train_data = input

    # we only use 5000 data points for training.
    # every process uses the same 5000 data points.
    Y_train = train_data[0:5000,0]
    X_train = train_data[0:5000,1:train_data.shape[1]]

    # repeat train set for better train effect
    Y_train = np.tile(Y_train, 20)
    X_train = np.tile(X_train, (20, 1))
    
    # chance = round(random.random()/2 + 0.5,1)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_classes' : 10,
        'metric': {'multi_logloss'},
        'num_leaves': 50,
        'learning_rate': 0.05,
        'feature_fraction': feature_fraction,
        'bagging_fraction': chance, # If model indexes are 1->20, this makes feature_fraction: 0.7->0.9
        'bagging_freq': 5,
        'max_depth': max_depth,
        'verbose': -1,
        'num_threads': 2
    }
    
    start_process = time.time()
    lgb_train = lgb.Dataset(X_train, Y_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_of_trees,
                    valid_sets=lgb_train,
                    # early_stopping_rounds=5
                    )
    
    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    count_match=0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_train[i]:
            count_match = count_match + 1
    # The accuracy on the training set  
    acc = count_match/len(y_pred)
    end_process = time.time()
    
    start_upload = time.time()
    store = frt.storage
    store.put(oa, gbm,dest_stages=['stage2'])
    end_upload = time.time()
    end = time.time()

    return {
        'compute_time': end_process - start_process, 
        'output_time': end_upload - start_upload, 
        'acc': acc
    }
handler = create_handler(train_entry)