import numpy as np
import lightgbm as lgb
import os
import time
from utils import MergedLGBMClassifier, get_files
from typing import Dict, List, Any, Optional
import logging
from faasit_runtime import FaasitRuntime, function

@function
def aggregate_models(frt: FaasitRuntime):
    start = time.time()
    _input = frt.input()
    
    input: Dict[str, str] = _input['input']
    output: Dict[str, str] = _input['output']
    
    logging.debug(f"input: {input}")
    assert len(input) == 2
    assert len(output) == 1

    start_download = time.time()
    store = frt.storage
    test_data = store.get(input['train_pca_transform'], timeout = 5, src_state='stage0')
    input_model = store.get(input['model'], timeout = 5, src_state='stage1')
    assert(test_data is not None and input_model is not None)
    end_download = time.time()

    Y_test = test_data[5000:,0]
    X_test = test_data[5000:,1:test_data.shape[1]]
    
    # Merge models
    start_process = time.time()
    forest = MergedLGBMClassifier([input_model])
    
    # Predict
    y_pred = forest.predict(X_test)
    count_match=0
    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_test[i]:
            count_match = count_match + 1

    # The accuracy on the training set  
    acc = count_match / len(y_pred)
    end_process = time.time()
    
    start_upload = time.time()
    store.put(output['predict'], y_pred,dest_state=['stage3'])
    end_upload = time.time()
    
    end = time.time()
    
    return frt.output({
        'input_time': end_download - start_download, 
        'compute_time': end_process - start_process, 
        'output_time': end_upload - start_upload, 
        'total_time': end - start, 
        'acc': acc
    })
