import numpy as np
import time
from utils import get_files
import logging
from faasit_runtime import FaasitRuntime, function
from typing import Dict, List, Any, Optional

@function
def test_forests(frt: FaasitRuntime):
    start = time.time()
    
    start_download = time.time()
    _input = frt.input()
    input = _input['input']
    assert len(input) == 2
    
    store = frt.storage
    test_data = store.get(input['train_pca_transform'], timeout = 5, src_state='stage0')
    pred: Optional[np.ndarray] = store.get(input['predict'], timeout = 5, src_state='state2')
    assert(test_data is not None and pred is not None)
    preds = [pred]

    end_download = time.time()
    
    Y_test = test_data[5000:,0]
    
    end_download = time.time()
    
    start_process = time.time()
    y_pred: np.ndarray = sum(preds) / len(preds) # type: ignore
    count_match=0

    for i in range(len(y_pred)):
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        if result == Y_test[i]:
            count_match = count_match + 1
    acc = count_match / len(y_pred)
    
    end_process = time.time()
    
    end = time.time()
    
    return frt.output({
        'input_time': end_download - start_download, 
        'compute_time': end_process - start_process,
        'output_time': 0,
        'total_time': end - start, 
        'acc': acc
    })
