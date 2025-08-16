from faasit_runtime import workflow, Workflow, FaasitRuntime, function
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from numpy import genfromtxt
from numpy import concatenate
from numpy import savetxt
import numpy as np

import json
import lightgbm as lgb
import random
import time
import io
from typing import Dict, Any, Optional
import numpy as np
import logging

from sklearn.base import BaseEstimator
from scipy.special import softmax
import numpy as np
from joblib import dump, load
from multiprocessing import Process
import time

class MergedLGBMClassifier(BaseEstimator):
    def __init__(self, model_list):
        assert isinstance(model_list, list)
        self.model_list = model_list

    def predict(self, X) -> np.ndarray:
        pred_list = []
        
        for m in self.model_list:
            pred_list.append(m.predict(X))

        # Average the predictions
        averaged_preds = sum(pred_list) / len(pred_list)

        return averaged_preds
    
    def save_model(self, model_path):
        dump(self, model_path)
    
    @staticmethod
    def load_model(model_path):
        return load(model_path)
    
class MyProcess(Process):
    def __init__(self, target, args):
        super().__init__()
        assert callable(target)
        self._result = None
        self._my_function = target
        self._args = args

    def run(self):
        if self._args is None:
            result = self._my_function()
        else:
            result = self._my_function(self._args)
        self._result = result

    @property
    def result(self):
        return self._result
    

# MyPool run() will not finish and sticks into the while loop
class MyPool:
    def __init__(self, size, processes):
        assert size > 0
        assert isinstance(size, int)
        assert isinstance(processes, list)
        assert len(processes) > 0
        assert all(isinstance(p, Process) for p in processes)
        
        self.size = size
        self.processes = processes
        self.start = [False] * len(processes)
        self.finish = [False] * len(processes)
        self.quota = self.size
        
    def update(self):
        for i in range(len(self.processes)):
            if self.start[i] and not self.finish[i]:
                if self.processes[i].is_alive():
                    continue
                else:
                    self.finish[i] = True
                    self.quota += 1
        
    def run(self):
        index = 0
        num = len(self.processes)
        t0 = time.time()
        while self.finish[num - 1] is False:
            self.update()
            ct = time.time()
            if ct - t0 > 15:
                print('quota: ', self.quota)
                print('index: ', index)
                print(self.processes[0].is_alive())
                break
            if self.quota > 0 and index < num:
                assert self.start[index] is False
                self.processes[index].start()
                self.start[index] = True
                self.quota -= 1
                index += 1
        for p in self.processes:
            p.join()

@function
def test_forests(frt: FaasitRuntime):
    start = time.time()
    
    start_download = time.time()
    _input = frt.input()
    input = _input['input']
    assert len(input) == 2
    
    store = frt.storage
    test_data = store.get(input['train_pca_transform'], timeout = 5, src_stage='stage0')
    pred: Optional[np.ndarray] = store.get(input['predict'], timeout = 5, src_stage='stage2')
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

class MergedLGBMClassifier(BaseEstimator):
    def __init__(self, model_list):
        assert isinstance(model_list, list)
        self.model_list = model_list

    def predict(self, X) -> np.ndarray:
        pred_list = []
        
        for m in self.model_list:
            pred_list.append(m.predict(X))

        # Average the predictions
        averaged_preds = sum(pred_list) / len(pred_list)

        return averaged_preds
    
    def save_model(self, model_path):
        dump(self, model_path)
    
    @staticmethod
    def load_model(model_path):
        return load(model_path)
    
class MyProcess(Process):
    def __init__(self, target, args):
        super().__init__()
        assert callable(target)
        self._result = None
        self._my_function = target
        self._args = args

    def run(self):
        if self._args is None:
            result = self._my_function()
        else:
            result = self._my_function(self._args)
        self._result = result

    @property
    def result(self):
        return self._result
    

# MyPool run() will not finish and sticks into the while loop
class MyPool:
    def __init__(self, size, processes):
        assert size > 0
        assert isinstance(size, int)
        assert isinstance(processes, list)
        assert len(processes) > 0
        assert all(isinstance(p, Process) for p in processes)
        
        self.size = size
        self.processes = processes
        self.start = [False] * len(processes)
        self.finish = [False] * len(processes)
        self.quota = self.size
        
    def update(self):
        for i in range(len(self.processes)):
            if self.start[i] and not self.finish[i]:
                if self.processes[i].is_alive():
                    continue
                else:
                    self.finish[i] = True
                    self.quota += 1
        
    def run(self):
        index = 0
        num = len(self.processes)
        t0 = time.time()
        while self.finish[num - 1] is False:
            self.update()
            ct = time.time()
            if ct - t0 > 15:
                print('quota: ', self.quota)
                print('index: ', index)
                print(self.processes[0].is_alive())
                break
            if self.quota > 0 and index < num:
                assert self.start[index] is False
                self.processes[index].start()
                self.start[index] = True
                self.quota -= 1
                index += 1
        for p in self.processes:
            p.join()

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
    test_data = store.get(input['train_pca_transform'], timeout = 5, src_stage='stage0')
    input_model = store.get(input['model'], timeout = 5, src_stage='stage1')
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
    store.put(output['predict'], y_pred,dest_stages=['stage3'])
    end_upload = time.time()
    
    end = time.time()
    
    return frt.output({
        'input_time': end_download - start_download, 
        'compute_time': end_process - start_process, 
        'output_time': end_upload - start_upload, 
        'total_time': end - start, 
        'acc': acc
    })

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

@function
def pca(frt: FaasitRuntime):
    params = frt.input()

    input = params['input']
    output = params['output']

    start_time = time.time()
    store = frt.storage
    file_str = store.get(input)
    assert(file_str is not None)
    
    end_input = time.time()

    # save it into a file, and then genfromtxt
    filename = "/tmp/train_data.txt" + str(random.randint(0, 10**10))

    with open(filename, "wb") as f:
        f.write(file_str)

    train_data = genfromtxt(filename, delimiter="\t")

    train_labels = train_data[:,0]
    #test_labels = test_data[:,0]

    A = train_data[:,1:train_data.shape[1]]
    #B = test_data[:,1:test_data.shape[1]]

    # calculate the mean of each column
    MA = mean(A.T, axis=1)
    #MB = mean(B.T, axis=1)

    # center columns by subtracting column means
    CA = A - MA
    #CB = B - MB

    # calculate covariance matrix of centered matrix
    VA = cov(CA.T)

    # eigendecomposition of covariance matrix
    values, vectors = eig(VA)

    # project data
    PA = vectors.T.dot(CA.T)
    #PB = vectors.T.dot(CB.T)

    # np.save("/tmp/vectors_pca.txt", vectors)

    # savetxt("/tmp/vectors_pca.txt", vectors, delimiter="\t")


    #print("vectors shape:")
    #print(vectors.shape)


    first_n_A = PA.T[:,0:100].real
    #first_n_B = PB.T[:,0:10].real
    train_labels =  train_labels.reshape(train_labels.shape[0],1)
    #test_labels = test_labels.reshape(test_labels.shape[0],1)

    first_n_A_label = concatenate((train_labels, first_n_A), axis=1)
    #first_n_B_label = concatenate((test_labels, first_n_B), axis=1)

    # savetxt("/tmp/train_pca_transform.txt", first_n_A_label, delimiter="\t")
    #savetxt("/tmp/Digits_Test_Transform.txt", first_n_B_label, delimiter="\t")

    end_compute = time.time()

    store.put(params['output']['vectors_pca'], vectors,dest_stages=['stage1'])
    store.put(params['output']['train_pca_transform'], first_n_A_label, dest_stages=['stage1', 'stage2', 'stage3'])

    # s3_client.upload_file("/tmp/vectors_pca.txt", bucket_name, output[0], Config=config)
    # s3_client.upload_file("/tmp/train_pca_transform.txt", bucket_name, output[1], Config=config)
    # s3_client.upload_file("/tmp/Digits_Test_Transform.txt", bucket_name, "LightGBM_Data/test_pca_transform.txt", Config=config)

    end_output = time.time() 

    return frt.output({
        'input_time': end_input - start_time,
        'compute_time': end_compute - end_input, 
        'output_time': end_output - end_compute, 
        'total_time': end_output - start_time
    })


@workflow
def mlpipeline(wf: Workflow):
    s0 = wf.call('stage0',{
        'input': 'Digits_Train.txt',
        'output': {
            'vectors_pca': 'vectors_pca',
            'train_pca_transform': 'train_pca_transform',
        }
    })
    s1 = wf.call('stage1',{'stage0':s0, 
        'input': {
            'train_pca_transform': 'train_pca_transform'
        },
        'output': {
            'model': 'model_tree_0_0'
        }
    })
    s2 = wf.call('stage2',{'stage1':s1, 'stage0':s0, 
        'input': {
            'train_pca_transform': 'train_pca_transform',
            'model': 'model_tree_0_0'
        },
        'output': {
            'predict': 'predict_0',
        }
    })
    s3 = wf.call('stage3',{'stage2':s2, 'stage0':s0,
        'input': {
            'train_pca_transform': 'train_pca_transform',
            'predict': 'predict_0',
        },
    })

    return s3

test = test_forests.export()
aggregate = aggregate_models.export()
train_func = train_entry.export()
pca = pca.export()
mlpipe = mlpipeline.export()