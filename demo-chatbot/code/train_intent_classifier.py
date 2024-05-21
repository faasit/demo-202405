import json
from scipy.linalg import svd
import numpy
from numpy import array
import numpy as np
import time
from multiprocessing import Process, Manager
from typing import List, Any
from faasit_runtime import function, create_handler
from faasit_runtime.runtime import FaasitRuntime
# def download_matrix(intent_name):
#     filename = "/tmp/" + intent_name
#     f = open(filename, "wb")
#     s3_client.download_fileobj(bucket_name, "ChatBotData/" + intent_name , f, Config=config)
#     f.close()
    
#     return numpy.loadtxt(filename)

@function
async def load_bow(frt: FaasitRuntime):
    _in = frt.input()
    tmp_path = './tmp/bos.txt'

    with open(tmp_path, 'wb') as f:
        f.write(frt.storage.get(f'train/{_in["input"]["bos"]}'))
    
    BOW = []
    with open(tmp_path, 'r', encoding='utf-8') as f:
        BOW = f.readlines()
    return frt.output({'BOW': BOW})    

@function
async def lambda_handler(frt: FaasitRuntime):
    _in = frt.input()
    input_time = 0.0
    compute_time = 0.0
    output_time = 0.0

    start_time = time.time()

    input_time -= time.time()
    all_intents = await frt.call("load_indents",_in)
    list_of_intents: List[Any] = frt.storage.get(f'train/{_in["input"]["list_of_indents"]}')
    list_of_intents = json.loads(list_of_intents)
    BOW = await frt.call('load_bow', _in)
    input_time += time.time()
    
    
    compute_time -= time.time()
    num_workers = len(list_of_intents)
    for w in range(num_workers):
        intent_name = list_of_intents[w]["intent_name"]
        skew = list_of_intents[w]["skew"]
        run_worker(all_intents, intent_name, skew, BOW)
    compute_time += time.time()

    end_time = time.time()

    return {
        'input_time': input_time,
        'compute_time': compute_time,
        'output_time': output_time,
        'total_time': end_time - start_time,
    }
    
    
def run_worker(all_intents, intent_name, skew, BOW):
    # Prepare positive and Negative Matrixes
    positive_matrix = get_matrix_for_intent(all_intents, intent_name, BOW)

    negative_matrix: np.ndarray = array([])
    count_negative = 0
    All_intents_names = all_intents.keys()
    for negative_intents in All_intents_names:
        if(negative_intents != intent_name):
            if(count_negative == 0):
                negative_matrix = get_matrix_for_intent(all_intents, intent_name, BOW)
            else:        
                negative_matrix = np.concatenate((negative_matrix, get_matrix_for_intent(all_intents, intent_name, BOW)), axis=0) # type: ignore
            count_negative += 1
            if (count_negative > 2):
                break
    
    if negative_matrix.shape[0] >positive_matrix.shape[0]:
        negative_matrix = negative_matrix[0:positive_matrix.shape[0], :]
    
    positive_labels = np.ones(positive_matrix.shape[0])
    negative_labels = np.zeros(negative_matrix.shape[0])
    
    y = np.concatenate((positive_labels, negative_labels), axis=0)
    for s in range(skew):
        y = np.concatenate((positive_labels, y), axis=0)
    
    y = y[:,np.newaxis] # type: ignore
    X_org = np.concatenate((positive_matrix, negative_matrix), axis=0)
    for s in range(skew):
        X_org = np.concatenate((positive_matrix, X_org), axis=0)
    
    #for i in range(skew):
    X = get_svd(X_org, intent_name)
    
    score=1
    
    X = np.hstack((np.ones((len(y),1)),X))
    #print(X.shape)
    n = np.size(X,1)
    #print(n)
    params = np.ones((n,1))

    iterations = 1500
    learning_rate = 0.03
    initial_cost = compute_cost(X, y, params)

    #print("Initial Cost is: {} \n".format(initial_cost))

    (cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)

    y_pred = predict(X, params_optimal)
    score = float(sum(y_pred == y))/ float(len(y))

    # output_time = -time.time()
    # upload_matrix(params_optimal, intent_name + "_params.txt")
    # output_time += time.time()

    # return output_time


###################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # type: ignore

def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    #print("Before Gradient:")
    #print(params.shape)
    for i in range(iterations):
        
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        #print("After Update:")    
        #print(params.shape)
        c = compute_cost(X, y, params)
        #print(c)
        cost_history[i] = c
    #print("After Gradient:")    
    #print(params.shape)
    
    return (cost_history, params)

def compute_cost(X, y, theta):
    #print(X.shape)
    #print(y.shape)
    #print(theta.shape)
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon))) # type: ignore
    return cost

def predict(X, params):
    #print(X.shape)
    #print(params.shape)
    
    multip = X @ params
    #print(multip.shape)
    return np.round(sigmoid(X @ params))
    
# def upload_matrix(A,  filename):
#      numpy.savetxt("/tmp/"+filename, A)
#      s3_client.upload_file("/tmp/"+filename, bucket_name, "ChatBotData/" + filename, Config=config)
         
def get_svd(A, intent_name):
    
    #if(intent_name !="Jokes"):
    #    return A
        
    U, s, VT = svd(A)
    Sigma = numpy.zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[0], :A.shape[0]] = numpy.diag(s)
    # select
    n_elements = 500
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # reconstruct
    B = U.dot(Sigma.dot(VT))
    # transform
    T = U.dot(Sigma)
    return T
    
@function
async def load_indents(frt: FaasitRuntime):
    _in = frt.input()
    tmp_path =  './tmp/intent.json'
    with open(tmp_path, 'wb') as f:
        f.write(frt.storage.get(_in['input']['indent']))
    
    data = []
    with open(tmp_path, 'r', encoding='utf-8') as file:
        data = file.read().replace('\n', '')
    
    j_data = json.loads(data)    
    all_unique_words = []
    
    all_intents={}
    for v in range(len(j_data["intents"])):
        newIntent = {}
        newIntent["name"] = j_data["intents"][v]["intent"]
        newIntent["data"] = j_data["intents"][v]["text"]
        newIntent["data"].extend(j_data["intents"][v]["responses"])
        for utterance in newIntent["data"]:
            words_list= utterance.split(" ") 
            all_unique_words.extend(words_list)
        all_intents[newIntent["name"]] = newIntent
    return frt.output(all_intents)    
    
def get_matrix_for_intent(all_intents, intent_name, BOW) -> np.ndarray:
    list_vectors = []
    for utterance in all_intents[intent_name]["data"]:
        words_list = utterance.split(" ")
        vector = [int(w in words_list) for w in BOW]
        list_vectors.append(vector)
    positive_matrix = array(list_vectors) 
    return positive_matrix
