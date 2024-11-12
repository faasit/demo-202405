from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from numpy import genfromtxt
from numpy import concatenate
from numpy import savetxt
import numpy as np

import json
import random
import time
import io
from faasit_runtime import FaasitRuntime, function, create_handler
from typing import Dict, Any, Optional
import numpy as np
import logging

@function
def lambda_handler(frt: FaasitRuntime):
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

handler = create_handler(lambda_handler)