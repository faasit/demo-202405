import json
from numpy import array
import numpy as np
import time
import logging
from faasit_runtime import function,workflow,create_handler
from faasit_runtime.runtime import FaasitRuntime

def upload_matrix(A,  filename):
    assert(0)
    # numpy.savetxt("/tmp/"+filename, A)
    # s3_client.upload_file("/tmp/"+filename, bucket_name, "ChatBotData/" + filename, Config=config)

@function
async def upload_BOW(frt:FaasitRuntime):
    tmp_path =  './tmp/bos.txt'
    _in = frt.input()
    BOW = _in['BOW']
    output = _in['output']

    with open(tmp_path,'w', encoding='utf-8') as f:
        for word in BOW:
            f.write(word+ '\n')

    with open(tmp_path, 'rb') as data:
        sth = data.read()
        frt.storage.put(f'train/{output["bos"]}', sth)
    return frt.output({})

        
@function
async def main(frt:FaasitRuntime):
    _in = frt.input()
    start_time = time.time()
    input_time = 0.0
    output_time = 0.0
    compute_time = 0.0

    data = []

    input_time -= time.time()

    tmp_path = './tmp/intent.json'
    with open(tmp_path, 'wb') as f:
        f.write(frt.storage.get(_in["input"]["indent"]))

    with open(tmp_path, 'r', encoding='utf-8') as file:
        data = file.read().replace('\n', '')

    j_data = json.loads(data)

    input_time += time.time()

    
    compute_time -= time.time()

    all_unique_words = []
    all_intents=[]
    for v in range(len(j_data["intents"])):
        newIntent = {}
        newIntent["name"] = j_data["intents"][v]["intent"]
        newIntent["data"] = j_data["intents"][v]["text"]
        newIntent["data"].extend(j_data["intents"][v]["responses"])
        for utterance in newIntent["data"]:
            words_list= utterance.split(" ") 
            all_unique_words.extend(words_list)
        all_intents.append(newIntent)
        
    BOW=set(all_unique_words)
    All_matrices=[]
    for newIntent in all_intents:
        list_vectors=[]
        for utterance in  newIntent["data"]:
            words_list = utterance.split(" ")
            vector = [int(w in words_list) for w in BOW]
            list_vectors.append(vector)
        A = array(list_vectors)
        All_matrices.append(A)
    
    
    list_of_intents = []

    bundle_size = _in["bundle_size"]
    
    for mat_index in range(len(All_matrices)):
        positive_A = All_matrices[mat_index]
        negative_A = []
        if(mat_index > len(All_matrices) -4):
             
            negative_A =  All_matrices[0]
            negative_A = np.concatenate((negative_A, All_matrices[1]), axis=0)
            negative_A = np.concatenate((negative_A, All_matrices[2]), axis=0)
            
        else:
            negative_A =  All_matrices[mat_index+1]
            negative_A = np.concatenate((negative_A, All_matrices[mat_index+2]), axis=0)
            negative_A = np.concatenate((negative_A, All_matrices[mat_index+3]), axis=0)
            
            
        j={ "intent_name":all_intents[mat_index]["name"], "skew" : _in["skew"]}
        list_of_intents.append(j)

    compute_time += time.time()

    output_time -= time.time()

    
    await frt.call("upload_BOW", {'BOW': BOW, 'output': _in['output']})
    frt.storage.put(f'train/{_in["output"]["list_of_indents"]}', json.dumps(list_of_intents).encode())
    output_time += time.time()

    end_time = time.time()
    
    
    return frt.output({
        'input_time': input_time,
        'compute_time': compute_time,
        'output_time': output_time,
        'total_time': end_time - start_time,
    })