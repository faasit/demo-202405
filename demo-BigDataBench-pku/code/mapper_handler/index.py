import time
from typing import List
import sys
import pickle

from faasit_runtime import function,create_handler
from faasit_runtime import FaasitRuntime

def threaded_input_function(frt: FaasitRuntime,input: dict):
    suc = True
    key_name = input.get('key_name',None)
    stage_name = input.get('stage_name',None)
    filename = None
    if stage_name == None:
        filename = key_name
    else:
        filename = f"{stage_name}/{key_name}"
    obj = frt.storage.get(filename)
    assert(obj is not None)
    return {
        'suc':suc,
        'obj_to_recv': obj
    }

def threaded_output_function(frt: FaasitRuntime,input: dict):
    suc = True
    key_name = input.get('key_name')
    stage_name = input.get('stage_name')
    obj_to_send = input.get('obj_to_send')
    filename = None
    if stage_name == None:
        filename = key_name
    else:
        filename = f"{stage_name}/{key_name}"
    frt.storage.put(filename, pickle.dumps(obj_to_send),dest_stages=[stage_name],active_send=True)
    return frt.output({
        'suc':suc
    })

@function
def mapper_handler(frt: FaasitRuntime):
    # input_address should be a single string.
    # Get its partition.
    # Split.
    # Map to tuple with counter one and aggregate.
    # Organize output stages and keys according to hash, and issue output.
    # Output is a dict.
    input = frt.input()

    num_reducers = input.get('num_reducers',0)
    assert(num_reducers > 0)
    stage: str = input['stage']
    task_id = int(stage.split('-')[-1])
    input_name = f'stage0-{task_id}-input'

    output_stages = [f'stage1-{tempi}' for tempi in range(num_reducers)]
    output_names = [f'stage1-{task_id}-{tempi}-input' for tempi in range(num_reducers)]

    output_dicts = []
    for i in range(num_reducers):
        output_dicts.append({})

    input_st = time.perf_counter()
    # TODO fetch data from redis
    res = threaded_input_function(frt=frt,input={'key_name': input_name})
    obj:bytes = res['obj_to_recv']
    input_ed = time.perf_counter()
    input_time = input_ed - input_st
    input_str : str = obj.decode('utf-8')
    input_list : List[str] = input_str.split(" ")

    for word in input_list:
        hashval = hash(word) % num_reducers
        if word in output_dicts[hashval]:
            output_dicts[hashval][word] += 1
        else:
            output_dicts[hashval][word] = 1

    comed = time.perf_counter()
    compute_time = comed - input_ed
    retval_lists = []

    for i in range(num_reducers):
        t = threaded_output_function(frt=frt,input={
            'stage_name': output_stages[i],
            'key_name': output_names[i],
            'obj_to_send': output_dicts[i]
        })
        retval_lists.append(t)
    i = 0
    for retval_lst in retval_lists:
        suc = retval_lst['suc']
        if suc == False:
            print(f"Failed to output to reducer {i} in mapper {task_id}",file=sys.stderr)
            raise Exception(f"Failed to output to reducer {i} in mapper {task_id}")
        i += 1
    outed = time.perf_counter()
    output_time = outed - comed
    return frt.output({
        'input_time' : input_time,
        'compute_time' : compute_time,
        'output_time' : output_time,
        'total_time' : outed - input_st
    })
    
handler = create_handler(mapper_handler)