from faasit_runtime import workflow,function,FaasitRuntime
from faasit_runtime.workflow import Workflow
from faasit_runtime import FaasitRuntime
from typing import List, Dict
import time
import pickle
import sys
import json

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
    frt.storage.put(filename, pickle.dumps(obj_to_send),dest_stages=[stage_name])
    return frt.output({
        'suc':suc
    })

@function
def mapper(frt: FaasitRuntime):
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
    output_names = [f'stage1-{tempi}-{task_id}-input' for tempi in range(num_reducers)]

    output_dicts = []
    for i in range(num_reducers):
        output_dicts.append({})

    input_st = time.perf_counter()
    # TODO fetch data from redis
    obj = frt.storage.get(input_name)
    assert obj is not None
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
        frt.storage.put(output_names[i], pickle.dumps(output_dicts[i]),dest_stages=[output_stages[i]], active_send=True)
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

@function
def reducer(frt: FaasitRuntime):
    input = frt.input()
    num_mappers = input.get('num_mappers', 0)
    assert num_mappers > 0
    stage: str = input['stage']
    task_id = int(stage.split('-')[-1])

    input_stage_names = [f'stage1-{i}' for i in range(num_mappers)]
    input_key_names = [f'stage1-{task_id}-{i}-input' for i in range(num_mappers)]

    input_st = time.perf_counter()

    dicts_list: List[Dict[str, int]] = []
    for i in range(num_mappers):
        temp_dict_to_merge = frt.storage.get(input_key_names[i], src_stage=input_stage_names[i],active_pull=False)
        temp_dict_to_merge = pickle.loads(temp_dict_to_merge)
        dicts_list.append(temp_dict_to_merge)

    input_ed = time.perf_counter()
    input_time = input_ed - input_st

    result_dict: Dict[str, int] = {}
    for temp_dict in dicts_list:
        for word, val in temp_dict.items():
            if word in result_dict:
                result_dict[word] += val
            else:
                result_dict[word] = val

    js_string = json.dumps(result_dict)
    comed = time.perf_counter()
    compute_time = comed - input_ed
    threaded_output_function(frt=frt,input={
        'key_name': f'stage1-finalresult-{task_id}',
        'obj_to_send': js_string
    })
    outed = time.perf_counter()
    output_time = outed - comed

    return frt.output({
        'input_time': input_time,
        'compute_time': compute_time,
        'output_time': output_time,
        'total_time': outed - input_st,
    })

@workflow
def wordcountflow(wf: Workflow):
    mapper_stage : str = 'stage0'
    tasks = []
    for i in range(4):
        t = wf.call(f'{mapper_stage}-{i}',{
            'stage': f'{mapper_stage}-{i}',
            'num_reducers': 4
            })
        tasks.append(t)

    reducer_stage : str = 'stage1'
    for i in range(4):
        params = {
            'stage': f'{reducer_stage}-{i}',
            'num_mappers': 4
        }
        for j in range(4):
            params.setdefault(f'{mapper_stage}-{j}',tasks[j])
        t = wf.call(f'{reducer_stage}-{i}',params)
    return tasks[0]

mapper = mapper.export()
reducer = reducer.export()
wordcountflow = wordcountflow.export()