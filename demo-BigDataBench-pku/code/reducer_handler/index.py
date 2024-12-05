import time
from typing import List, Dict
import sys
import json
import pickle

from faasit_runtime import function, create_handler
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

    obj = frt.storage.get(src_stage=stage_name,filename=filename,active_pull=True,tcp_direct=True)
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
    frt.storage.put(filename, pickle.dumps(obj_to_send), dest_stages=[stage_name])
    return frt.output({
        'suc':suc
    })


@function
def reducer_handler(frt: FaasitRuntime):
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
        retval = threaded_input_function(frt=frt,input={
            'stage_name': input_stage_names[i],
            'key_name': input_key_names[i]
        })
        suc = retval['suc']
        temp_dict_to_merge: bytes = retval['obj_to_recv']
        temp_dict_to_merge = pickle.loads(temp_dict_to_merge)
        if not suc:
            print(f"Failed to get input from mapper {i} in reducer {task_id}", file=sys.stderr)
            raise Exception(f"Failed to get input from mapper {i} in reducer {task_id}")
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

handler = create_handler(reducer_handler)