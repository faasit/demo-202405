import time
import json

import train
import aggregate
import test

def handler(event, context):
    if('dummy' in event) and (event['dummy'] == 1):
        print("Dummy call, doing nothing")
        return
    
    if event['func_id'] == 1:
        num_tasks = int(event['num_tasks'])
        num_vcpu = int(event['num_vcpu'])
        if 'total_trees' in event:
            total_trees = int(event['total_trees'])
        else:
            total_trees = 16
        task_id = int(event['task_id'])
        loc = {'input_address' : event['input_address'], 
               'output_address' : event['output_address']}
        
        assert total_trees >= num_tasks
        
        # we are going to train total_trees trees
        # all trees are divided equally among tasks
        # a task has multiple processes, each for a tree


        average_tasks = int(total_trees / num_tasks)
        remain_tasks = total_trees % num_tasks
        
        if task_id < remain_tasks:
            num_processes = average_tasks + 1
        else:
            num_processes = average_tasks
            
        res = train.train_with_multprocess(task_id, num_processes, num_vcpu, loc)
        
    elif event['func_id'] == 2:
        num_tasks = int(event['num_tasks'])
        task_id = int(event['task_id'])
        loc = {'input_address' : event['input_address'], 
               'output_address' : event['output_address']}
        res = aggregate.aggregate_models(num_tasks, task_id, loc)
        
    elif event['func_id'] == 3:
        loc = {'input_address' : event['input_address'], 
               'output_address' : event['output_address']}
        res = test.test_forests(loc)
    else:
        raise Exception('Invalid func_id')
    
    return_val = {
        'breakdown': res
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(return_val)
    }