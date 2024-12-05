
from faasit_runtime import workflow,create_handler
from faasit_runtime.workflow import Workflow
from faasit_runtime import FaasitRuntime


@workflow
def wordcountflow(wf: Workflow):
    mapper_stage : str = 'stage0'
    tasks = []
    for i in range(4):
        t = wf.call(f'{mapper_stage}-{i}',{'params':{
            'stage': f'{mapper_stage}-{i}',
            'num_reducers': 4
            }
        })
        tasks.append(t)

    reducer_stage : str = 'stage1'
    for i in range(4):
        params = {'params' :{
            'stage': f'{reducer_stage}-{i}',
            'num_mappers': 4}
        }
        for j in range(4):
            params.setdefault(f'{mapper_stage}-{j}',tasks[j])
        t = wf.call(f'{reducer_stage}-{i}',params)
    return tasks[0]

handler = create_handler(wordcountflow)