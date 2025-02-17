from faasit_runtime import workflow,create_handler
from faasit_runtime.workflow import Workflow

@workflow
def safe_resersation(wf:Workflow):
    wf.call('reversehotel',{
        'lambdaId': 'reversehotel',
        'instanceId': 3,
        'selected_totel': [1,2,3]
    })

handler = create_handler(safe_resersation)