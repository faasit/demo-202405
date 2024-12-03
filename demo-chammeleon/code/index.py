from faasit_runtime import create_handler, workflow, Workflow

def get_timestamp():
    from datetime import datetime
    return str(datetime.now())

@workflow
def chammeleon_workflow(wf:Workflow):
    r = wf.call("stage0",{'params':{}})
    return r

handler = create_handler(chammeleon_workflow)