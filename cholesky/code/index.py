from faasit_runtime import create_handler
from faasit_runtime import workflow, Workflow

@workflow
def cholesky(wf:Workflow):
    s0 = wf.call('split', {
        "input": {
            "matrix_size": 5000,
            "split_num": 10 # = matrix_size / block_size
        },
        "output": {
            "pattern": "cholesky_split_result_%i%_%j%"
        }
    })
    s1 = wf.call('compute', {"s0": s0,
        "input": {
            "pattern": "cholesky_split_result_%i%_%j%",
            "split_num": 10
        },
        "output": {
            "pattern": "cholesky_compute_result_%i%_%j%"
        }
    })
    s2 = wf.call('merge', {"s1": s1,
        "input": {
            "pattern": "cholesky_compute_result_%i%_%j%",
            "split_num": 10,
            "matrix_size": 5000
        },
        "output": {
            "filename": "cholesky_result.npy"
        }
    })
    return s2

handler = create_handler(cholesky)