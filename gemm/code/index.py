from faasit_runtime import workflow,Workflow,create_handler

@workflow
def gemm(wf:Workflow):
    split = wf.call('split', {
        'input': {
            "m": 4096,
            "k": 4096,
            'n': 4096,
            "M": 2,
            "N": 2,
        }, 
        'output': {
            "pattern_A": "gemm_split_result_A_%i%",
            "pattern_B": "gemm_split_result_B_%j%",
        }
    })
    compute = wf.call('compute', {'split': split,
        'input': {
            "pattern_A": 'gemm_split_result_A_%i%',
            "pattern_B": "gemm_split_result_B_%j%",
            "M": 2,
            "N": 2,
        }, 
        'output': {
            "pattern": "gemm_compute_result_%i%_%j%",
        }
    })
    merge = wf.call('merge', {'compute': compute,
        'input': {
            "pattern": "gemm_compute_result_%i%_%j%",
            "M": 2,
            "N": 2,
        },
        'output': {
            "filename": "gemm_merge_result",
        }
    })
    return merge


handler = create_handler(gemm)
