from faasit_runtime import create_handler, workflow, Workflow

@workflow
def ParallelSVD(wf: Workflow):
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    num_splits = 2
    res = wf.call("stage0",{"X": X, "numSplits": num_splits})
    return res

handler = create_handler(ParallelSVD)

