from faasit_runtime import function, FaasitRuntime
from faasit_runtime import workflow, Workflow
import time
import numpy as np


def generate_positive_definite_matrix(size):
    A = np.random.rand(size, size)
    return np.dot(A, A.T) + np.eye(size) * size

def split(matrix, block_size):
    n = matrix.shape[0]
    blocks = []
    for i in range(0, n, block_size):
        row = []
        for j in range(0, n, block_size):
            block = matrix[i:i+block_size, j:j+block_size]
            row.append(block)
        blocks.append(row)
    return blocks

def compute_block(block):
    try:
        return np.linalg.cholesky(block)
    except np.linalg.LinAlgError:
        min_eig = np.min(np.real(np.linalg.eigvals(block)))
        if min_eig < 0:
            block += (-min_eig + 1e-6) * np.eye(block.shape[0])
        else:
            block += 1e-6 * np.eye(block.shape[0])
        return np.linalg.cholesky(block)

def compute(blocks):
    n = len(blocks)
    L_blocks = [[np.zeros_like(block) for block in row] for row in blocks]

    for i in range(n):
        L_blocks[i][i] = compute_block(blocks[i][i])

        # Parallel processing for non-diagonal blocks
        def process_block(j):
            L_blocks[j][i] = np.linalg.solve(L_blocks[i][i].T, blocks[j][i].T).T
            return j, L_blocks[j][i]

        results = []
        for j in range(i+1, n):
            results.append(process_block(j))
        # results = Parallel(n_jobs=-1)(delayed(process_block)(j) for j in range(i+1, n))
        for j, result in results:
            L_blocks[j][i] = result

        for j in range(i+1, n):
            for k in range(i+1, j+1):
                blocks[j][k] -= np.dot(L_blocks[j][i], L_blocks[k][i].T)

    return L_blocks

def merge(blocks):
    return np.block(blocks)

def cholesky_workflow(matrix, block_size):
    blocks = split(matrix, block_size)
    computed_blocks = compute(blocks)
    result = merge(computed_blocks)
    return np.tril(result)  # Ensure lower triangular

class NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

nullcontext = NullContext

@function
def workflow_split(frt: FaasitRuntime):
    _input = frt.input()
    store = frt.storage
    with nullcontext():
        input_start = time.time()
        matrix_size = int(_input['input']['matrix_size'])
        split_num = int(_input['input']['split_num'])
        block_size = matrix_size // split_num
        output_pattern = str(_input['output']['pattern'])

        A = np.eye(matrix_size) * matrix_size
        blocks = split(A, block_size)
        input_end = time.time()

    with nullcontext():
        output_start = time.time()
        for i in range(split_num):
            for j in range(split_num):
                store.put(output_pattern.replace('%i%', str(i)).replace('%j%', str(j)), blocks[i][j], dest_stages=['compute'])
        output_end = time.time()

    return frt.output({
        'input_time': input_end - input_start,
        'output_time': output_end - output_start,
        'total_time': output_end - input_start
    })

@function
def workflow_compute(frt:FaasitRuntime):
    _input = frt.input()
    store = frt.storage
    with nullcontext():
        input_start = time.time()
        input_pattern = str(_input['input']['pattern'])
        output_pattern = str(_input['output']['pattern'])
        split_num = int(_input['input']['split_num'])

        blocks = [[None for _ in range(split_num)] for _ in range(split_num)]
        for i in range(split_num):
            for j in range(split_num):
                blocks[i][j] = store.get(input_pattern.replace('%i%', str(i)).replace('%j%', str(j)), src_stage='split')
        input_end = time.time()

    with nullcontext():
        compute_start = time.time()
        computed_blocks = compute(blocks)
        compute_end = time.time()

    with nullcontext():
        output_start = time.time()
        for i in range(split_num):
            for j in range(split_num):
                store.put(output_pattern.replace('%i%', str(i)).replace('%j%', str(j)), computed_blocks[i][j],dest_stages=['merge'])
        output_end = time.time()

    return frt.output({
        'input_time': input_end - input_start,
        'compute_time': compute_end - compute_start,
        'output_time': output_end - output_start,
        'total_time': output_end - input_start
    })

@function
def workflow_merge(frt: FaasitRuntime):
    _input = frt.input()
    store = frt.storage
    with nullcontext():
        input_start = time.time()
        input_pattern = str(_input['input']['pattern'])
        split_num = int(_input['input']['split_num'])
        matrix_size = int(_input['input']['matrix_size'])
        block_size = matrix_size // split_num

        blocks = [[None for _ in range(split_num)] for _ in range(split_num)]
        for i in range(split_num):
            for j in range(split_num):
                blocks[i][j] = store.get(input_pattern.replace('%i%', str(i)).replace('%j%', str(j)), src_stage='compute')
        input_end = time.time()

    with nullcontext():
        compute_start = time.time()
        result = merge(blocks)
        result = np.tril(result)
        compute_end = time.time()

    with nullcontext():
        output_start = time.time()
        filename = str(_input['output']['filename'])
        # Result is too big (>512MB) for Redis, so we output it to a file.
        # md.output([None], filename, result)
        with open(filename, 'wb') as f:
            np.save(f, result)
        output_end = time.time()

    return frt.output({
        'input_time': input_end - input_start,
        'compute_time': compute_end - compute_start,
        'output_time': output_end - output_start,
        'total_time': output_end - input_start
    })

@workflow
def cholesky(wf:Workflow):
    split_num = 2
    matrix_size = 5000
    s0 = wf.call('split', {
        "input": {
            "matrix_size": matrix_size,
            "split_num": split_num # = matrix_size / block_size
        },
        "output": {
            "pattern": "cholesky_split_result_%i%_%j%"
        }
    })
    
    s1 = wf.call('compute', {"s0": s0,
        "input": {
            "pattern": "cholesky_split_result_%i%_%j%",
            "split_num": split_num
        },
        "output": {
            "pattern": "cholesky_compute_result_%i%_%j%"
        }
    })
    s2 = wf.call('merge', {"s1": s1,
        "input": {
            "pattern": "cholesky_compute_result_%i%_%j%",
            "split_num": split_num,
            "matrix_size": matrix_size
        },
        "output": {
            "filename": "cholesky_result.npy"
        }
    })
    return s2

cholesky = cholesky.export()
workflow_split = workflow_split.export()
workflow_compute = workflow_compute.export()
workflow_merge = workflow_merge.export()