from typing import List, Tuple
import time
import numpy as np


def split(A: np.ndarray, B: np.ndarray, M: int, N: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    A_blocks = np.array_split(A, M, axis=0)
    B_blocks = np.array_split(B, N, axis=1)
    return A_blocks, B_blocks

def gemm(A_block: np.ndarray, B_block: np.ndarray) -> np.ndarray:
    return np.dot(A_block, B_block)

def merge(C_blocks: List[List[np.ndarray]], M: int, N: int) -> np.ndarray:
    return np.block(C_blocks)

def distributed_matrix_multiply(A: np.ndarray, B: np.ndarray, M: int, N: int) -> np.ndarray:
    A_blocks, B_blocks = split(A, B, M, N)
    C_blocks = [[np.zeros(0) for _ in range(N)] for _ in range(M)]
    
    for i, A_block in enumerate(A_blocks):
        for j, B_block in enumerate(B_blocks):
            C_blocks[i][j] = gemm(A_block, B_block)

    return merge(C_blocks, M, N)

def test_distributed_matrix_multiply():
    np.random.seed(42)  # 为了结果可重现
    m, k, n = 1024, 512, 1024
    A = np.random.rand(m, k)
    B = np.random.rand(k, n)
    M, N = 4, 4
    
    C_distributed = distributed_matrix_multiply(A, B, M, N)
    C_numpy = np.dot(A, B)
    
    # print("Distributed result:")
    # print(C_distributed)
    # print("\nNumPy result:")
    # print(C_numpy)
    print("\nMax difference:", np.max(np.abs(C_distributed - C_numpy)))
    print("Mean absolute error:", np.mean(np.abs(C_distributed - C_numpy)))

if __name__ == "__main__":
    test_distributed_matrix_multiply()



class NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

nullcontext = NullContext

from faasit_runtime import function, create_handler,FaasitRuntime

@function
def workflow_merge(frt: FaasitRuntime):
    _input = frt.input()
    store = frt.storage
    with nullcontext():
        input_start = time.time()
        input_pattern = str(_input['input']['pattern'])
        M = int(_input['input']['M'])
        N = int(_input['input']['N'])

        C_blocks = [[None for _ in range(N)] for _ in range(M)]
        for i in range(M):
            for j in range(N):
                C_blocks[i][j] = store.get(input_pattern.replace('%i%', str(i)).replace('%j%', str(j)), src_stage='compute')
        input_end = time.time()

    with nullcontext():
        compute_start = time.time()
        result = merge(C_blocks, M, N)
        compute_end = time.time()

    with nullcontext():
        output_start = time.time()
        filename = str(_input['output']['filename'])
        # Result is too big (>512MB) for Redis, so we output it to a file.
        # md.output([None], filename, result)
        with open("/tmp"+filename, "wb") as f:
            np.save(f, result)
        output_end = time.time()

    return frt.output({
        'input_time': input_end - input_start,
        'compute_time': compute_end - compute_start,
        'output_time': output_end - output_start,
        'total_time': output_end - input_start
    })

handler = create_handler(workflow_merge)