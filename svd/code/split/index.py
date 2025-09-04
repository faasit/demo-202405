from typing import List, Tuple
import time
import numpy as np
from faasit_runtime import function, create_handler,FaasitRuntime

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

@function
def workflow_split(frt: FaasitRuntime):
    _input = frt.input()
    store = frt.storage
    with nullcontext():
        input_start = time.time()
        m = int(_input['input']['m'])
        k = int(_input['input']['k'])
        n = int(_input['input']['n'])
        M = int(_input['input']['M'])
        N = int(_input['input']['N'])
        output_pattern_A = str(_input['output']['pattern_A'])
        output_pattern_B = str(_input['output']['pattern_B'])

        A = np.random.rand(m, k)
        B = np.random.rand(k, n)
        A_blocks, B_blocks = split(A, B, M, N)
        input_end = time.time()

    with nullcontext():
        output_start = time.time()
        for i, A_block in enumerate(A_blocks):
            store.put(output_pattern_A.replace('%i%', str(i)), A_block, dest_stages=['compute'])
        for j, B_block in enumerate(B_blocks):
            store.put(output_pattern_B.replace('%j%', str(j)), B_block, dest_stages=['compute'])
        output_end = time.time()

    return {
        'input_time': input_end - input_start,
        'output_time': output_end - output_start,
        'total_time': output_end - input_start
    }

handler = create_handler(workflow_split)