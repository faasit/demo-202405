# from serverless_framework import limit_numpy_multithread
# limit_numpy_multithread()

import numpy as np
from joblib import Parallel, delayed

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

        results = Parallel(n_jobs=-1)(delayed(process_block)(j) for j in range(i+1, n))
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

def test_cholesky():
    # 设置参数
    matrix_size = 1000
    block_size = 200

    # don't use other workload. The cholesky decomposition algorithm is not steady.
    A = np.eye(matrix_size) * matrix_size

    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Block size: {block_size}x{block_size}")

    # 验证原始矩阵是否为正定矩阵
    eigenvalues = np.linalg.eigvals(A)
    if np.all(eigenvalues > 0):
        print("Original matrix is positive definite.")
    else:
        print("Warning: Original matrix is not positive definite.")
        raise ValueError("Original matrix is not positive definite.")

    # 执行分块Cholesky分解
    try:
        result = cholesky_workflow(A, block_size)
        print("\nCholesky decomposition completed successfully.")
    except Exception as e:
        print(f"\nError during Cholesky decomposition: {e}")
        return

    # 验证结果
    reconstructed = np.dot(result, result.T)
    error = np.max(np.abs(A - reconstructed))
    print(f"Maximum error: {error}")

    # 检查结果矩阵是否为下三角矩阵
    is_lower = np.allclose(result, np.tril(result))
    print(f"Result is lower triangular: {is_lower}")

    # 与NumPy的Cholesky分解比较
    try:
        np_cholesky = np.linalg.cholesky(A)
        cholesky_diff = np.max(np.abs(result - np_cholesky))
        print(f"Maximum difference from NumPy Cholesky: {cholesky_diff}")
    except np.linalg.LinAlgError:
        print("NumPy's Cholesky decomposition also failed.")

    # 检查重构矩阵的正定性
    reconstructed_eigenvalues = np.linalg.eigvals(reconstructed)
    if np.all(reconstructed_eigenvalues > 0):
        print("Reconstructed matrix is positive definite.")
    else:
        print("Warning: Reconstructed matrix is not positive definite.")



import time
# from serverless_framework import WorkerMetadata
from faasit_runtime import function, create_handler, FaasitRuntime

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

handler = create_handler(workflow_split)