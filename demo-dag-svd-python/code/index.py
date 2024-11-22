from faasit_runtime import create_handler, workflow, Workflow
from faasit_runtime import FaasitRuntime, function, create_handler
import numpy as np

def block_matrix_concat(U_matrices):
    U1 = U_matrices[0]  
    for i in range(1, len(U_matrices)):
        U = U_matrices[i]
        subyup = np.zeros((U1.shape[0], U.shape[1]))
        x1 = np.hstack((U1, subyup))  
        subydown = np.zeros((U.shape[0], U1.shape[1]))
        subydowncom = np.hstack((subydown, U)) 
        U1 = np.vstack((x1, subydowncom))
    
    return U1

def merge(input_data):
    Xi_svds = input_data["Xi_svds"]
    U_tilde = block_matrix_concat([v["U"] for v in Xi_svds])

    Y = []
    for Xi_svd in Xi_svds:
        d = np.diag(Xi_svd["S"])  # 对角矩阵 S
        yi = np.dot(d, Xi_svd["Vt"].T)  # yi = S * V^T
        Y.append(yi)

    Y = np.concatenate(Y, axis=0)

    U_Y, S_Y, V_Y = np.linalg.svd(Y, full_matrices=False)
    U = np.dot(U_tilde, U_Y)
    S = S_Y
    V = V_Y

    return {"U": U.tolist(), "S": S.tolist(), "V": V.tolist()}


def split(input_data):
    X = input_data["X"]
    num_splits = input_data["numSplits"]

    m = len(X)
    row_size = m // num_splits
    sub_Xs = []

    for i in range(num_splits):
        start_idx = i * row_size
        end_idx = (i + 1) * row_size
        sub_Xs.append(X[start_idx:end_idx])

    return {
        "subXs": sub_Xs
    }

def compute(input_data):
    Xis = input_data["Xis"]
    result = []
    for Xi in Xis:
        U, S, Vt = np.linalg.svd(Xi,full_matrices=False)
        Xi_svd = {"U": U, "S": S, "Vt": Vt}  # 将 SVD 结果组织成一个字典
        result.append(Xi_svd)

    return {
        "result": result
    }

@function
def executor(frt: FaasitRuntime):
    input_data = frt.input()
    X = input_data["X"]
    num_splits = frt.input().get("numSplits", 3)
    split_res = split({"X": X, "numSplits": num_splits})
    sub_Xs = split_res["subXs"]
    compute_res = compute({"Xis": sub_Xs})
    Xi_svds = compute_res["result"]
    merge_res = merge({"Xi_svds": Xi_svds})
    return frt.output({
        "message": "ok",
        "data": merge_res
    })

@workflow
def ParallelSVD(wf: Workflow):
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    num_splits = 2
    res = wf.call("executor", {"X": X, "numSplits": num_splits})
    return res

handler = create_handler(ParallelSVD)

