import os
import socket
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
# from flask import Flask, jsonify
import sys
import threading
import logging
from faasit_runtime import function, FaasitRuntime, workflow, Workflow

# 配置日志
log_file = "app.log"

# 创建日志记录器
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建文件处理器（写入日志到文件）
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # 设定最低日志级别

# 创建控制台处理器（输出日志到终端）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设定最低日志级别

# 创建日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 配置参数
REPLICATION = 5  # 数据复制次数
DATA_SPLIT = 0.6  # 训练测试分割比例


def download_data():
    logger.info('fetching data....')
    data_path = 'mnist_data.npz'
    """Download a small subset of the MNIST dataset using scikit-learn"""
    if os.path.exists(data_path):
        logger.info('Hitting cache. Loading data from local file...')
        X = np.load('mnist_X.npy')
        y = np.load('mnist_y.npy')
    else:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser="pandas")  # Fetch MNIST dataset
        X, y = mnist.data, mnist.target.astype(np.int32)  # Ensure labels are integers
        logger.info('data prepared')
    # Reshape data to match the original format (28x28 images)
    X = X.reshape(-1, 28, 28)

    return {'X': X, 'y': y}


def preprocess_data(data):
    """预处理数据（仍使用小数据）"""
    X = data['X'].reshape((-1, 28 * 28)).astype('float32') / 255.0
    y = data['y']
    split = int(DATA_SPLIT * len(X))
    return {'X':X[:split], 'y':y[:split]}, {'X':X[split:], 'y':y[split:]}


def enhanced_send(sock, data):
    """带长度前缀的发送函数"""
    serialized = pickle.dumps(data)
    header = len(serialized).to_bytes(4, 'big')
    sock.sendall(header + serialized)


def stream_receive(conn):
    """流式接收函数（返回第一个有效数据）"""
    first_data = None
    total_received = 0

    while True:
        header = conn.recv(4)
        if not header: break
        length = int.from_bytes(header, 'big')
        logger.info(f"required length is ({length} bytes)")
        chunks = []
        bytes_received = 0

        while bytes_received < length:
            chunk = conn.recv(min(4096, length - bytes_received))
            if not chunk: break
            chunks.append(chunk)
            bytes_received += len(chunk)
        logger.info(f'receiving...')

        data = pickle.loads(b''.join(chunks))
        total_received += length

        if first_data is None:
            first_data = data
            logger.info(f"Received first payload ({length} bytes)")

    logger.info(f"Total received: {total_received / 1024 / 1024:.2f} MB")
    return first_data

# 在文件顶部新增 Flask 应用
# app = Flask(__name__)
trigger_lock = threading.Lock()  # 防止并发触发
def sender_thread(host, port, data_type, data):
    """发送线程（带数据复制）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(60)
        try:
            s.connect((host, port))
            for _ in range(REPLICATION):
                logger.info(f'sending times {_} data length {len(data)}')
                enhanced_send(s, {'type': data_type, 'data': data})
        except socket.timeout:
            logger.info("Socket timeout, aborting...")
        except Exception as e:
            logger.info(f"Error occurred during sending: {e}")
        finally:
            logger.info('sending end...')



def trigger_download(raw_data):
    logger.info('Download triggered via HTTP, starting data send...')
    sender_thread(
        os.environ['NEXT_HOST'],
        int(os.environ['NEXT_PORT']),
        'raw_data',
        raw_data
    )
    trigger_lock.release()

@function
def download_handler(frt: FaasitRuntime):
    raw_data = download_data()
    if trigger_lock.acquire(blocking=False):
        frt.call('preprocess', {'raw_data': raw_data})
        # threading.Thread(target=trigger_download, args=(raw_data,)).start()
        trigger_lock.release()
        return frt.output({"status": "started"})
    else:
        return frt.output({"status": "already running"}, 429)

@function
def preprocess_handler(frt: FaasitRuntime):
    store = frt.storage
    data = store.get('raw_data', active_pull=False, src_stage='download')
    (train_data, test_data) = preprocess_data(data)
    frt.call("train", {"train_data": train_data})
    frt.call("test", {"test_data": test_data})
    return frt.output({"status": "started"})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 启用端口重用
    s.bind(('0.0.0.0', int(os.environ['LISTEN_PORT'])))
    s.listen()
    while True:
        try:
            conn, _ = s.accept()
            received = stream_receive(conn)

            if received is None:
                continue
            (train_data, test_data) = preprocess_data(received['data'])
            logger.info('preprocess is over. sending data....')

            train_sending = threading.Thread(target=sender_thread,
                             args=(os.environ['TRAIN_HOST'],
                                   int(os.environ['TRAIN_PORT']),
                                   'train_data', train_data))

            test_sending = threading.Thread(target=sender_thread,
                             args=(os.environ['TEST_HOST'],
                                   int(os.environ['TEST_PORT']),
                                   'test_data', test_data))
            train_sending.start()
            test_sending.start()
            train_sending.join()
            test_sending.join()
        except Exception as e:
            logger.info(f"Error occurred during receiving: {e}")
            continue

@function
def train_handler(frt: FaasitRuntime):
    store = frt.storage
    data = store.get('train_data',active_pull=False, src_stage='proprocess')
    X = data['X']
    y = data['y']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    serialized_model = pickle.dumps(model)
    frt.call("test", {"model": serialized_model})
    return frt.output({"status": "started"})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 启用端口重用
    s.bind(('0.0.0.0', int(os.environ['LISTEN_PORT'])))
    s.listen()
    while True:
        try:
            conn, _ = s.accept()
            received = stream_receive(conn)
            if received is None:
                continue
            model = LogisticRegression(max_iter=1000)
            model.fit(received['data']['X'], received['data']['y'])

            # Serialize the trained model
            serialized_model = pickle.dumps(model)
            logger.info(f"Serialized model size: {len(serialized_model)} bytes")

            testing_send = threading.Thread(target=sender_thread,
                             args=(os.environ['TEST_HOST'],
                                   int(os.environ['MODEL_PORT']),
                                   'model', serialized_model))
            testing_send.start()
            testing_send.join()
        except Exception as e:
            logger.info(f"Error occurred during receiving: {e}")
            continue

@function
def test_handler(frt: FaasitRuntime):
    store = frt.storage
    results = {}

    results['model'] = store.get('model',active_pull=False, src_stage='train')
    results['test'] = store.get('test_data',active_pull=False, src_stage='preprocess')
    model = pickle.loads(results['model'])
    accuracy = model.score(results['test']['X'], results['test']['y'])
    return frt.output({"accuracy": accuracy})
    while True:
        results = {}
        lock = threading.Lock()

        def receive_model():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', int(os.environ['MODEL_PORT'])))
                s.listen()
                conn, _ = s.accept()
                data = stream_receive(conn)
                with lock:
                    if data:
                        results['model'] = data['data']

        def receive_test():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', int(os.environ['TEST_PORT'])))
                s.listen()
                conn, _ = s.accept()
                data = stream_receive(conn)
                with lock:
                    if data:
                        results['test'] = data['data']

        # 启动接收线程并等待完成
        threads = [
            threading.Thread(target=receive_model),
            threading.Thread(target=receive_test)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        if 'model' in results.keys() and 'test' in results.keys():
            if results['test'] and results['model']:

                serialized_model = results['model']
                # Deserialize the model
                model = pickle.loads(serialized_model)
                logger.info("Model deserialization successful.")
            # 计算准确率
                accuracy = model.score(
                    results['test']['X'],
                    results['test']['y']
                )

download = download_handler.export()
preprocess = preprocess_handler.export()
train = train_handler.export()
test = test_handler.export()

@workflow
def mlpipe(wf: Workflow):
    s0 = wf.call("download",{})
    s1 = wf.call("preprocess", {"s0": s0})
    s2 = wf.call("train", {"s1": s1})
    s3 = wf.call("test", {"s2": s2, "s1": s1})
    return s3

mlpipe = mlpipe.export()
# if __name__ == "__main__":
#     role = os.environ['ROLE']

#     if role == 'download':
#         download_handler()
#     elif role == 'preprocess':
#         preprocess_handler()
#     elif role == 'train':
#         train_handler()
#     elif role == 'test':
#         test_handler()
#     else:
#         sys.exit("Invalid role specified")