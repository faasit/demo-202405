## 演示流程

### 演示`Helloworld`

- 切换到目录`demo/helloworld`

```bash
cd ~/projects/demo/helloworld
```

- 查看`code/index.py`代码文件
- 查看`main.ft`文件
- 部署到`knative`上

```bash
ft deploy -p Knative
```

- 部署到`aliyun`上

```bash
ft deploy -p Aliyun
```

- 调用阿里云上的函数

```bash
ft invoke -p Aliyun
```

- 调用`kn`上的函数

```bash
kubectl get ksvc # 查看部署状态
ft invoke -p Knative
```


### 演示`SVD`

- 切换到目录`demo/demo-dag-svd`

```bash
cd ~/projects/demo/demo-dag-svd
```

- 查看`javascript/functions.js`以及`javascript/index.js`代码文件
- 运行

```bash
ft run
```

### 演示`txn`

- 切换到目录`demo/demo-txn`

```bash
cd ~/projects/demo/demo-txn
```

- 查看`javascript/functions.js`文件以及`javascript/index.js`文件
- 运行

```bash
ft run
```

### 演示`demo-durable-py`

- 切换到目录`demo/demo-durable-py`

```bash
cd ~/projects/demo/demo-durable-py
```

- 查看`code/index.py`文件
- 运行

```bash
ft run
```

### 集成

```bash
cd ~/projects/demo/demo-BigDataBench
ft run

cd ~/projects/demo/demo-chatbot
ft run

cd ~/projects/demo/demo-MovieReview
ft run
```