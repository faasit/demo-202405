## 演示流程

### 介绍整个演示的界面

在本次演示中，我们搭建了在线编辑工具 vscode

在该工具中准备了若干演示的示例

每个演示实例都是基于我们设计的 Runtime SDK 进行开发，并使用部署描述进行统一的管理，可以部署到阿里云、Knative 等多个平台

### 演示`Helloworld`

在 Hello World 例子中，我们展示了一个最简单的函数是如何进行开发和部署的

首先进入目录 `demo/helloworld`

- 切换到目录`demo/helloworld`

```bash
cd ~/projects/demo/helloworld
```

查看函数的代码文件，通过 faasit_runtime 库，我们可以声明 Serverless 函数，并指定函数的输入和输出。Runtime 库的优势在于，它统一了不同云平台在 Serverless 函数开发方面的差异性，使得我们可以用一致的方式来编写和部署函数。

- 查看`code/index.py`代码文件

```
code ~/projects/demo/helloworld/code/index.py
```

接着我们查看该函数的部署描述 `main.ft`，在部署描述中，我们定义了 hello world 函数的部署细节，例如编程语言类型、代码文件路径，和部署的目标平台，例如阿里云和 knative

利用该部署描述，我们可以使用 faasit 工具将函数部署到多个平台中

- 查看`main.ft`文件

```
code ~/projects/demo/helloworld/main.ft
```

部署到 阿里云平台

- 部署到`aliyun`上

```bash
ft deploy -p Aliyun
```

部署到 knative 平台

- 部署到`knative`上

```bash
ft deploy -p Knative
```

可以看到，该函数在两个平台都成功部署了，我们可以使用 ft invoke 命令调用函数，验证函数是否能够正常工作

调用在阿里云上部署的函数

- 调用阿里云上的函数

```bash
ft invoke -p Aliyun
```

调用在 Knative 上部署的函数

- 调用`kn`上的函数

```bash
ft invoke -p Knative
```

上面展示了一个最基本的函数开发和部署流程，下面我们来看一个稍微复杂的例子，该例子支持多函数编排和状态持久化功能。

### 演示`demo-durable-py`

切换到目录 `demo-durable-py`

- 切换到目录`demo/demo-durable-py`

```bash
cd ~/projects/demo/demo-durable-py
```

查看多函数编排与状态持久化例子的代码文件 `code/index.py`

- 查看`code/index.py`文件

```bash
code ~/projects/demo/demo-durable-py/code/index.py
```

在代码中，我们定义了多个函数，并引入了 workflow 装饰器描述多个函数的编排过程，引入了 durable 描述函数的状态持久化功能。通过 frt.call 我们实现了函数的同步与异步调用

尝试运行该函数，可以看到这是我们基于录制回放模型实现持久化功能的运行记录，通过录制回放模型，确保了长时运行函数编排的可靠性

- 运行

```bash
ft run
```

我们也已经开展了部分与其他课题的集成，例如北京大学的例子已经可以用我们的开发模型进行开发，这里以 MovieReview 为例。

### 集成

MovieReview 实现了 7 个 Serverless 函数形成了一个函数工作流，实现了一个影片评测的 Web 项目。运行该函数，可以看到，工作流能够正常运行，并最终输出结果

```bash
cd ~/projects/demo/demo-MovieReview
ft run
```
