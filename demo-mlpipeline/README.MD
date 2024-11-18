## demo-mlpipe

### 环境准备

- 将目录切换到`demo-mlpipeline`

```sh
cd demo-mlpipeline
```

- 创建`python`虚拟环境

```sh
python -m venv venv
source venv/bin/activate
```

- 安装`faasit_runtime`

```sh
cd faasit/faasit-runtime/faasit-python-runtime
pip install .
```

- 创建一个`packages`目录，将所有运行时需要的包`install`进去

```sh
mkdir -p packages
pip install ... -t packages
```


### 部署运行

- 部署描述文件

见`main.ft`

- 部署到`spilot`

```sh
ft deploy
```

- 运行

```sh
ft invoke
```




### TODO LIST

- [ ] 写一个脚本自动打包所有的库到`packages`文件夹中，该文件夹装入应用所有需要的运行时库
- [ ] 将`serverless-framework`集成到运行时中
- [ ] `upload.py`作为常量藏在部署文件里头