from faasit_runtime import workflow, Workflow
import json
import os
@workflow
def mlworkflow(wf: Workflow):
    # cur_stage_cnt: 2
    # cur_stage_idx: 0
    # input:
    #   input_prefix: image      # 输入图像文件前缀，如 "image"，实际文件为 "image_0.png", "image_1.png", ...
    #   input_amount: 300        # 图像数量
    #   input_test_amount: 100   # 测试图像数量
    # output:
    #   output_prefix: processed_image   # 输出图像文件前缀，如 "processed_image_0.png"
    #   output_test_prefix: processed_test_image
    _in = json.loads(os.environ.get("FAASIT_PARAMS", "\{\}"))
    stage0 = {}
    if 'stage0' in _in:
        num = _in['stage0'] 
        for i in range(num):
            stage00 = wf.call(f"stage0-{i}", {
                "cur_stage_cnt": 2,
                "cur_stage_idx": i,
                "input": {
                    "input_prefix": "image",
                    "input_amount": 300,
                    "input_test_amount": 100
                },
                "output": {
                    "output_prefix": "processed_image",
                    "output_test_prefix": "processed_test_image"
                }
            })
            stage0[f"stage0-{i}"] = stage00
    else:
        stage00 = wf.call("stage0", {
            "cur_stage_cnt": 2,
            "cur_stage_idx": 0,
            "input": {
                "input_prefix": "image",
                "input_amount": 300,
                "input_test_amount": 100
            },
            "output": {
                "output_prefix": "processed_image",
                "output_test_prefix": "processed_test_image"
            }
        })
        stage0["stage0"] = stage00


    # input:
    #   preprocessed_images_prefix: processed_image    # 预处理后图像的前缀，与 stage0 output_prefix 保持一致
    #   preprocessed_images_count: 300                 # 图像数量，须与 stage0 input_amount 对齐
    #   num_classes: 10                                # 类别数
    #   epochs: 1                                      # 训练epoch数
    # output:
    #   model: resnet_model/model.pth                 # 训练后模型的输出路径
    stage1 = {}
    if 'stage1' in _in:
        num = _in['stage1'] 
        for i in range(num):
            stage10 = wf.call(f"stage1-{i}", {
                **stage0,
                "input": {
                    "preprocessed_images_prefix": "processed_image",
                    "preprocessed_images_count": 300,
                    "num_classes": 10,
                    "epochs": 1
                },
                "output": {
                    "model": f"resnet_model/model_{i}.pth"
                }
            })
            stage1[f"stage1-{i}"] = stage10
    else:
        stage10 = wf.call("stage1", {
            **stage0,
            "input": {
                "preprocessed_images_prefix": "processed_image",
                "preprocessed_images_count": 300,
                "num_classes": 10,
                "epochs": 1
            },
            "output": {
                "model": "resnet_model/model.pth"
            }
        })
        stage1["stage1"] = stage10

    # input:
    #   test_images_prefix: processed_test_image      # 预处理后图像的前缀，与 stage0 output_prefix 保持一致
    #   test_images_offset: 300                       
    #   test_images_count: 100                        # 图像数量，须与 stage0 input_test_amount 对齐
    #   num_classes: 10                               # 类别数
    #   model: resnet_model/model.pth                 # 训练后模型的输出路径
    # output:
    #   metrics: resnet_model/metrics.json            # 测试后性能数据的输出路径
    if 'stage2' in _in:
        num = _in['stage2'] 
        for i in range(num):
            stage20 = wf.call(f"stage2-{i}", {
                **stage0,
                **stage1,
                "input": {
                    "test_images_prefix": "processed_test_image",
                    "test_images_offset": 300,
                    "test_images_count": 100,
                    "num_classes": 10,
                    "model": f"resnet_model/model_{i}.pth"
                },
                "output": {
                    "metrics": f"resnet_model/metrics_{i}.json"
                }
            })
    else:
        stage20 = wf.call("stage2", {
            **stage0,
            **stage1,
            "input": {
                "test_images_prefix": "processed_test_image",
                "test_images_offset": 300,
                "test_images_count": 100,
                "num_classes": 10,
                "model": "resnet_model/model.pth"
            },
            "output": {
                "metrics": "resnet_model/metrics.json"
            }
        })

    return stage20

mlworkflow = mlworkflow.export()