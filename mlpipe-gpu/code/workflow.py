from faasit_runtime import workflow, Workflow

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
    stage00 = wf.call("stage00", {
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

    # cur_stage_cnt: 2
    # cur_stage_idx: 1
    # input:
    #   input_prefix: image      # 输入图像文件前缀，如 "image"，实际文件为 "image_0.png", "image_1.png", ...
    #   input_amount: 300        # 图像数量
    #   input_test_amount: 100   # 测试图像数量
    # output:
    #   output_prefix: processed_image   # 输出图像文件前缀，如 "processed_image_0.png"
    #   output_test_prefix: processed_test_image  
    stage01 = wf.call("stage01", {
        "cur_stage_cnt": 2,
        "cur_stage_idx": 1,
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

    # input:
    #   preprocessed_images_prefix: processed_image    # 预处理后图像的前缀，与 stage0 output_prefix 保持一致
    #   preprocessed_images_count: 300                 # 图像数量，须与 stage0 input_amount 对齐
    #   num_classes: 10                                # 类别数
    #   epochs: 1                                      # 训练epoch数
    # output:
    #   model: resnet_model/model.pth                 # 训练后模型的输出路径
    stage10 = wf.call("stage10", {
        "stage0-0": stage00,
        "stage0-1": stage01,
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

    # input:
    #   test_images_prefix: processed_test_image      # 预处理后图像的前缀，与 stage0 output_prefix 保持一致
    #   test_images_offset: 300                       
    #   test_images_count: 100                        # 图像数量，须与 stage0 input_test_amount 对齐
    #   num_classes: 10                               # 类别数
    #   model: resnet_model/model.pth                 # 训练后模型的输出路径
    # output:
    #   metrics: resnet_model/metrics.json            # 测试后性能数据的输出路径
    stage20 = wf.call("stage20", {
        "stage0-0": stage00,
        "stage0-1": stage01,
        "stage1-0": stage10,
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