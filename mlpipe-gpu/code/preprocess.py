import json
import io
import os
import torch
import torchvision.transforms as transforms
# from serverless_framework import WorkerMetadata
from faasit_runtime import FaasitRuntime, function
from PIL import Image
from typing import Dict, Optional, Any, List
import time

@function
def lambda_handler(frt: FaasitRuntime):
    start_time = time.time()
    params: Dict = frt.input()

    cur_stage_cnt = params.get('cur_stage_cnt', 1)
    cur_stage_idx = params.get('cur_stage_idx', 0)
    input = params['input']
    output = params['output']
    # 从参数中读取文件前缀、文件数量和输出前缀
    input_prefix = input['input_prefix']
    input_amount = input['input_amount']
    input_test_amount = input['input_test_amount']
    output_prefix = output['output_prefix']
    output_test_prefix = output['output_test_prefix']

    input_range_begin = (input_amount // cur_stage_cnt) * cur_stage_idx
    input_range_end = input_amount \
        if cur_stage_idx == cur_stage_cnt - 1 \
        else input_range_begin + input_amount // cur_stage_cnt
    test_range_begin = input_amount + (input_test_amount // cur_stage_cnt) * cur_stage_idx
    test_range_end = input_amount + input_test_amount \
        if cur_stage_idx == cur_stage_cnt - 1 \
        else test_range_begin + input_test_amount // cur_stage_cnt

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 记录输入处理完成时的时间
    end_input = time.time()

    # 根据输入前缀和文件数量构造完整的文件名，文件格式为 "prefix_id.png"
    # 对每个图像执行预处理，并将预处理后的图像上传至下游
    store = frt.storage
    def process(range_begin: int, range_end: int, out_prefix: str):
        results = []
        for i in range(range_begin, range_end):
            # 构造输入图像的 key，例如 "input_prefix + i.png"
            key = f"{input_prefix}_{i}.png"
            # 从原始（raw）阶段获取图像字节
            image_bytes = store.get(key)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = transform(image)
            processed_tensor = image_tensor
            processed_image = transforms.ToPILImage()(processed_tensor)
            
            # 将预处理后的图像保存在临时文件中
            tmp_filename = f"/tmp/processed_{os.path.basename(key)}"
            processed_image.save(tmp_filename)
            
            # 读取临时文件获取处理后图像的字节
            with open(tmp_filename, "rb") as f:
                processed_bytes = f.read()

            # 提取文件中的 id 和后缀，根据 “prefix_id.png” 格式拆分文件名
            basename = os.path.basename(key)
            parts = basename.split('_', 1)
            if len(parts) == 2:
                id_and_ext = parts[1]
            else:
                id_and_ext = basename
            # 构造输出 key，格式为 out_prefix + id_and_ext
            output_key = out_prefix + '_' + id_and_ext
            results.append((output_key, processed_bytes))
        return results
    
    results = process(input_range_begin, input_range_end, output_prefix)
    test_results = process(test_range_begin, test_range_end, output_test_prefix)
    print("[preprocess] process done")
    
    end_compute = time.time()

    for output_key, processed_bytes in results:
        # 将处理后的图像通过 md.output 上传到下游（stage1 输入）
        store.put(output_key, processed_bytes, dest_stages=['stage1-0'])
        # md.output(['stage1-0'], output_key, processed_bytes)
    for output_key, processed_bytes in test_results:
        # 将处理后的图像通过 md.output 上传到下游（stage2 输入）
        store.put(output_key, processed_bytes, dest_stages=['stage2-0'])
        # md.output(['stage2-0'], output_key, processed_bytes)

    end_output = end_compute  # 如果还有独立输出阶段此处再单独记录

    return {
        'input_time': end_input - start_time,
        'compute_time': end_compute - end_input, 
        'output_time': end_output - end_compute, 
        'total_time': end_output - start_time
    }

handler = lambda_handler.export()

# from dataclasses import dataclass
# @dataclass
# class MockMetadata:
#     params: Dict

#     def get_object(self, _src_stage: Optional[str], key: str) -> Optional[Any]:
#         with open(f"../../Redis/preload/mlpipe-gpu/{key}", "rb") as f:
#             return f.read() 
    
#     def output(self, dest_stages: List[Optional[str]], key: str, obj: Any):
#         for stage in dest_stages:
#             os.makedirs(f"../tmp/{stage}", exist_ok=True)
#             with open(f"../tmp/{stage}/{key}", "wb") as f:
#                 f.write(obj)

# if __name__ == "__main__":
#     md = MockMetadata(
#         params={
#             'input': {
#                 'input_prefix': 'image',
#                 'input_amount': 300,
#                 'input_test_amount': 100,
#             },
#             'output': {
#                 'output_prefix': 'processed_image',
#                 'output_test_prefix': 'processed_test_image',
#             }
#         }
#     )
#     output = lambda_handler(md)
#     print(output)