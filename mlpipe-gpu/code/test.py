import io
import os
import json
import time
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# from serverless_framework import WorkerMetadata
from faasit_runtime import function, FaasitRuntime


class S3ImageDataset(Dataset):
    def __init__(self, image_files, transform=None, num_classes=2):
        self.image_files = image_files
        self.transform = transform
        # TODO: change to real labels
        self.labels = [i % num_classes for i in range(len(image_files))]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

@function
def lambda_handler(frt: FaasitRuntime):
    start_time = time.time()
    params: Dict = frt.input()
    store = frt.storage

    input_params = params['input']
    output_params = params['output']
    # This stage currently does not support multiple function instances;
    # all data is consumed here
    test_images_prefix = input_params['test_images_prefix']
    test_images_offset = input_params['test_images_offset']
    test_images_count = input_params['test_images_count']
    num_classes = input_params.get('num_classes', 10)
    dp = torch.cuda.device_count()
    model_key = input_params['model']

    # Output
    metrics_output_key = output_params['metrics']

    # ------------------------
    # Stage: fetch inputs
    # ------------------------
    tmp_dir = '/tmp/test_images'
    os.makedirs(tmp_dir, exist_ok=True)
    local_image_paths = []

    # Fetch test images from stage0
    for i in range(test_images_offset, test_images_offset+test_images_count):
        key = f"{test_images_prefix}_{i}.png"
        stage_list = ["stage0"] + [f"stage0{idx}" for idx in range(100)]
        image_bytes = None
        for stage in stage_list:
            try:
                image_bytes = store.get(key, src_stage=stage)
                # image_bytes = md.get_object(stage, key)
                if image_bytes is not None:
                    break
            except KeyError:
                continue
        local_path = os.path.join(tmp_dir, os.path.basename(key))
        with open(local_path, "wb") as f:
            f.write(image_bytes)
        local_image_paths.append(local_path)

    # Fetch trained model bytes from stage1
    model_bytes = store.get(model_key, src_stage='stage10')
    # model_bytes = md.get_object('stage1-0', model_key)
    input_end = time.time()

    # ------------------------
    # Stage: build dataloader
    # ------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    assert torch.cuda.is_available() 
    ngpu = torch.cuda.device_count()
    assert dp <= ngpu
    device_ids = list(range(ngpu))[-dp:]
    device = torch.device(f'cuda:{device_ids[0]}')

    base_batch = 4
    effective_batch = max(base_batch * max(1, ngpu), base_batch)

    dataloader = DataLoader(
        S3ImageDataset(local_image_paths, transform=transform, num_classes=num_classes),
        batch_size=effective_batch,
        shuffle=False,
        num_workers=min(os.cpu_count() or 2, 4),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ------------------------
    # Stage: build & load model
    # ------------------------
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Load the state dict from bytes
    state_buf = io.BytesIO(model_bytes)
    state_dict = torch.load(state_buf, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Wrap in DataParallel if DP is enabled
    if dp > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"[train] enabling DP for {dp} GPUs")

    criterion = nn.CrossEntropyLoss().to(device)

    # ------------------------
    # Stage: evaluate
    # ------------------------
    model.eval()
    compute_start = time.time()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    # Confusion matrix (row = true, col = pred)
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[t.long(), p.long()] += 1

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    accuracy = (correct / total_samples) if total_samples > 0 else 0.0

    # Per-class accuracy
    per_class_acc = []
    conf_np = conf_mat.cpu().numpy()
    for cls in range(num_classes):
        cls_total = conf_np[cls].sum()
        cls_correct = conf_np[cls, cls]
        per_class_acc.append(float(cls_correct) / float(cls_total) if cls_total > 0 else 0.0)

    compute_end = time.time()
    output_end = time.time()

    # ------------------------
    # Stage: emit outputs
    # ------------------------
    metrics = {
        'device_ids': device_ids,
        'batch_size': effective_batch,
        'num_classes': num_classes,
        'samples': total_samples,
        'avg_loss': avg_loss,
        'accuracy_top1': accuracy,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': conf_np.tolist(),
        # 'timing': {
            'input_time': input_end - start_time,
            'compute_time': compute_end - compute_start,
            'output_time': output_end - compute_end,
            'total_time': output_end - start_time,
        # }
    }

    metrics_bytes = json.dumps(metrics).encode('utf-8')
    store.put(metrics_output_key, metrics_bytes, dest_stages=['stage20'])
    # md.output(['stage2-0'], metrics_output_key, metrics_bytes)

    return metrics

handler = lambda_handler.export()

# from dataclasses import dataclass
# @dataclass
# class MockMetadata:
#     params: Dict

#     def get_object(self, src_stage: Optional[str], key: str) -> Optional[Any]:
#         with open(f"../tmp/{src_stage}/{key}", "rb") as f:
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
#                 'test_images_prefix': 'processed_test_image',
#                 'test_images_offset': 300,
#                 'test_images_count': 100,
#                 'num_classes': 10,
#                 'model': 'resnet_model/model.pth'
#             },
#             'output': {
#                 'metrics': 'resnet_model/metrics.json'
#             }
#         }
#     )
#     output = lambda_handler(md)
#     print("Output:", output)
