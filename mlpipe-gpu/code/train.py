import json
import os

# from serverless_framework import WorkerMetadata
from faasit_runtime import function, FaasitRuntime
from torchvision import models
from PIL import Image
from typing import Dict, Optional, Any, List
import time

@function
def lambda_handler(frt: FaasitRuntime):
    # Ugly fix: must import torch after we set CUDA_VISIBLE_DEVICES
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
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
        
    start_time = time.time()
    params: Dict = frt.input()

    input_params = params['input']
    output_params = params['output']
    # This stage does not support multiple function instances;
    # all data is consumed here
    preprocessed_images_prefix = input_params['preprocessed_images_prefix']
    preprocessed_images_count = input_params['preprocessed_images_count']
    num_classes = input_params.get('num_classes', 10)
    epochs = input_params.get('epochs', 1)
    dp = torch.cuda.device_count()
    model_output_key = output_params['model']
    
    tmp_dir = '/tmp/train_images'
    os.makedirs(tmp_dir, exist_ok=True)
    local_image_paths = []
    
    input_end = time.time()
    store = frt.storage
    for i in range(preprocessed_images_count):
        key = f"{preprocessed_images_prefix}_{i}.png"
        stage_list = ["stage0"] + [f"stage0-{idx}" for idx in range(100)]
        for stage in stage_list:
            try:
                image_bytes = store.get(key, src_stage=stage)
                # image_bytes = md.get_object(stage, key)
                if image_bytes is not None:
                    break
            except KeyError:
                continue
        # raise ValueError("Image data not found")
        local_path = os.path.join(tmp_dir, os.path.basename(key))
        with open(local_path, "wb") as f:
            f.write(image_bytes)
        local_image_paths.append(local_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    assert torch.cuda.is_available() 
    ngpu = torch.cuda.device_count()
    assert dp <= ngpu
    device_ids = list(range(ngpu))[:dp]
    device = torch.device(f'cuda:{device_ids[0]}')

    # Scale batch size with number of GPUs when available
    base_batch_size = 4
    effective_batch_size = max(base_batch_size * max(1, ngpu), base_batch_size)

    dataset = S3ImageDataset(local_image_paths, transform=transform, num_classes=num_classes)
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count() or 2, 4),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # Optional speedup on fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Build model
    model = models.resnet18(weights=None)
    # Load weights before replacing the classifier head
    state = torch.load('resnet18-f37072fd.pth', map_location='cpu')
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Wrap in DataParallel if DP is enabled
    if dp > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"[train] enabling DP for {dp} GPUs")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    compute_start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    compute_end = time.time()

    # Save underlying module weights if wrapped
    model_path = '/tmp/model.pth'
    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, model_path)

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    store.put(model_output_key, model_bytes, dest_stages=['stage2-0'])
    # md.output(['stage2-0'], model_output_key, model_bytes)

    if os.path.exists(model_path):
        os.remove(model_path)
    for local_path in local_image_paths:
        if os.path.exists(local_path):
            os.remove(local_path)
    output_end = compute_end
    
    return {
        'device_ids': device_ids,
        'batch_size': effective_batch_size,
        'avg_loss': avg_loss,
        'input_time': input_end - start_time,
        'compute_time': compute_end - compute_start,
        'output_time': output_end - compute_end,
        'total_time': output_end - start_time,
    }

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
#                 'preprocessed_images_prefix': 'processed_image',
#                 'preprocessed_images_count': 300,
#                 'num_classes': 10
#             },
#             'output': {
#                 'model': 'resnet_model/model.pth'
#             }
#         }
#     )
#     output = lambda_handler(md)
#     print("Output:", output)
