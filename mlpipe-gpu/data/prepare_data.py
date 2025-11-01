import os
from PIL import Image
import numpy as np

def generate_dummy_images():
    for i in range(1024):
        # 生成 224x224 的随机 RGB 图像
        data = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        image = Image.fromarray(data)
        image.save(f"./image_{i}.png")
    print("Dummy images generated in 'input_images/'.")

if __name__ == "__main__":
    generate_dummy_images()