import random
import string
import os
import tqdm
# 定义生成随机单词的函数
def generate_random_word(min_length=3, max_length=10):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# 目标文件大小（以字节为单位）
target_size = 10 * 1024 * 1024  # 100MB

# 初始化一个空字符串用于存储生成的单词

for i in range(4):
    current_size = 0
    with open(f'stage0-{i}-input', 'w', encoding='utf-8') as file:
        for _ in tqdm.tqdm(range(50000), desc=f"Generating stage0-{i}-input"):
            word = generate_random_word()
            file.write(word + ' ')
