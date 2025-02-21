import random
import string
import os
# 定义生成随机单词的函数
def generate_random_word(min_length=3, max_length=10):
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# 目标文件大小（以字节为单位）
target_size = 10 * 1024 * 1024  # 100MB

# 初始化一个空字符串用于存储生成的单词
current_size = 0

with open('data/random_words.txt', 'w', encoding='utf-8') as file:
    while current_size < target_size:
        word = generate_random_word()
        # 如果不是第一个单词，添加一个空格
        current_size = os.path.getsize('random_words.txt')
        file.write(word)
        file.write(' ')

print("已生成 10MB 的随机单词并保存到 random_words.txt 文件中。")