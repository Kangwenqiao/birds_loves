import json
from sklearn.model_selection import train_test_split

# 逐行读取 JSON 文件
with open('llama_data.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 划分数据集，保存为新的 JSON 文件
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

with open('train_data.json', 'w', encoding='utf-8') as train_file:
    for entry in train_data:
        json.dump(entry, train_file, ensure_ascii=False)
        train_file.write('\n')

with open('val_data.json', 'w', encoding='utf-8') as val_file:
    for entry in val_data:
        json.dump(entry, val_file, ensure_ascii=False)
        val_file.write('\n')
