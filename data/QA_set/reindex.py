import json

if __name__ == '__main__':
    file_path = '/Users/aaroncui/Desktop/UCL/NLP/NLP_project/data/QA_set/easy_single.json'
    # 读取json文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 重新索引
    for i, item in enumerate(data):
        item['id'] = i + 1

    # 写入新的json文件
    with open('data/QA_set/easy_single_reindexed.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)