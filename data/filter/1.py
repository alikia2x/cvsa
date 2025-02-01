import json
import random

def process_data(input_file, output_file):
    """
    从输入文件中读取数据，找出model和human不一致的行，
    删除"model"键，将"human"键重命名为"label"，
    然后将处理后的数据添加到输出文件中。
    在写入之前，它会加载output_file中的所有样本，
    并使用aid键进行去重过滤。

    Args:
        input_file (str): 输入文件的路径。
        output_file (str): 输出文件的路径。
    """

    # 加载output_file中已有的数据，用于去重
    existing_data = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    data = json.loads(line)
                    existing_data.add(data['aid'])
                except json.JSONDecodeError:
                    pass  # 忽略JSON解码错误，继续读取下一行
    except FileNotFoundError:
        pass  # 如果文件不存在，则忽略

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)

                if data['model'] != data['human'] or random.random() < 0.2:
                    if data['aid'] not in existing_data:  # 检查aid是否已存在
                        del data['model']
                        data['label'] = data['human']
                        del data['human']
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        existing_data.add(data['aid'])  # 将新的aid添加到集合中

            except json.JSONDecodeError as e:
                print(f"JSON解码错误: {e}")
                print(f"错误行内容: {line.strip()}")
            except KeyError as e:
                print(f"KeyError: 键 '{e}' 不存在")
                print(f"错误行内容: {line.strip()}")

# 调用函数处理数据
input_file = 'real_test.jsonl'
output_file = 'labeled_data.jsonl'
process_data(input_file, output_file)
print(f"处理完成，结果已写入 {output_file}")

