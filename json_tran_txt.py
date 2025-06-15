import json
import re

json_file = 'subtrain.json'
txt_file = 'subtrain.txt'


def convert_json_to_txt(json_path, txt_path):
    with open(json_path, 'r', encoding='utf-8') as f_json:
        data = json.load(f_json)

    lines = []
    # 正则表达式用于清洗[END]和多余空格
    pattern = re.compile(r'\s*\[END\]\s*|\s*\|\s*')  # 匹配[END]和管道符前后空格

    for item in data:
        content = item['content']
        output = item['output']

        # 1. 彻底移除[END]并清洗字符串
        cleaned_output = re.sub(r'\[END\]', '', output).strip()

        # 2. 处理多条目分隔（[SEP]）
        sub_entries = [s.strip() for s in cleaned_output.split('[SEP]') if s.strip()]

        entries = []
        for sub in sub_entries:
            # 3. 按|分割并过滤空值
            parts = [p for p in sub.split('|') if p.strip()]
            if len(parts) >= 4:  # 确保至少包含4个有效字段
                # 提取前4个字段并去除首尾空格
                entity = parts[0].strip()
                text = parts[1].strip()
                label = parts[2].strip()
                sentiment = parts[3].strip()
                entries.append([entity, text, label, sentiment])

        # 4. 构建标准格式行
        if entries:
            entry_str = str(entries).replace("'", '"')  # 使用双引号
            line = f"{content}####{entry_str}"
            lines.append(line)

    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write('\n'.join(lines))


if __name__ == '__main__':
    convert_json_to_txt(json_file, txt_file)
    print(f"已完成转换，确保无[END]残留并生成{txt_file}")