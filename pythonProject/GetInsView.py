input_file = "/home/sduu39/zHongchang/data/v4_107view.jsonl"
output_file = "/home/sduu39/zHongchang/data/v4_107viewInsAbstract.jsonl"
# %%
import json
import argparse
from collections import deque

# %%
total = 0
longAbstract = 0
repeatCount = 0
written = 0
seen_set = set()
seen_deque = deque()
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception as e:
            print(line)
            print(e)
            print('------------------------------------------------------')

        val = obj.get('abstract', None)
        view=obj.get('viewpoints', None)
        if val is None:
            continue
        else:
            val = val.strip()

        if not view.endswith(']'):
            print(val)
            longAbstract += 1
            continue

        if val in seen_set:
            repeatCount += 1
            continue

        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        written += 1
        seen_deque.append(val)
        seen_set.add(val)
        if len(seen_deque) > 1000:
            oldest = seen_deque.popleft()
            seen_set.discard(oldest)

        if total % 1000 == 0:
            print(f"处理了{total}行，写入{written}行，重复的有{repeatCount}行，过长的有{longAbstract}行")

    print(total)
    print(written)
    print(repeatCount)
    print(longAbstract)