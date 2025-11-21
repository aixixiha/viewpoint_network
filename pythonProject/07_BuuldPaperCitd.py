import json
import networkx as nx
import pickle
from pathlib import Path

paperL1 = "/home/sduu39/zHongchang/data/label/paperLabel1.jsonl"

# 你希望保留到节点上的属性字段（按需增删）
NODE_FIELDS = [
    "index", "title", "year", "authors", "venue",
    "label", "references"
]
G = nx.Graph()  # 如果你想无向图，用 nx.Graph()
# 第一步：单次扫描，收集所有“合法论文”的节点属性 & 引用列表
valid_nodes = {}          # index -> 属性字典
out_edges = {}            # index -> list(references)

with open(paperL1, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line.strip())
            index = data.get("index")
            if not index:
                continue  # 没有 index 的直接跳过

            # 取出你关心的字段作为节点属性（过滤掉 None）
            attrs = {k: data.get(k) for k in NODE_FIELDS}

            # 也可以做一些清洗或类型转换
            # 例如：如果 index 是数字字符串，转成 int（可选）
            try:
                attrs["index"] = int(index)
                index_key = attrs["index"]
            except Exception:
                print("---")  # 保持原样

            G.add_node(index_key, **attrs)


        except Exception as e:
            print(f"警告：第{line_num}行处理失败，错误：{str(e)}")

print("123")
# 第二步：构图（只在 valid_nodes 中的节点进行连边）

missing_ref = 0
kept_edges = 0
with open(paperL1, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line.strip())
            index = data.get("index")
            if index is None:
                continue
            try:
                src = int(index)
            except Exception:
                src = str(index)

            if src not in G:
                continue  # 第1遍未入图的节点直接跳过

            refs = data.get("references") or []
            # 简单去重，避免重复边
            seen = set()
            for r in refs:
                try:
                    dst = int(r)
                except Exception:
                    dst = str(r)
                if dst in seen:
                    continue
                seen.add(dst)

                if dst in G:              # 只连“有效”引用
                    G.add_edge(src, dst)  # A -> B 表示 A 引用 B
                    kept_edges += 1
                else:
                    missing_ref += 1
        except Exception as e:
            print(f"警告：第{line_num}行处理失败，错误：{e}")

print(f"完成：节点 {G.number_of_nodes()}，边 {G.number_of_edges()}（过滤掉无效引用 {missing_ref}，保留 {kept_edges}）")

# 保存（gpickle 会保留所有节点/边属性）
with open("/home/sduu39/zHongchang/data/networkx/paperL1_graph.gpickle", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)