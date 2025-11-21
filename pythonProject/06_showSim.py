import json
import numpy as np
from tqdm import tqdm
import random
import os

# ========= 配置部分 =========

# 8 个大类的目录（每个目录下面有一个 metadata.jsonl）
CATEGORY_DIRS = [
    "/home/sduu39/zHongchang/data/label/1_Computing Systems",  # TODO: 改成真实路径
    "/home/sduu39/zHongchang/data/label/2_Theoretical Computer Science",
    "/home/sduu39/zHongchang/data/label/3_Computer Networks & Wireless Communication",
    "/home/sduu39/zHongchang/data/label/4_Computer Graphics",
    "/home/sduu39/zHongchang/data/label/5_Human Computer Interaction",
    "/home/sduu39/zHongchang/data/label/6_Computational Linguistics",
    "/home/sduu39/zHongchang/data/label/7_Computer Vision & Pattern Recognition",
    "/home/sduu39/zHongchang/data/label/8_Databases & Information Systems",
]

METADATA_NAME = "metadata.jsonl"

# 抽样上限：> 这个数就随机抽样这么多节点
MAX_NODES_PER_CLASS = 100_000  # 如果内存吃不消，可以改成 80_000

# 为了结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def load_embeddings(metadata_path: str) -> np.ndarray:
    """从一个 metadata.jsonl 中读取所有 embedding，返回 (N, D) 的 float32 数组"""
    sample_vectors = []
    with open(metadata_path, "r", encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {metadata_path}")):
            data = json.loads(line.strip())
            sample_vectors.append(np.array(data["embedding"], dtype=np.float32))

    sample_vectors = np.array(sample_vectors, dtype=np.float32)
    print(f"[INFO] Loaded {sample_vectors.shape[0]} vectors, dim = {sample_vectors.shape[1]}")
    return sample_vectors


def compute_and_save_pairwise_sim(sample_vectors: np.ndarray, out_path: str):
    """
    给定一个类的所有（或抽样后）向量，计算全部 pairwise 相似度（上三角，不含对角线），并保存为 npy。
    """
    # 如果你的向量已经是单位向量，这里可以不归一化。
    # 如果不确定，可以打开这一段做一次归一化：
    # norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True) + 1e-12
    # sample_vectors = sample_vectors / norms

    SUBSET_SIZE = len(sample_vectors)
    print(f"[INFO] SUBSET_SIZE = {SUBSET_SIZE}")

    if SUBSET_SIZE > 80_000:
        print("[WARN] SUBSET_SIZE > 80,000，计算 sim_mat 可能会非常吃内存，请确认机器内存够用。")

    # 全量相似度矩阵：sim_mat[i,j] = 向量 i 与 向量 j 的内积（如果归一化，就是余弦相似度）
    print("[INFO] Start computing sim_mat = X @ X.T ...")
    sim_mat = sample_vectors @ sample_vectors.T

    # 只取上三角（不含对角线），每一对(i,j)只保留一次
    row_idx, col_idx = np.triu_indices(SUBSET_SIZE, k=1)
    vals = sim_mat[row_idx, col_idx].astype(np.float32)   # 形状是 (N*(N-1)/2, )

    print("相似度对数 =", vals.shape[0])

    np.save(out_path, vals)
    print(f"[OK] 已保存到 {out_path}")


def process_one_category(cat_dir: str):
    """处理一个大类：读 metadata.jsonl，抽样（如有必要），计算并保存 pairwise 相似度"""
    metadata_path = os.path.join(cat_dir, METADATA_NAME)
    if not os.path.exists(metadata_path):
        print(f"[WARN] {metadata_path} 不存在，跳过该类")
        return

    print(f"\n========== 处理大类：{cat_dir} ==========")
    vectors = load_embeddings(metadata_path)
    N = len(vectors)

    # 抽样逻辑：大于 100000 的类，在本类内部随机抽样 100000 个
    if N > MAX_NODES_PER_CLASS:
        print(f"[INFO] 本类共有 {N} 个节点，大于 {MAX_NODES_PER_CLASS}，开始随机抽样...")
        idx = np.random.choice(N, size=MAX_NODES_PER_CLASS, replace=False)
        vectors = vectors[idx]
        print(f"[INFO] 抽样后节点数 = {len(vectors)}")
        suffix = f"sample{MAX_NODES_PER_CLASS}"
    else:
        print(f"[INFO] 本类节点数 {N} ≤ {MAX_NODES_PER_CLASS}，全量计算。")
        suffix = f"all{N}"

    out_path = os.path.join(cat_dir, f"pairwise_subset_similarities_{suffix}.npy")
    compute_and_save_pairwise_sim(vectors, out_path)


def main():
    for cat_dir in CATEGORY_DIRS:
        process_one_category(cat_dir)


if __name__ == "__main__":
    main()
