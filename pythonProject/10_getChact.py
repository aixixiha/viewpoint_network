import os
import csv
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.csgraph import connected_components


# 兼容两种 npz 格式的加载函数
def load_adj_any(path):
    """
    同时支持：
    1) scipy.sparse.save_npz 保存的稀疏矩阵
    2) 你之前 fallback 的 rows/cols/data/shape 三元组格式
    返回 csr_matrix
    """
    try:
        # 优先按 SciPy 官方格式读
        adj = load_npz(path)
        return adj
    except Exception:
        # 如果不是 SciPy 格式，尝试读取 rows/cols/data/shape
        arr = np.load(path)
        files = set(arr.files)
        if {"rows", "cols", "data", "shape"} <= files:
            rows = arr["rows"]
            cols = arr["cols"]
            data = arr["data"]
            shape = tuple(arr["shape"])
            adj = csr_matrix((data, (rows, cols)), shape=shape)
            return adj
        else:
            raise ValueError(f"{path} 的 npz 格式不认识，字段: {arr.files}")

def compute_graph_stats_from_npz(path):
    """
    给一个 adjacency_tauXXX.npz，返回该图的拓扑指标。
    这里我们在原有基础上增加：
      - degree_centralization
      - std_degree
      - p90_degree, p99_degree
    """
    adj = load_adj_any(path)

    n = adj.shape[0]           # 节点数
    nnz = adj.nnz              # 非零元素个数（无向图存了两次）
    e = nnz // 2               # 无向边数

    # 度向量（每行求和）
    deg = np.array(adj.sum(axis=1)).ravel()
    avg_deg = float(deg.mean())
    max_deg = int(deg.max())
    std_deg = float(deg.std())

    # 度分位数
    p90_deg = float(np.percentile(deg, 90))
    p99_deg = float(np.percentile(deg, 99))

    # 密度
    if n > 1:
        density = 2.0 * e / (n * (n - 1))
    else:
        density = 0.0

    # 度中心化（Freeman 公式）
    # sum(max_deg - k_i) / ((n-1)*(n-2))
    if n > 2:
        sum_diff = float((max_deg - deg).sum())
        deg_centralization = sum_diff / ((n - 1) * (n - 2))
    else:
        deg_centralization = 0.0

    # 连通分量
    n_components, labels = connected_components(adj, directed=False)
    comp_sizes = np.bincount(labels)
    largest_cc_size = int(comp_sizes.max())

    stats = {
        "nodes": int(n),
        "edges": int(e),
        "avg_degree": avg_deg,
        "max_degree": max_deg,
        "std_degree": std_deg,
        "p90_degree": p90_deg,
        "p99_degree": p99_deg,
        "density": density,
        "degree_centralization": deg_centralization,
        "n_components": int(n_components),
        "largest_cc_size": largest_cc_size,
    }
    return stats


# 根目录：8 个类都在这里
ROOT = "/home/sduu39/zHongchang/data/label"

out_csv = os.path.join(ROOT, "all_graph_stats.csv")

with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    # 表头
    writer.writerow([
        "label_dir", "tau_tag",
        "nodes", "edges",
        "avg_degree", "max_degree", "std_degree",
        "p90_degree", "p99_degree",
        "density", "degree_centralization",
        "n_components", "largest_cc_size"
    ])

    # 遍历 ROOT 下所有子目录（每个子目录就是一个类）
    for label_name in sorted(os.listdir(ROOT)):
        label_path = os.path.join(ROOT, label_name)
        if not os.path.isdir(label_path):
            continue

        thresholds_dir = os.path.join(label_path, "thresholds")
        if not os.path.isdir(thresholds_dir):
            # 不是我们要的类目录，跳过
            continue

        print(f"[INFO] 处理类别目录: {label_path}")

        # 自动发现所有 tau_xxx 子目录，而不是写死 tau_050...
        for tau_dir_name in sorted(os.listdir(thresholds_dir)):
            tau_dir_path = os.path.join(thresholds_dir, tau_dir_name)
            if not os.path.isdir(tau_dir_path):
                continue
            if not tau_dir_name.startswith("tau_"):
                continue

            # tau_dir_name 形如 "tau_050"，我们取 tau_tag = "050"
            tau_tag = tau_dir_name.split("_", 1)[1]

            # 默认 npz 文件命名为 adjacency_tauXXX.npz
            npz_name = f"adjacency_tau{tau_tag}.npz"
            npz_path = os.path.join(tau_dir_path, npz_name)

            if not os.path.exists(npz_path):
                # 如果命名有变化，也可以改成自动找目录下第一个 .npz
                npz_candidates = [x for x in os.listdir(tau_dir_path) if x.endswith(".npz")]
                if npz_candidates:
                    npz_path = os.path.join(tau_dir_path, npz_candidates[0])
                    print(f"[WARN] {npz_name} 不存在，改用 {npz_candidates[0]}")
                else:
                    print(f"[WARN] {tau_dir_path} 下没有 npz 文件，跳过")
                    continue

            print(f"[INFO]  处理 {label_name} | {tau_tag} | 文件: {npz_path}")
            try:
                stats = compute_graph_stats_from_npz(npz_path)
            except Exception as e:
                print(f"[ERROR]  计算失败: {npz_path}, error={e}")
                continue

            writer.writerow([
                label_name,
                tau_tag,
                stats["nodes"],
                stats["edges"],
                f"{stats['avg_degree']:.6f}",
                stats["max_degree"],
                f"{stats['std_degree']:.6f}",
                f"{stats['p90_degree']:.6f}",
                f"{stats['p99_degree']:.6f}",
                f"{stats['density']:.10f}",
                f"{stats['degree_centralization']:.6f}",
                stats["n_components"],
                stats["largest_cc_size"],
            ])

print(f"[OK] 所有结果已写入: {out_csv}")
