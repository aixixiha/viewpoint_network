import os
import numpy as np
from scipy.sparse import load_npz, save_npz

ROOT = "/home/sduu39/zHongchang/data/label"   # 8个大类所在目录

for label_name in sorted(os.listdir(ROOT)):
    label_path = os.path.join(ROOT, label_name)
    if not os.path.isdir(label_path):
        continue

    thresholds_dir = os.path.join(label_path, "thresholds")
    if not os.path.isdir(thresholds_dir):
        continue

    print(f"[INFO] 处理类别目录: {label_path}")

    for tau_dir_name in sorted(os.listdir(thresholds_dir)):
        tau_dir_path = os.path.join(thresholds_dir, tau_dir_name)
        if not os.path.isdir(tau_dir_path):
            continue
        if not tau_dir_name.startswith("tau_"):
            continue

        tau_tag = tau_dir_name.split("_", 1)[1]  # 比如 "060"
        npz_name = f"adjacency_tau{tau_tag}.npz"
        npz_path = os.path.join(tau_dir_path, npz_name)
        if not os.path.exists(npz_path):
            print(f"[WARN] 找不到 {npz_path}，跳过")
            continue

        out_name = f"adj.npz"
        out_path = os.path.join(tau_dir_path, out_name)

        print(f"[INFO]  生成 0/1 邻接矩阵: {npz_path} -> {out_path}")
        adj = load_npz(npz_path)
        adj_bin = adj.copy()
        adj_bin.data = np.ones_like(adj_bin.data, dtype=np.int8)  # 或者 dtype=bool 也行

        save_npz(out_path, adj_bin)

print("[DONE] 所有 0/1 链接矩阵已生成。")
