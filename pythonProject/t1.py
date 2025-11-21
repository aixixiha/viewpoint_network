#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 直接计算相似度建边
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Any, Optional

import numpy as np
import faiss
import networkx as nx

# 稀疏邻接矩阵
try:
    from scipy.sparse import coo_matrix, save_npz
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ========= 路径 & 参数 =========
JSONL_PATH = "/home/sduu39/zHongchang/data/label/1_Computing Systems/viewLabel1.jsonl"
OUT_DIR    = "/home/sduu39/zHongchang/data/label/1_Computing Systems/thresholds/tau_070"

# 多个阈值（固定写死，避免浮点误差）
TAU_LIST = [
           0.7]

# 构边参数（自适应 K 扩张）
K0         = 500      # 初始 top-K
STEP       = 100      # 扩张步长
MAX_K      = 8000     # 单节点最大 K（护栏）
DEGREE_CAP = None     # 每个节点最大度（如 300）；None 表示不限制

# 保存前是否保留节点上的 embedding（占内存很大，建议 False）
KEEP_EMBEDDING_IN_G = False

# ======== 工具函数 ========
def _as_float_list(x) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).tolist()
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None

# ======== 读取 JSONL，建图节点 + 收集向量 ========
def load_nodes_and_embeddings(jsonl_path: str) -> Tuple[nx.Graph, np.ndarray, List[Any]]:
    """
    读取 JSONL：
      - G：NetworkX 图（节点属性不包含 embedding，当 KEEP_EMBEDDING_IN_G=False）
      - X：(N, D) 的 float32 向量矩阵，已 L2 归一化
      - ids：节点 index 列表，对应 X 的行/列顺序
    """
    G = nx.Graph()
    ids: List[Any] = []
    embs: List[List[float]] = []
    bad_cnt, ok_cnt = 0, 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_cnt += 1
                continue

            if "index" not in obj or "embedding" not in obj:
                bad_cnt += 1
                continue

            idx = obj["index"]
            emb = _as_float_list(obj["embedding"])
            if emb is None:
                bad_cnt += 1
                continue

            # 节点属性：按配置决定是否保留 embedding
            attr = obj.copy()
            if not KEEP_EMBEDDING_IN_G:
                attr.pop("embedding", None)
            G.add_node(idx, **attr)

            ids.append(idx)
            embs.append(emb)
            ok_cnt += 1

    X = np.asarray(embs, dtype=np.float32)
    print(f"[INFO] Loaded nodes: {ok_cnt}, skipped: {bad_cnt}, X.shape = {X.shape}")
    return G, X, ids

# ======== 建立 Faiss 索引（CPU -> GPU） ========
def build_faiss_index(X: np.ndarray):
    """
    根据已归一化的向量 X 建立 Faiss 索引。
    优先使用 GPU（所有可见 GPU），失败则退回 CPU。
    """
    N, D = X.shape
    print(f"[INFO] Building Faiss IndexFlatIP, N={N}, D={D}")

    index_cpu = faiss.IndexFlatIP(D)
    index_cpu.add(X)

    # 尝试搬到所有 GPU 上
    try:
        index = faiss.index_cpu_to_all_gpus(index_cpu)
        print("[INFO] Faiss index moved to all available GPUs.")
    except Exception as e:
        print(f"[WARN] index_cpu_to_all_gpus failed, falling back to CPU index. Error: {e}")
        index = index_cpu

    return index

# ======== 自适应 K 扩张检索并加边到 G（使用外部传入的 index） ========
def build_edges_adaptive(
    G: nx.Graph,
    X: np.ndarray,
    ids: List[Any],
    index,
    tau: float,
    k0: int = K0,
    step: int = STEP,
    max_k: int = MAX_K,
    degree_cap: Optional[int] = DEGREE_CAP,
) -> None:
    """
    使用已有的 Faiss 索引 index（内积 = 余弦）：
      - 初始 top-K0；若第 K 名仍 >= tau，则扩张 K；
      - 最大不超过 max_k；
      - 无向图，仅在 id_i < id_j 时加边；边权为相似度。
    """
    N, D = X.shape  # D 实际不再用，只是保持接口一致

    current_K = np.full(N, k0, dtype=np.int32)
    need_more = np.ones(N, dtype=bool)

    pos2id = np.array(ids)

    def add_edges_for_block(query_pos: np.ndarray, S_blk: np.ndarray, I_blk: np.ndarray, K_used: int):
        nonlocal need_more
        for r, qpos in enumerate(query_pos):
            sims = S_blk[r]
            nbrs = I_blk[r]

            # 去自身
            if nbrs[0] == qpos:
                nbrs = nbrs[1:]
                sims = sims[1:]
            else:
                mself = nbrs != qpos
                nbrs = nbrs[mself]
                sims = sims[mself]

            # 阈值过滤
            m = sims >= tau
            if np.any(m):
                nbrs_kept = nbrs[m]
                sims_kept = sims[m]

                # 最大度限制（可选）
                if degree_cap is not None and len(nbrs_kept) > degree_cap:
                    sel = np.argpartition(-sims_kept, degree_cap - 1)[:degree_cap]
                    nbrs_kept = nbrs_kept[sel]
                    sims_kept = sims_kept[sel]

                # 无向去重：仅在 id_i < id_j 时加边
                id_i = pos2id[qpos]
                for npos, s in zip(nbrs_kept.tolist(), sims_kept.tolist()):
                    id_j = pos2id[npos]
                    a, b = (id_i, id_j) if str(id_i) < str(id_j) else (id_j, id_i)
                    if a != b and not G.has_edge(a, b):
                        G.add_edge(a, b, weight=float(s))

            # 是否继续扩张
            if len(sims) >= K_used:
                kth_sim = sims[K_used - 1]
                need_more[qpos] = bool(kth_sim >= tau and K_used < max_k)
            else:
                need_more[qpos] = False

    print(f"[INFO] Initial search: K={k0}, tau={tau}")
    S, I = index.search(X, k0 + 1)
    add_edges_for_block(np.arange(N), S, I, k0)

    round_cnt = 0
    while True:
        candidates = np.where(need_more)[0]
        if len(candidates) == 0:
            break
        round_cnt += 1
        current_K[candidates] = np.minimum(current_K[candidates] + step, max_k)
        Kq = int(current_K[candidates].max())
        if round_cnt % 5 == 0:
            print(f"[INFO] Round {round_cnt}: expand {len(candidates)} nodes to K={Kq} (tau={tau})")
        S2, I2 = index.search(X[candidates], Kq + 1)
        add_edges_for_block(candidates, S2, I2, Kq)

    print(f"[INFO] Adaptive expansion finished (tau={tau}).")

# ======== 保存邻接矩阵（稀疏） ========
def save_adjacency_matrix_sparse(G: nx.Graph, ids: List[Any], out_npz_path: Path):
    """
    用 'ids' 的顺序作为矩阵的行列顺序，保存无向邻接矩阵（对称），权重=相似度。
    """
    id2pos = {nid: i for i, nid in enumerate(ids)}
    rows, cols, vals = [], [], []

    # 无向：写 i->j 与 j->i
    for u, v, data in G.edges(data=True):
        i = id2pos[u]
        j = id2pos[v]
        w = float(data.get("weight", 1.0))
        rows.append(i); cols.append(j); vals.append(w)
        rows.append(j); cols.append(i); vals.append(w)

    n = len(ids)
    if SCIPY_AVAILABLE:
        coo = coo_matrix((np.asarray(vals, dtype=np.float32),
                          (np.asarray(rows, dtype=np.int64),
                           np.asarray(cols, dtype=np.int64))),
                         shape=(n, n))
        csr = coo.tocsr()
        save_npz(str(out_npz_path), csr)
    else:
        # 兜底：不用 scipy 时，存三元组
        np.savez_compressed(str(out_npz_path),
                            rows=np.asarray(rows, dtype=np.int64),
                            cols=np.asarray(cols, dtype=np.int64),
                            data=np.asarray(vals, dtype=np.float32),
                            shape=np.array([n, n], dtype=np.int64))

# ======== 主流程：循环多个 TAU ========
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) 读取一次节点 & 向量
    G_base, X, ids = load_nodes_and_embeddings(JSONL_PATH)

    # 2) 建立一次 Faiss 索引（CPU -> GPU），后续复用
    index = build_faiss_index(X)

    # 3) 保存一次 index 的顺序映射（行/列号）
    index_order_path = Path(OUT_DIR) / "index_order.npy"
    np.save(str(index_order_path), np.array(ids, dtype=object))
    print(f"[OK] Saved index order: {index_order_path}  (len={len(ids)})")

    # 4) 逐个 TAU 构图/保存
    summary = []
    for tau in TAU_LIST:
        tau_tag = f"{tau:.2f}".replace(".", "")
        g_path  = Path(OUT_DIR) / f"view_similarity_graph_tau{tau_tag}.gpickle"
        adj_path= Path(OUT_DIR) / f"adjacency_tau{tau_tag}.npz"

        # 4.1 拷贝节点（浅拷贝属性字典，避免修改 base）
        G_tau = nx.Graph()
        G_tau.add_nodes_from((n, data.copy()) for n, data in G_base.nodes(data=True))

        # 4.2 构边（自适应 K + 阈值）
        build_edges_adaptive(
            G=G_tau,
            X=X,
            ids=ids,
            index=index,
            tau=tau,
            k0=K0,
            step=STEP,
            max_k=MAX_K,
            degree_cap=DEGREE_CAP,
        )

        # 4.3 如果 KEEP_EMBEDDING_IN_G=True，这里才考虑删除；目前为 False，此步基本为 0
        if not KEEP_EMBEDDING_IN_G:
            removed = 0
            for n, d in G_tau.nodes(data=True):
                if "embedding" in d:
                    del d["embedding"]
                    removed += 1
            if removed > 0:
                print(f"[INFO] (tau={tau}) removed 'embedding' from {removed} nodes before save.")

        # 4.4 保存 gpickle
        with open(str(g_path), "wb") as f:
            pickle.dump(G_tau, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Graph saved: {g_path}  (nodes={G_tau.number_of_nodes()}, edges={G_tau.number_of_edges()})")

        # 4.5 保存稀疏邻接矩阵（对称，权重=相似度）
        save_adjacency_matrix_sparse(G_tau, ids, adj_path)
        print(f"[OK] Adjacency (sparse) saved: {adj_path}")

        # 4.6 记录统计
        n, e = G_tau.number_of_nodes(), G_tau.number_of_edges()
        avg_deg = (2.0 * e) / max(1, n)
        summary.append((tau, n, e, avg_deg))

        # 4.7 释放本轮图
        del G_tau

    # 5) 汇总统计另存
    sum_path = Path(OUT_DIR) / "tau_summary.csv"
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("tau,nodes,edges,avg_degree\n")
        for tau, n, e, avg_deg in summary:
            f.write(f"{tau:.2f},{n},{e},{avg_deg:.6f}\n")
    print(f"[OK] Summary saved: {sum_path}")

if __name__ == "__main__":
    main()
