#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from pathlib import Path

# ====== 路径配置 ======
IN_PATH  = Path("/home/sduu39/zHongchang/data/label/viewLabel1.jsonl")
OUT_PATH = Path("/home/sduu39/zHongchang/data/label/viewLabel1.l2norm1.jsonl")

# 可选：小数保留位数（减少文件体积）；None 表示不四舍五入
DECIMALS = None  # 例如设为 6 可显著减小体积：DECIMALS = 6

# 打印进度频率
LOG_EVERY = 20000

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """L2 归一化；零向量则原样返回（或改为返回全零）。"""
    # 确保 float32 一致
    v = np.asarray(vec, dtype=np.float32)
    # 展平为 1D
    if v.ndim != 1:
        v = v.reshape(-1)
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v = v / nrm
    else:
        # 你也可以选择：v[:] = 0.0
        pass
    if DECIMALS is not None:
        v = np.round(v, DECIMALS)
    return v

def main():
    assert IN_PATH.exists(), f"Input not found: {IN_PATH}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    zero_cnt = 0
    skip_no_emb = 0
    bad_json = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue

            emb = obj.get("embedding", None)
            if emb is None:
                skip_no_emb += 1
                # 也可以原样写回
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            v = np.asarray(emb, dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(v)) == 0.0:
                zero_cnt += 1

            v_norm = normalize_vec(v)
            # 覆盖写回
            obj["embedding"] = v_norm.tolist()
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            ok += 1

            if total % LOG_EVERY == 0:
                print(f"[PROGRESS] lines={total:,}  normalized={ok:,}  zero_vecs={zero_cnt:,}  bad_json={bad_json:,}  no_emb={skip_no_emb:,}")

    print("\n[DONE]")
    print(f"  total lines     : {total:,}")
    print(f"  normalized      : {ok:,}")
    print(f"  zero vectors    : {zero_cnt:,}")
    print(f"  bad json lines  : {bad_json:,}")
    print(f"  no-embedding    : {skip_no_emb:,}")
    print(f"[OK] Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
