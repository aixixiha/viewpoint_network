import os
import pandas as pd

# ======= 配置 =======
ROOT = "/home/sduu39/zHongchang/data/label"   # 8 个大类所在的目录
ALL_CSV = os.path.join(ROOT, "all_graph_stats.csv")  # 你刚刚生成的总表

# 每个 tau 目录下输出的小文件名
OUT_NAME = "characters.csv"

# ======= 读取总表 =======
df = pd.read_csv(ALL_CSV)

# 防止列名不一致，这里自动识别“指标列”
# 假定前两列是 label_dir, tau_tag，其余都是指标
non_metric_cols = ["label_dir", "tau_tag"]
metric_cols = [c for c in df.columns if c not in non_metric_cols]

print("[INFO] 指标列:", metric_cols)

# ======= 按行遍历，总表中的每一行对应一个 (label_dir, tau_tag) 图 =======
for idx, row in df.iterrows():
    label_dir = row["label_dir"]        # 比如 "1_Computing Systems"
    tau_tag   = str(row["tau_tag"])     # 比如 "060"

    # 对应 tau 目录：ROOT / label_dir / thresholds / tau_XXX
    tau_dir = os.path.join(ROOT, label_dir, "thresholds", f"tau_0{tau_tag}")
    if not os.path.isdir(tau_dir):
        print(f"[WARN] 找不到目录: {tau_dir}，这一行先跳过")
        continue

    out_path = os.path.join(tau_dir, OUT_NAME)

    # 构造“纵向”的 DataFrame：metric | value
    metrics = []
    values  = []
    for col in metric_cols:
        metrics.append(col)
        values.append(row[col])

    sub_df = pd.DataFrame({"metric": metrics, "value": values})

    # 写 CSV
    sub_df.to_csv(out_path, index=False)
    print(f"[OK] 写入: {out_path}")

print("[DONE] 已根据总表拆分到各个 tau 目录。")
