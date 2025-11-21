from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# 模型路径
pro_path="/home/sduu39/zHongchang/"
model_path = pro_path+"model"
# 加载 tokenizer 和模型
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",         # 自动分配到可用 GPU/CPU
    torch_dtype="float16",     # 半精度，减少显存占用
)

# 建立一个pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype="float16"
)
data_path=pro_path+"data/"

def build_prompt(abstract):
    return f"""
[The Start of Abstract]
State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL.
[The End of Abstract]
Your Answer:
[Sentence 1]
State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories.
[Extracted Viewpoints in Sentence 1]
[State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories.]
[Sentence 2]
This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept.
[Extracted Viewpoints in Sentence 2]
[The generality and usability of state-of-the-art computer vision systems are limited by being trained to predict a fixed set of predetermined object categories.]
[Sentence 3]
Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.
[Extracted Viewpoints in Sentence 3]
[Learning directly from raw text about images is a promising alternative to learning to predict a fixed set of predetermined object categories.]
[Learning directly from raw text about images leverages a much broader source of supervision than learning to predict a fixed set of predetermined object categories.]
[Sentence 4]
We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet.
[Extracted Viewpoints in Sentence 4]
[The pre-training task of predicting which caption goes with which image on a dataset of millions of (image, text) pairs collected from the internet is simple.]
[The pre-training task of predicting which caption goes with which image on a dataset of millions of (image, text) pairs collected from the internet is efficient.]
[The pre-training task of predicting which caption goes with which image on a dataset of millions of (image, text) pairs collected from the internet is scalable.]
[The pre-training task of predicting which caption goes with which image can be used to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet.]
[Sentence 5]
After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.
[Extracted Viewpoints in Sentence 5]
[After pre-training on a dataset of millions of (image, text) pairs collected from the internet to predict which caption corresponds to which image, natural language is used to reference learned visual concepts or describe new ones.]
[Using natural language to reference learned visual concepts or describe new ones enables zero-shot transfer of the model to downstream tasks.]
[Sentence 6]
We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification.
[Extracted Viewpoints in Sentence 6]
[The performance of the proposed approach was evaluated by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification.]
[Sentence 7]
The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training.
[Extracted Viewpoints in Sentence 7]
[The proposed model transfers non-trivially to most tasks across over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification.]
[The proposed model is often competitive with a fully supervised baseline without the need for any dataset specific training.]
[Sentence 8]
For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
[Extracted Viewpoints in Sentence 8]
[The proposed model matches the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any training examples it was trained on.]
[Sentence 9]
We release our code and pre-trained model weights at this https URL.
[Extracted Viewpoints in Sentence 9]
[The code and pre-trained model weights are released.]
摘要：{abstract}
输出：
"""

input_file=data_path+"DBLP-Citation-network-V18.jsonl"
output_file=data_path+"output1.jsonl"


import pandas as pd
import json
max_samples = 200000  # 只处理 300 条
chunk_size=50

batch_size = 16  # 每次送进去多少条
prompts = []
records = []
processed = 0

with open(output_file, "w", encoding="utf-8") as f_out:
    reader = pd.read_json(
        input_file,
        lines=True,
        chunksize=chunk_size,
        dtype={"id": "string", "year": "Int32", "abstract": "string"}
    )

    for chunk in reader:
        for _, row in chunk.iterrows():
            if pd.isna(row["abstract"]):
                continue
            if processed >= max_samples:
                break

            prompt = build_prompt(row["abstract"])
            prompts.append(prompt)
            records.append({
                "id": row["id"],
                "title": row["title"],
                "year": int(row["year"]) if not pd.isna(row["year"]) else None
            })
            processed += 1

            if len(prompts) >= batch_size:
                responses = generator(
                    prompts,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    batch_size=batch_size
                )
                for prompt, resp in zip(prompts, responses):

                    generated = resp[0]["generated_text"]

                    # 去掉提示词部分（防止输出里包含 prompt）
                    cleaned = generated[len(prompt):].strip()

                    rec = {
                        "id": row["id"],
                        "title": row["title"],
                        "year": int(row["year"]) if not pd.isna(row["year"]) else None,
                        "summary_points": cleaned
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                prompts, records = [], []
            print(f"已处理 {processed} 条")
        if processed >= max_samples:
            break

print(f"已处理 {processed} 条，结果保存在 {output_file}")

python t4.py
python t5.py
python t6.py
python t7.py
python t8.py