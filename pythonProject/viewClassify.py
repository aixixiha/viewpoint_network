from VenueMap import *
import json
from tqdm import tqdm
total_len=3216264
input_file = "/home/sduu39/zHongchang/data/v4_embeddings.jsonl"
cla2_file="/home/sduu39/zHongchang/data/label/Label2/viewLabel2.jsonl"
# cla3_file="/home/sduu39/zHongchang/data/label/Label3/viewLabel2.jsonl"
cla4_file="/home/sduu39/zHongchang/data/label/Label4/viewLabel4.jsonl"
cla5_file="/home/sduu39/zHongchang/data/label/Label5/viewLabel5.jsonl"
cla6_file="/home/sduu39/zHongchang/data/label/Label6/viewLabel6.jsonl"
cla7_file="/home/sduu39/zHongchang/data/label/Label7/viewLabel7.jsonl"
cla8_file="/home/sduu39/zHongchang/data/label/Label8/viewLabel8.jsonl"
new_order = ['index','paperIndex', 'viewpoint', 'sentence', 'embedding', 'title', 'authors', 'year', 'venue', 'references']
c2=1
c4=1
c5=1
c6=1
c7=1
c8=1
with open(input_file,"r") as fin, open(cla2_file,"w") as fout2, open(cla4_file,"w") as fout4, open(cla5_file,"w") as fout5,open(cla6_file,"w") as fout6, open(cla7_file,"w") as fout7,open(cla8_file,"w") as fout8:
    for line in tqdm(fin,total=total_len, desc="Processing"):
        data=json.loads(line)

        venue=data["venue"]

        if venue in cla2:
            data["index"] = c2
            c2 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout2.write(json.dumps(data, ensure_ascii=False) + '\n')

        elif venue in cla4:
            data["index"] = c4
            c4 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout4.write(json.dumps(data, ensure_ascii=False) + '\n')

        elif venue in cla5:
            data["index"] = c5
            c5 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout5.write(json.dumps(data, ensure_ascii=False) + '\n')

        elif venue in cla6:
            data["index"] = c6
            c6 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout6.write(json.dumps(data, ensure_ascii=False) + '\n')

        elif venue in cla7:
            data["index"] = c7
            c7 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout7.write(json.dumps(data, ensure_ascii=False) + '\n')

        elif venue in cla8:
            data["index"] = c8
            c8 += 1
            data = {key: data[key] for key in new_order if key in data}
            fout8.write(json.dumps(data, ensure_ascii=False) + '\n')
