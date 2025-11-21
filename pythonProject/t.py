from VenueMap import *
import json
file="/home/sduu39/zHongchang/data/label/Label7/viewLabel7.jsonl"
with open(file,"r") as f:
    for line in f:
        data=json.loads(line)
        venue=data['venue']
        if venue not in cla6:
            print(line)
            break