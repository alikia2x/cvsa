import os
import requests
import json
import time

with open("./pred/2", "r") as fp:
    raw = fp.readlines()
    aids = [ int(x.strip()) for x in raw ]

for aid in aids:
    if os.path.exists(f"./data/pred/{aid}.json"):
        continue
    url = f"https://api.bunnyxt.com/tdd/v2/video/{aid}/record?last_count=5000"
    r = requests.get(url)
    data = r.json()
    with open (f"./data/pred/{aid}.json", "w") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    time.sleep(5)
    print(aid)