import requests
import json
import time

with open("1", "r") as fp:
    raw = fp.readlines()
    aids = [ int(x.strip()) for x in raw ]

for aid in aids:
    url = f"https://api.bunnyxt.com/tdd/v2/video/{aid}/record?last_count=5000"
    r = requests.get(url)
    data = r.json()
    with open (f"./data/pred/{aid}.json", "w") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    time.sleep(5)