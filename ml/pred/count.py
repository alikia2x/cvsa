# iterate all json files in ./data/pred

import os
import json

count = 0
for filename in os.listdir('./data/pred'):
    if filename.endswith('.json'):
        with open('./data/pred/' + filename, 'r') as f:
            data = json.load(f)
            count += len(data)
print(count)