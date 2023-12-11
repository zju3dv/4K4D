import json
import numpy as np
import pandas as pd

metrics_json = json.load(open('data/metrics.json'))

table = np.zeros((len(metrics_json), 10))

for i, exp in enumerate(metrics_json):
    
    met_id = 0
    for j, key in enumerate(metrics_json[exp]):
        for k, met in enumerate(metrics_json[exp][key]):
            table[i, met_id] = metrics_json[exp][key][met]
            met_id += 1

w = pd.ExcelWriter('data/metrics.xlsx')

df = pd.DataFrame(table)
__import__('ipdb').set_trace()
df.to_excel(w)