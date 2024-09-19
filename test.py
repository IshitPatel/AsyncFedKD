
import pandas as pd
import numpy as np
import json
data = None

with open('mobilenetv2_history.json', 'r') as f:
    data = json.load(f)

for i in range(len(data['loss'])):
    print(str(data['loss'][i]) + ',' + str(data['accuracy'][i]) + ',' + str(i + 1))
# print(data)
