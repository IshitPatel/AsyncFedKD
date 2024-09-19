import pandas as pd
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv("./calcifications.csv")

y = data['severity'].copy()
X = data.drop(columns=['severity'])

skf = StratifiedKFold(n_splits=4)

indexes = None
idx = 0

for train_index, test_index in skf.split(X,y):
    print(train_index,test_index)