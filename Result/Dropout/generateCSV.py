import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.style.use('ggplot')

features = ['Accuracy','F1_score']

for j in [5,10,15,20]:  
        data = []
        labels = []
        Rate = pd.Series([])
        Accuracy = pd.Series([])
        F_score=  pd.Series([])
        files = os.listdir('.')
        files.sort()
        for k, f in enumerate(files):
            if f[-4:] != '.csv':
                continue
            if f[:7] != 'testing':
                continue
            if f[7:8] != str(j) and f[7:9] != str(j):
                continue
            print(f)
            Rate[k] = f[-15:-12]
            print(f[-15:-12])
            x = []
            for i, feature in enumerate(features):
                df = pd.read_csv('./' + f)
                x.append(df.loc[:, feature].values)
            Accuracy[k] = x[0]
            F_score[k] = x[1]
        
        res = pd.DataFrame({
                "Rate":Rate,
                "Accuracy":Accuracy,
                "F_score":F_score
                })
        res.to_csv(str(j)+"_metrics.csv")