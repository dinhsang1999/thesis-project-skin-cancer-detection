from sklearn.utils import class_weight
import pandas as pd
import numpy as np

datafile = 'csvfile/full_train.csv'
ds = pd.read_csv(datafile)

y = ds['diagnosis'].tolist()

classes = ['MEL','NV','BCC','BKL','AK','SCC','VASC','DF','unknown']

class_weight = class_weight.compute_class_weight(class_weight='balanced',classes=classes,y=y)
print(np.around(class_weight,2))
