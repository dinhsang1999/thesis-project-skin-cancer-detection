import pandas as pd

path_2019 = '2018_2019/train_2019.csv'
path_2020 = '2020/train_2020.csv'

# df_1 = pd.read_csv('train_2019.csv')
# df_2 = pd.read_csv('train_2020.csv')

df = pd.concat(
    map(pd.read_csv, [path_2019,path_2020]), ignore_index=True)

df.to_csv('train.csv',index=True)