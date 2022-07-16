from tbparse import SummaryReader
import pandas as pd

trial = 35
log_dir = f"log/trial_{trial}"
reader = SummaryReader(log_dir)
df = reader.scalars

epoch = 23

df = df[df.step == epoch]
df = df[df['tag'].str.contains('test')]
df = df.reset_index(drop=True)

df_s = pd.DataFrame()

tag = []
avg = []

step = 1
fold = 1

for i in range(0,len(df),step):
    sum = 0
    for j in range(fold):
        sum += df.iloc[i+j][2]
    avg_ = sum / fold
    tag_ = df.iloc[i][1][:-5]

    tag.append(tag_)
    avg.append(round(avg_,4))

df_s['tag'] = tag
df_s['avg'] = avg

path_s = f'avg_result/trial_{trial}.csv'
df_s.to_csv(path_s,index=False)


