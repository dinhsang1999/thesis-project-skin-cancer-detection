import pandas as pd
import os
import cv2
from tqdm import tqdm


rawCSV_path = "./csvfile/test.csv" #FIXME:
url_dataframe = pd.read_csv(rawCSV_path)
url_dataframe["image_name"] = [str(x) + ".jpg" for x in url_dataframe["image_name"]]
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
# url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

url_dataframe['sex'] = (url_dataframe['sex'].values == 'male')*1
url_dataframe['age_approx'] = url_dataframe['age_approx'].fillna(url_dataframe['age_approx'].mean())
url_dataframe['age_approx'] = url_dataframe['age_approx'] / url_dataframe['age_approx'].values.max()
url_dataframe['anatom_site_general_challenge'] = url_dataframe['anatom_site_general_challenge'].fillna('unknown')
# url_dataframe['diagnosis'] =  url_dataframe['diagnosis'].fillna('unknown')
url_dataframe['width'] = url_dataframe['width'].fillna(url_dataframe['width'].mean())
url_dataframe['width'] = url_dataframe['width'] / url_dataframe['width'].values.max()

url_dataframe['height'] = url_dataframe['height'].fillna(url_dataframe['height'].mean())
url_dataframe['height'] = url_dataframe['height'] / url_dataframe['height'].values.max()

list_anterior_torso = []
list_poster_torso = []
list_torso = []
list_lateral_torso = []
list_upper_ex = []
list_lower_ex = []
list_head_neck = []
list_palm = []
list_oral = []
list_unknown = []

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'anterior torso':
        list_anterior_torso.append(1)
    else: list_anterior_torso.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'torso':
        list_torso.append(1)
    else: list_torso.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'posterior torso':
        list_poster_torso.append(1)
    else: list_poster_torso.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'lower extremity':
        list_lower_ex.append(1)
    else: list_lower_ex.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'upper extremity':
        list_upper_ex.append(1)
    else: list_upper_ex.append(0)
    
for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'lateral torso':
        list_lateral_torso.append(1)
    else: list_lateral_torso.append(0)  

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'head/neck':
        list_head_neck.append(1)
    else: list_head_neck.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'palms/soles':
        list_palm.append(1)
    else: list_palm.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'oral/genital':
        list_oral.append(1)
    else: list_oral.append(0)

for idx,pos in enumerate(url_dataframe['anatom_site_general_challenge']):
    if pos == 'unknown':
        list_unknown.append(1)
    else: list_unknown.append(0)

url_dataframe['anter_torso'] = list_anterior_torso
url_dataframe['poster_torso'] = list_poster_torso
url_dataframe['torso'] = list_torso
url_dataframe['later_torso'] = list_lateral_torso
url_dataframe['upper_ex'] = list_upper_ex
url_dataframe['lower_ex'] = list_lower_ex
url_dataframe['head_neck'] = list_head_neck
url_dataframe['palm_soles'] = list_palm
url_dataframe['oral_genital'] = list_oral
url_dataframe['unknown'] = list_unknown

mean_1 = []
mean_2 = []
mean_3 = []
std_1 = []
std_2 = []
std_3 = []

for item in tqdm(url_dataframe['image_name']):
    path = os.path.join("/mnt/data_lab513/dhsang/data/256x256",str(item)) #FIXME:
    img = cv2.imread(path)
    # img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean, std = cv2.meanStdDev(img)
    mean_1.append(mean[0][0])
    mean_2.append(mean[1][0])
    mean_3.append(mean[2][0])
    std_1.append(std[0][0])
    std_2.append(std[1][0])
    std_3.append(std[2][0])

url_dataframe['mean_1'] = mean_1
url_dataframe['mean_2'] = mean_2
url_dataframe['mean_3'] = mean_3

url_dataframe['std_1'] = std_1
url_dataframe['std_2'] = std_2
url_dataframe['std_3'] = std_3

df = pd.DataFrame(url_dataframe)
df.to_csv('csvfile/full_test.csv',index=False)


