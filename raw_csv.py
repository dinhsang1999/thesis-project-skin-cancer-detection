import pandas as pd
import os
import cv2
from tqdm import tqdm


class Extract_Metadata():
    def __init__(self,path_csv,mode="train",path_image = ''):
        '''
        Agrs:
            path_csv: path to csv file
            mode: "train" || "test"
            path_image: path to image folder
        Return:
            dataframe
        '''
        self.path_csv = path_csv
        self.mode = mode
        self.url_dataframe = pd.read_csv(self.path_csv)
        self.path_image = path_image

    def __call__(self):
        self.url_dataframe["image_name"] = [str(x) + ".jpg" for x in self.url_dataframe["image_name"]]
        if self.mode == "train":
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
            self.url_dataframe['diagnosis']  = self.url_dataframe['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))
            self.url_dataframe['diagnosis'] =  self.url_dataframe['diagnosis'].fillna('unknown')

        self.url_dataframe['sex'] = (self.url_dataframe['sex'].values == 'male')*1
        self.url_dataframe['age_approx'] = self.url_dataframe['age_approx'].fillna(self.url_dataframe['age_approx'].mean())
        self.url_dataframe['age_approx'] = self.url_dataframe['age_approx'] / self.url_dataframe['age_approx'].values.max()
        self.url_dataframe['anatom_site_general_challenge'] = self.url_dataframe['anatom_site_general_challenge'].fillna('unknown')
        self.url_dataframe['width'] = self.url_dataframe['width'].fillna(self.url_dataframe['width'].mean())
        self.url_dataframe['width'] = self.url_dataframe['width'] / self.url_dataframe['width'].values.max()

        self.url_dataframe['height'] = self.url_dataframe['height'].fillna(self.url_dataframe['height'].mean())
        self.url_dataframe['height'] = self.url_dataframe['height'] / self.url_dataframe['height'].values.max()

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

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'anterior torso':
                list_anterior_torso.append(1)
            else: list_anterior_torso.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'torso':
                list_torso.append(1)
            else: list_torso.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'posterior torso':
                list_poster_torso.append(1)
            else: list_poster_torso.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'lower extremity':
                list_lower_ex.append(1)
            else: list_lower_ex.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'upper extremity':
                list_upper_ex.append(1)
            else: list_upper_ex.append(0)
            
        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'lateral torso':
                list_lateral_torso.append(1)
            else: list_lateral_torso.append(0)  

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'head/neck':
                list_head_neck.append(1)
            else: list_head_neck.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'palms/soles':
                list_palm.append(1)
            else: list_palm.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'oral/genital':
                list_oral.append(1)
            else: list_oral.append(0)

        for idx,pos in enumerate(self.url_dataframe['anatom_site_general_challenge']):
            if pos == 'unknown':
                list_unknown.append(1)
            else: list_unknown.append(0)

        self.url_dataframe['anter_torso'] = list_anterior_torso
        self.url_dataframe['poster_torso'] = list_poster_torso
        self.url_dataframe['torso'] = list_torso
        self.url_dataframe['later_torso'] = list_lateral_torso
        self.url_dataframe['upper_ex'] = list_upper_ex
        self.url_dataframe['lower_ex'] = list_lower_ex
        self.url_dataframe['head_neck'] = list_head_neck
        self.url_dataframe['palm_soles'] = list_palm
        self.url_dataframe['oral_genital'] = list_oral
        self.url_dataframe['unknown'] = list_unknown

        mean_1 = []
        mean_2 = []
        mean_3 = []
        std_1 = []
        std_2 = []
        std_3 = []

        for item in tqdm(self.url_dataframe['image_name']):
            path = os.path.join(self.path_image,str(item)) #FIXME:
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

        self.url_dataframe['mean_1'] = mean_1
        self.url_dataframe['mean_2'] = mean_2
        self.url_dataframe['mean_3'] = mean_3

        self.url_dataframe['std_1'] = std_1
        self.url_dataframe['std_2'] = std_2
        self.url_dataframe['std_3'] = std_3

        df = pd.DataFrame(self.url_dataframe)
        return df   
if __name__ == "__main__":
    csv_file = Extract_Metadata(path_csv = "./csvfile/train.csv",mode = "train",path_image = "/mnt/data_lab513/dhsang/data/256x256")
    csv_file().to_csv("./csvfile/b.csv")