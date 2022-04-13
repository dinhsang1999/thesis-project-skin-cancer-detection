import torch
import os
from torch import nn
import argparse
import cv2
import pandas as pd
import albumentations as A  
from torchvision import transforms
from utils import get_item
from model import MelanomaNet,BaseNetwork,MetaMelanoma


class Predicter():
    def __init__(self,use_meta=True, parallel_use = True, path_pretrained_model = ''):
        '''
        Agrs:
            - use_meta (Boolean): Do model using metadata or not?
            - parallel_use (Boolean): Do model using parallel to predict? If the model have '.module' and you need only use a gpu to predict, u should delete '.module' within model
            - path_pretrained_model (String): model path
        Return:
            - Accuracy of 9 class
        '''
        self.use_meta = use_meta
        path = os.path.join(args.model_save,"trial_" + str(6) + "_meta","fold_" + str(1) + ".pth") #FIXME:

        if self.use_meta:
            self.model = MetaMelanoma(out_dim=args.out_dim,n_meta_features=args.n_meta_features,n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],network=args.network)
        else:
            self.model = BaseNetwork(network=args.network)

        if torch.cuda.device_count() > 1 & parallel_use == True:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model,device_ids=[0,1])
            self.model.to(self.device)
            model_loader = torch.load(path)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            model_loader = torch.load(path)
            model_loader = {key.replace("module.", ""): value for key, value in model_loader.items()} #delete modul into model to train 1 gpu

        self.model.load_state_dict(model_loader)

        # Switch model to evaluation mode
        self.model.eval()

        #Transform image
        list_agu = [A.Normalize()]
        self.transform = A.Compose(list_agu)

    def predict(self,path_image, meta_features=None,rescale=512):
        """
        Predict image in image_path.

        Args:
            image_path (str): Directory of image file.
            meta_feature (array): features to input the model
            rescale (int): rescale image satify the model
        Returns:
            result (dict): Dictionary of propability of 9 classes,
                and predicted class of the image.
        """
        # Read image
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Transform
        image = cv2.resize(image,(rescale,rescale))
        transformed = self.transform(image=image)
        image = transformed["image"]

        image = image.transpose(2, 0, 1)

        image = torch.tensor(image).float()
        image = image.to(self.device)
        meta_features = meta_features

        image = image.view(1, *image.size()).to(self.device)
        meta_features = torch.unsqueeze(meta_features, dim = 0)

        if meta_features != None:
            with torch.no_grad():
                pred = self.model(image.float(),meta_features.float())
        else:
            with torch.no_grad():
                pred = self.model(image.float())

        pred = torch.nn.functional.softmax(pred, dim=1)
        return pred

if __name__ == '__main__':
    args = get_item()
    test = Predicter()
    data = pd.read_csv("../test_data_split/test_dict.csv")
    data_dir = "/mnt/data_lab513/dhsang/data/768x768"
    image_path = os.path.join(data_dir,data.iloc[0,0])
    meta_feature = ['sex','age_approx','width','height'] + [col for col in data.columns[11:27]]
    meta_features = torch.tensor(data.iloc[0][meta_feature])
    temp = test.predict(image_path,meta_features)
    print(temp)