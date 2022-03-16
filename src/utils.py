import argparse
import torch
import random
import sys
import cv2
import os
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score,roc_auc_score, accuracy_score, precision_score,recall_score,f1_score,classification_report,balanced_accuracy_score,confusion_matrix
from warmup_scheduler import GradualWarmupScheduler # https://github.com/ildoonet/pytorch-gradual-warmup-lr


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_item():
    parser = argparse.ArgumentParser()
    #folder
    parser.add_argument('--data-dir', type=str, default="/mnt/data_lab513/dhsang/data/256x256")
    parser.add_argument('--test-dir', type=str, default="/mnt/data_lab513/dhsang/data_2020/archive/test")
    parser.add_argument('--data-folder', type=int, default=256)
    parser.add_argument('--csv-dir', type=str, default="csvfile/full_train.csv")
    parser.add_argument('--model-save', type=str, default="/mnt/data_lab513/dhsang/model/")
    parser.add_argument('--type-save',type=str, default='checkpoint') #'checkpoint' & 'full_save'
    #hyperparameter
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--weight-decay',type=float,default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scheduler', action='store_false')
    parser.add_argument('--max-norm',type=float,default=0)
    #network
    #FIXME: Stacking model
    parser.add_argument('--n-network', type=int, default=2)
    parser.add_argument('--network-1', type=str, default="efficientnet_b2")
    parser.add_argument('--network-2', type=str, default="resnet101")
    parser.add_argument('--network', type=str, default="efficientnet_b0")
    #dataset
    parser.add_argument('--test-size', type=float, default=0)
    #initial hardware
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='2')
    #others
    parser.add_argument('--trial', type=str, default="8-stacking") #FIXME: 
    parser.add_argument('--architecture', type=str, default="custom-stacking")
    parser.add_argument('--start-from', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--rescale', type=int, default=None)
    parser.add_argument('--ignore-warnings', action='store_false')
    parser.add_argument('--grad-norm', action='store_true')
    #cross-validate
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--log-dir', type=str, default='./log')
    #metadata
    ##Note: n-network == 2
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-meta-features', type=int, default=20)

    args, _ = parser.parse_known_args()
    return args



def preprocess_csv(csv_dir,test_size=0,mode="train"):
    url_dataframe = pd.read_csv(csv_dir)

    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('MEL', '0'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('NV', '1'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('BCC', '2'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('BKL', '3'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('AK', '4'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('SCC', '5'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('VASC', '6'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('DF', '7'))
    url_dataframe['diagnosis']  = url_dataframe['diagnosis'].apply(lambda x: x.replace('unknown', '8'))
    url_dataframe['diagnosis'] = url_dataframe['diagnosis'].apply(lambda x: int(x))

    if mode == "test":
        ignore_idex = 0.95
        df_train,_ = train_test_split(url_dataframe,test_size=ignore_idex,random_state=45)
        return df_train

    if test_size != 0:

        df_0 = url_dataframe.groupby('diagnosis').get_group(0)
        df_1 = url_dataframe.groupby('diagnosis').get_group(1)
        df_2 = url_dataframe.groupby('diagnosis').get_group(2)
        df_3 = url_dataframe.groupby('diagnosis').get_group(3)
        df_4 = url_dataframe.groupby('diagnosis').get_group(4)
        df_5 = url_dataframe.groupby('diagnosis').get_group(5)
        df_6 = url_dataframe.groupby('diagnosis').get_group(6)
        df_7 = url_dataframe.groupby('diagnosis').get_group(7)
        df_8 = url_dataframe.groupby('diagnosis').get_group(8)

        df_0_train,df_0_test = train_test_split(df_0,test_size=test_size,random_state=107) 
        df_1_train,df_1_test = train_test_split(df_1,test_size=test_size,random_state=107) 
        df_2_train,df_2_test = train_test_split(df_2,test_size=test_size,random_state=54) 
        df_3_train,df_3_test = train_test_split(df_3,test_size=test_size,random_state=54) 
        df_4_train,df_4_test = train_test_split(df_4,test_size=test_size,random_state=23) 
        df_5_train,df_5_test = train_test_split(df_5,test_size=test_size,random_state=54) 
        df_6_train,df_6_test = train_test_split(df_6,test_size=test_size,random_state=23) 
        df_7_train,df_7_test = train_test_split(df_7,test_size=test_size,random_state=23) 
        df_8_train,df_8_test = train_test_split(df_8,test_size=test_size,random_state=107) 


        df_train = pd.concat([df_0_train,df_1_train,df_2_train,df_3_train,df_4_train,df_5_train,df_6_train,df_7_train,df_8_train])
        df_test = pd.concat([df_0_test,df_1_test,df_2_test,df_3_test,df_4_test,df_5_test,df_6_test,df_7_test,df_8_test])

        return df_train,df_test
 
    return url_dataframe

def calculate_metrics(out_gt, out_pred):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity

    """
    y_true = out_gt.cpu().detach().numpy()
    out_gt = F.one_hot(out_gt,num_classes=9).float()

    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()

    top_2_acc = top_k_accuracy_score(y_true, out_pred, k=2)
    top_3_acc = top_k_accuracy_score(y_true, out_pred, k=3)
    top_4_acc = top_k_accuracy_score(y_true, out_pred, k=4)
    top_5_acc = top_k_accuracy_score(y_true, out_pred, k=5)
    auc_score = roc_auc_score(out_gt,out_pred)

    out_gt = out_gt.argmax(1)
    out_pred = out_pred.argmax(1)

    accuracy = accuracy_score(out_gt,out_pred)
    balance_accuracy = balanced_accuracy_score(out_gt,out_pred)
    precision = precision_score(out_gt,out_pred,average='weighted')
    f1 = f1_score(out_gt,out_pred,average = 'weighted')
    recall = recall_score(out_gt,out_pred,average = 'weighted')
    report = classification_report(out_gt,out_pred,output_dict=True)

    matrix_dia = confusion_matrix(out_gt,out_pred)
    matrix_dia = matrix_dia.astype('float') / matrix_dia.sum(axis=1)[:, np.newaxis]
    acc_each_class = matrix_dia.diagonal()

    return accuracy, precision, recall, f1,auc_score,report,balance_accuracy,acc_each_class,top_2_acc,top_3_acc,top_4_acc,top_5_acc

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calc_avg_mean_std(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean, std = cv2.meanStdDev(img)

    return mean,std
# Block
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    args=get_item()
    with open('./xoa/test.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

