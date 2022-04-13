import argparse
import os
import torch
import json
import warnings
import pandas as pd
from torch import nn
from tqdm import tqdm

from src.utils import get_item,set_seed,preprocess_csv,GradualWarmupSchedulerV2
from src.dataset import CustomImageDataset
from src.model import MelanomaNet,BaseNetwork,MetaMelanoma
from src.trainer import epoch_train,epoch_evaluate

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader


def cross_validate():
    set_fold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state = 69)
    X_,y_ = dataset()

    for k_fold, (train_ids, val_ids) in enumerate (set_fold.split(X_,y_)):
        if k_fold < args.start_from: 
            continue
        trainloader, valloader = set_up_training_for_cross_validation(train_ids,val_ids,X_,y_)
        model, optimizer, loss_fn, device, scaler = set_up_training()

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

        writer_fold = SummaryWriter(os.path.join(args.log_dir,"trial_"+ args.trial,"fold_" +str(k_fold)))
        saved_model_path = os.path.join(args.model_save,"trial_" + args.trial,"fold_" + str(k_fold) + ".pth")

        best_acc = 0

        for epoch in range(args.n_epochs):
            acc_train,loss_train, prec_train, recall_train, f1_train,auc_train,report_train,bal_acc_train,acc_each_class_train,lr =  epoch_train(trainloader,model,loss_fn,device,optimizer,scaler,args.use_meta,args.max_norm)

            if args.scheduler:
                scheduler_warmup.step()
                if epoch==2:
                    scheduler_warmup.step()
            
            writer_fold.add_scalar("AUC/train", auc_train, epoch)
            writer_fold.add_scalar("Loss/train", loss_train, epoch)
            writer_fold.add_scalar("Accuracy/train", acc_train, epoch)
            writer_fold.add_scalar("Balance_Accuracy/train", bal_acc_train, epoch)
            writer_fold.add_scalar("Precision/train", prec_train, epoch)
            writer_fold.add_scalar("Recall/train", recall_train, epoch)
            writer_fold.add_scalar("F1_score/train", f1_train, epoch)

            writer_fold.add_scalar("Accuracy_MEL/train", acc_each_class_train[0], epoch)
            writer_fold.add_scalar("Accuracy_NV/train", acc_each_class_train[1], epoch)
            writer_fold.add_scalar("Accuracy_BCC/train", acc_each_class_train[2], epoch)
            writer_fold.add_scalar("Accuracy_BKL/train", acc_each_class_train[3], epoch)
            writer_fold.add_scalar("Accuracy_AK/train", acc_each_class_train[4], epoch)
            writer_fold.add_scalar("Accuracy_SCC/train", acc_each_class_train[5], epoch)
            writer_fold.add_scalar("Accuracy_VASC/train", acc_each_class_train[6], epoch)
            writer_fold.add_scalar("Accuracy_DF/train", acc_each_class_train[7], epoch)
            writer_fold.add_scalar("Accuracy_Unknow/train", acc_each_class_train[8], epoch)


            writer_fold.add_scalar("Precision_MEL/train", report_train.get('0').get('precision'), epoch)
            writer_fold.add_scalar("Precision_NV/train", report_train.get('1').get('precision'), epoch)
            writer_fold.add_scalar("Precision_BCC/train", report_train.get('2').get('precision'), epoch)
            writer_fold.add_scalar("Precision_BKL/train", report_train.get('3').get('precision'), epoch)
            writer_fold.add_scalar("Precision_AK/train", report_train.get('4').get('precision'), epoch)
            writer_fold.add_scalar("Precision_SCC/train", report_train.get('5').get('precision'), epoch)
            writer_fold.add_scalar("Precision_VASC/train", report_train.get('6').get('precision'), epoch)
            writer_fold.add_scalar("Precision_DF/train", report_train.get('7').get('precision'), epoch)
            writer_fold.add_scalar("Precision_Unknow/train", report_train.get('8').get('precision'), epoch)

            writer_fold.add_scalar("Recall_MEL/train", report_train.get('0').get('recall'), epoch)
            writer_fold.add_scalar("Recall_NV/train", report_train.get('1').get('recall'), epoch)
            writer_fold.add_scalar("Recall_BCC/train", report_train.get('2').get('recall'), epoch)
            writer_fold.add_scalar("Recall_BKL/train", report_train.get('3').get('recall'), epoch)
            writer_fold.add_scalar("Recall_AK/train", report_train.get('4').get('recall'), epoch)
            writer_fold.add_scalar("Recall_SCC/train", report_train.get('5').get('recall'), epoch)
            writer_fold.add_scalar("Recall_VASC/train", report_train.get('6').get('recall'), epoch)
            writer_fold.add_scalar("Recall_DF/train", report_train.get('7').get('recall'), epoch)
            writer_fold.add_scalar("Recall_Unknow/train", report_train.get('8').get('recall'), epoch)

            writer_fold.add_scalar("f1_MEL/train", report_train.get('0').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_NV/train", report_train.get('1').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_BCC/train", report_train.get('2').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_BKL/train", report_train.get('3').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_AK/train", report_train.get('4').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_SCC/train", report_train.get('5').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_VASC/train", report_train.get('6').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_DF/train", report_train.get('7').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_Unknow/train", report_train.get('8').get('f1-score'), epoch)

            writer_fold.add_scalar("learning_rate", lr, epoch)

            acc_test,loss_test, prec_test, recall_test, f1_test,auc_test,report_test,bal_acc_test,acc_each_class_test,top_2_acc,top_3_acc,top_4_acc,top_5_acc =  epoch_evaluate(valloader,model,loss_fn,device,args.use_meta)

            writer_fold.add_scalar("AUC/test", auc_test, epoch)
            writer_fold.add_scalar("Loss/test", loss_test, epoch)
            writer_fold.add_scalar("Accuracy/test", acc_test, epoch)
            writer_fold.add_scalar("Balance_Accuracy/test", bal_acc_test, epoch)
            writer_fold.add_scalar("Precision/test", prec_test, epoch)
            writer_fold.add_scalar("Recall/test", recall_test, epoch)
            writer_fold.add_scalar("F1_score/test", f1_test, epoch)

            writer_fold.add_scalar("Accuracy_MEL/test", acc_each_class_test[0], epoch)
            writer_fold.add_scalar("Accuracy_NV/test", acc_each_class_test[1], epoch)
            writer_fold.add_scalar("Accuracy_BCC/test", acc_each_class_test[2], epoch)
            writer_fold.add_scalar("Accuracy_BKL/test", acc_each_class_test[3], epoch)
            writer_fold.add_scalar("Accuracy_AK/test", acc_each_class_test[4], epoch)
            writer_fold.add_scalar("Accuracy_SCC/test", acc_each_class_test[5], epoch)
            writer_fold.add_scalar("Accuracy_VASC/test", acc_each_class_test[6], epoch)
            writer_fold.add_scalar("Accuracy_DF/test", acc_each_class_test[7], epoch)
            writer_fold.add_scalar("Accuracy_Unknow/test", acc_each_class_test[8], epoch)


            writer_fold.add_scalar("Precision_MEL/test", report_test.get('0').get('precision'), epoch)
            writer_fold.add_scalar("Precision_NV/test", report_test.get('1').get('precision'), epoch)
            writer_fold.add_scalar("Precision_BCC/test", report_test.get('2').get('precision'), epoch)
            writer_fold.add_scalar("Precision_BKL/test", report_test.get('3').get('precision'), epoch)
            writer_fold.add_scalar("Precision_AK/test", report_test.get('4').get('precision'), epoch)
            writer_fold.add_scalar("Precision_SCC/test", report_test.get('5').get('precision'), epoch)
            writer_fold.add_scalar("Precision_VASC/test", report_test.get('6').get('precision'), epoch)
            writer_fold.add_scalar("Precision_DF/test", report_test.get('7').get('precision'), epoch)
            writer_fold.add_scalar("Precision_Unknow/test", report_test.get('8').get('precision'), epoch)

            writer_fold.add_scalar("Recall_MEL/test", report_test.get('0').get('recall'), epoch)
            writer_fold.add_scalar("Recall_NV/test", report_test.get('1').get('recall'), epoch)
            writer_fold.add_scalar("Recall_BCC/test", report_test.get('2').get('recall'), epoch)
            writer_fold.add_scalar("Recall_BKL/test", report_test.get('3').get('recall'), epoch)
            writer_fold.add_scalar("Recall_AK/test", report_test.get('4').get('recall'), epoch)
            writer_fold.add_scalar("Recall_SCC/test", report_test.get('5').get('recall'), epoch)
            writer_fold.add_scalar("Recall_VASC/test", report_test.get('6').get('recall'), epoch)
            writer_fold.add_scalar("Recall_DF/test", report_test.get('7').get('recall'), epoch)
            writer_fold.add_scalar("Recall_Unknow/test", report_test.get('8').get('recall'), epoch)

            writer_fold.add_scalar("f1_MEL/test", report_test.get('0').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_NV/test", report_test.get('1').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_BCC/test", report_test.get('2').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_BKL/test", report_test.get('3').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_AK/test", report_test.get('4').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_SCC/test", report_test.get('5').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_VASC/test", report_test.get('6').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_DF/test", report_test.get('7').get('f1-score'), epoch)
            writer_fold.add_scalar("f1_Unknow/test", report_test.get('8').get('f1-score'), epoch)

            writer_fold.add_scalar("top_2_acc/test", top_2_acc, epoch)
            writer_fold.add_scalar("top_3_acc/test", top_3_acc, epoch)
            writer_fold.add_scalar("top_4_acc/test", top_4_acc, epoch)
            writer_fold.add_scalar("top_5_acc/test", top_5_acc, epoch)


            if args.type_save == 'checkpoint':
                if acc_test > best_acc:
                    print(f'Validate accuracy increased ({best_acc:.3f} --> {acc_test:.3f}).  Saving model to {saved_model_path}')
                    best_acc = acc_test
                    torch.save(model.state_dict(), saved_model_path)
                else:
                    print('Accuracy_validation not improve! the model will not save')
            elif args.type_save == 'full_save':
                torch.save(model.state_dict(), args.model_save+"last-model" + ".pth")
            else: raise ValueError("Check type-save!!!")


def set_up_training_for_cross_validation(train_ids,val_ids,datatrain,target):

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_dataset = CustomImageDataset(
            datatrain,
            target,
            data_dir=args.data_dir,
            target_transform=None,
            mode="train",
            image_size=args.image_size,
            rescale=args.rescale,
            use_meta=args.use_meta)

    val_dataset = CustomImageDataset(
            datatrain,
            target,
            data_dir=args.data_dir,
            target_transform=Lambda(
                lambda y: torch.tensor(y).type(
                    torch.long)),
                    mode="val",image_size=args.image_size,rescale=args.rescale,use_meta=args.use_meta)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_subsampler)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_subsampler)
    
    return train_dataloader,val_dataloader

def set_up_training():
    if args.n_network == 2:
        if args.use_meta:
            model = MetaMelanoma(out_dim=args.out_dim,n_meta_features=args.n_meta_features,n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],network=args.network)
        else:
            model = MelanomaNet(network_1=args.network_1,network_2=args.network_2)
    elif args.n_network == 1:
        model = BaseNetwork(network=args.network)
    else:
        raise ValueError("Wrong choose model within set_up_training!!!")
        
    if args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError("Wrong optimizer!!! within set_up_training")

    if args.CUDA_VISIBLE_DEVICES == "0":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.CUDA_VISIBLE_DEVICES == "1":
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    elif args.CUDA_VISIBLE_DEVICES == "2":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = nn.DataParallel(model,device_ids=[0,1])
    else:
        raise ValueError("Wrong choose cuda within set_up_training")

    scaler =  torch.cuda.amp.grad_scaler.GradScaler()
    loss_fn = nn.CrossEntropyLoss().to(device)
    model.to(device)
    return model, optimizer, loss_fn, device,scaler

def dataset():
    if args.test_size != 0:
        df_train,df_test = preprocess_csv(csv_dir=args.csv_dir,test_size=args.test_size,mode=args.trial)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        df_test.to_csv('./test_data_split/test_dict.csv',index=False)
    else:
        df_train = preprocess_csv(csv_dir=args.csv_dir,test_size=args.test_size,mode=args.trial)
        df_train = df_train.reset_index(drop=True)
    
    if args.use_meta:
        meta_feature = ['sex','age_approx','width','height'] + [col for col in df_train.columns[11:27]]
        _input = []
        _target = []
        for idx in tqdm(range(len(df_train))):
            image_name = df_train.iloc[idx,0]
            meta_features = torch.tensor(df_train.iloc[idx][meta_feature])
            label = df_train.iloc[idx,5]
            input_ = [image_name,meta_features]
            _input.append(input_)
            _target.append(label)
        return _input,_target
    else:
        X,y = df_train['image_name'],df_train['diagnosis']
        return X,y

if __name__ == '__main__':
    args = get_item()
    os.makedirs("./trial_info",exist_ok=True)
    os.makedirs(os.path.join(args.model_save,"trial_" + args.trial),exist_ok=True)
    save_info_path = os.path.join("./trial_info","trial_"+str(args.trial))
    with open(save_info_path+".txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.ignore_warnings:
        warnings.filterwarnings("ignore")
    os.makedirs(args.log_dir, exist_ok=True)
    set_seed(args.seed)
    cross_validate()

