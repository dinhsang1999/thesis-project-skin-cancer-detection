import argparse
import os
import torch
import json
import warnings
import pandas as pd
import numpy as np
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
from focal_loss import FocalLoss

def cross_validate():
    set_fold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state = 69)
    X_,y_ = dataset()
    first_epoch = True

    for k_fold, (train_ids, val_ids) in enumerate (set_fold.split(X_,y_)):

        if ((args.train_k_fold_and_stop != 0) and (k_fold !=0)):
            if k_fold - (args.train_k_fold_and_stop+args.start_from_fold) == 0:
                print(f'Stop here!! I just train {args.train_k_fold_and_stop} fold by your set up')
                exit()

        if (k_fold < args.start_from_fold) and (args.start_from_fold != 0):
            print(f'Skip fold-{k_fold}')
            continue
        else:
            print('Current fold = ',str(k_fold))

        trainloader, valloader = set_up_training_for_cross_validation(train_ids,val_ids,X_,y_)
        model, optimizer, loss_fn, device, scaler = set_up_training(first_epoch)

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

        writer_fold = SummaryWriter(os.path.join(args.log_dir,"trial_"+ args.trial,"fold_" +str(k_fold)))
        saved_model_path = os.path.join(args.model_save,"trial_" + args.trial,"fold_" + str(k_fold) + ".pth")
        saved_output_pred_train_path = os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"train","fold_" + str(k_fold) + "_pred" + ".pt")
        saved_output_gt_train_path = os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"train","fold_" + str(k_fold) + "_gt" + ".pt")
        saved_output_pred_evaluate_path = os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"evaluate","fold_" + str(k_fold) + "_pred" + ".pt")
        saved_output_gt_evaluate_path = os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"evaluate","fold_" + str(k_fold) + "_gt" + ".pt")
        
        best_acc = 0
        pred_full_train_save = []
        gt_full_train_save = []
        pred_full_evaluate_save = []
        gt_full_evaluate_save = []

        for epoch in range(args.n_epochs):

            if epoch == args.start_from_epoch:
                first_epoch = False

            if (epoch < args.start_from_epoch) and (args.start_from_epoch != 0) and (first_epoch == True):
                print(f'Skip epoch-{epoch}')
                continue
            else:
                print('Current epoch = ',str(epoch))

            ### TRAIN
            acc_train,loss_train, prec_train, recall_train, f1_train,auc_train,report_train,bal_acc_train,acc_each_class_train,lr,out_gt_train,out_pred_train,sensitivity,specificity =  epoch_train(trainloader,model,loss_fn,device,optimizer,scaler,args.use_meta,args.max_norm,args.use_focal_loss)

            pred_full_train_save.append(out_pred_train)
            gt_full_train_save.append(out_gt_train)

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

            writer_fold.add_scalar("sensitivity_MEL/train", sensitivity[0], epoch)
            writer_fold.add_scalar("sensitivity_NV/train", sensitivity[1], epoch)
            writer_fold.add_scalar("sensitivity_BCC/train", sensitivity[2], epoch)
            writer_fold.add_scalar("sensitivity_BKL/train",sensitivity[3], epoch)
            writer_fold.add_scalar("sensitivity_AK/train",sensitivity[4], epoch)
            writer_fold.add_scalar("sensitivity_SCC/train", sensitivity[5], epoch)
            writer_fold.add_scalar("sensitivity_VASC/train", sensitivity[6], epoch)
            writer_fold.add_scalar("sensitivity_DF/train", sensitivity[7], epoch)
            writer_fold.add_scalar("sensitivity_Unknow/train", sensitivity[8], epoch)

            writer_fold.add_scalar("specificity_MEL/train", specificity[0], epoch)
            writer_fold.add_scalar("specificity_NV/train", specificity[1], epoch)
            writer_fold.add_scalar("specificity_BCC/train", specificity[2], epoch)
            writer_fold.add_scalar("specificity_BKL/train",specificity[3], epoch)
            writer_fold.add_scalar("specificity_AK/train",specificity[4], epoch)
            writer_fold.add_scalar("specificity_SCC/train", specificity[5], epoch)
            writer_fold.add_scalar("specificity_VASC/train", specificity[6], epoch)
            writer_fold.add_scalar("specificity_DF/train", specificity[7], epoch)
            writer_fold.add_scalar("specificity_Unknow/train", specificity[8], epoch)

            writer_fold.add_scalar("specificity/train", np.mean(specificity), epoch)
            writer_fold.add_scalar("sensitivity/train", np.mean(sensitivity), epoch)


            writer_fold.add_scalar("learning_rate", lr, epoch)

            ### EVALUATE
            acc_test,loss_test, prec_test, recall_test, f1_test,auc_test,report_test,bal_acc_test,acc_each_class_test,top_2_acc,top_3_acc,top_4_acc,top_5_acc,out_gt_eval,out_pred_eval,top_2_accuracy_score_inclass,top_3_accuracy_score_inclass,sensitivity,specificity =  epoch_evaluate(valloader,model,loss_fn,device,args.use_meta,args.use_focal_loss)

            pred_full_evaluate_save.append(out_pred_eval)
            gt_full_evaluate_save.append(out_gt_eval)

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

            writer_fold.add_scalar("top_2_acc_inclass_MEL/test", top_2_accuracy_score_inclass[0], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_NV/test", top_2_accuracy_score_inclass[1], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_BCC/test", top_2_accuracy_score_inclass[2], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_BKL/test", top_2_accuracy_score_inclass[3], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_AK/test", top_2_accuracy_score_inclass[4], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_SCC/test", top_2_accuracy_score_inclass[5], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_VASC/test", top_2_accuracy_score_inclass[6], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_DF/test", top_2_accuracy_score_inclass[7], epoch)
            writer_fold.add_scalar("top_2_acc_inclass_Unknow/test", top_2_accuracy_score_inclass[8], epoch)

            writer_fold.add_scalar("top_3_acc_inclass_MEL/test", top_3_accuracy_score_inclass[0], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_NV/test", top_3_accuracy_score_inclass[1], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_BCC/test", top_3_accuracy_score_inclass[2], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_BKL/test", top_3_accuracy_score_inclass[3], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_AK/test", top_3_accuracy_score_inclass[4], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_SCC/test", top_3_accuracy_score_inclass[5], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_VASC/test", top_3_accuracy_score_inclass[6], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_DF/test", top_3_accuracy_score_inclass[7], epoch)
            writer_fold.add_scalar("top_3_acc_inclass_Unknow/test", top_3_accuracy_score_inclass[8], epoch)

            writer_fold.add_scalar("sensitivity_MEL/test", sensitivity[0], epoch)
            writer_fold.add_scalar("sensitivity_NV/test", sensitivity[1], epoch)
            writer_fold.add_scalar("sensitivity_BCC/test", sensitivity[2], epoch)
            writer_fold.add_scalar("sensitivity_BKL/test",sensitivity[3], epoch)
            writer_fold.add_scalar("sensitivity_AK/test",sensitivity[4], epoch)
            writer_fold.add_scalar("sensitivity_SCC/test", sensitivity[5], epoch)
            writer_fold.add_scalar("sensitivity_VASC/test", sensitivity[6], epoch)
            writer_fold.add_scalar("sensitivity_DF/test", sensitivity[7], epoch)
            writer_fold.add_scalar("sensitivity_Unknow/test", sensitivity[8], epoch)

            writer_fold.add_scalar("specificity_MEL/test", specificity[0], epoch)
            writer_fold.add_scalar("specificity_NV/test", specificity[1], epoch)
            writer_fold.add_scalar("specificity_BCC/test", specificity[2], epoch)
            writer_fold.add_scalar("specificity_BKL/test",specificity[3], epoch)
            writer_fold.add_scalar("specificity_AK/test",specificity[4], epoch)
            writer_fold.add_scalar("specificity_SCC/test", specificity[5], epoch)
            writer_fold.add_scalar("specificity_VASC/test", specificity[6], epoch)
            writer_fold.add_scalar("specificity_DF/test", specificity[7], epoch)
            writer_fold.add_scalar("specificity_Unknow/test", specificity[8], epoch)

            writer_fold.add_scalar("specificity/test", np.mean(specificity), epoch)
            writer_fold.add_scalar("sensitivity/test", np.mean(sensitivity), epoch)



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

            torch.save(model.state_dict(), "/mnt/data_lab513/dhsang/model/temp/cnt_model.pth")


        
        torch.save(pred_full_train_save, saved_output_pred_train_path)
        torch.save(gt_full_train_save, saved_output_gt_train_path)
        torch.save(pred_full_evaluate_save, saved_output_pred_evaluate_path)
        torch.save(gt_full_evaluate_save,saved_output_gt_evaluate_path)


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

def set_up_training(first_epoch=False):
    if args.n_network == 2:
        print('Use 2 network combine')
        if args.use_meta:
            print('Use Metadata')
            model = MetaMelanoma(out_dim=args.out_dim,n_meta_features=args.n_meta_features,n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],network=args.network)
        else:
            print('Use Stacking Model')
            model = MelanomaNet(network_1=args.network_1,network_2=args.network_2)
    elif args.n_network == 1:
        print('Use BaseModel:',args.network)
        model = BaseNetwork(network=args.network)
    else:
        raise ValueError("Wrong choose model within set_up_training!!!")
        
    if args.optimizer == "adam":
        print('Use Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        print('Use SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise ValueError("Wrong optimizer!!! within set_up_training")

    if args.CUDA_VISIBLE_DEVICES == "0":
        print('Use CUDA=0')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif args.CUDA_VISIBLE_DEVICES == "1":
        print('Use CUDA=1')
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    elif args.CUDA_VISIBLE_DEVICES == "2":
        print('Use CUDA=2-DataParllel')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = nn.DataParallel(model,device_ids=[0,1])
    else:
        raise ValueError("Wrong choose cuda within set_up_training")

    scaler =  torch.cuda.amp.grad_scaler.GradScaler()
    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=torch.tensor([1.27207642,0.35948761,1.95462601,2.28142684,7.49160579,10.34271054,25.67281511,27.17666202,0.23944637]),gamma=args.fl_gamma).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss(torch.tensor([7.0,0.2,5.0,5.0,5.0,10.0,10.0,10.0,0.1])).to(device) #TODO: del .to(device)
    model.to(device)
    if (args.start_from_epoch) != 0 and (first_epoch == True) and (args.start_from_fold !=0):
        print('Skip fold and epoch')
        path_model = "/mnt/data_lab513/dhsang/model/temp/cnt_model.pth"
        model_loader = torch.load(path_model)
        model.load_state_dict(model_loader)
    return model, optimizer, loss_fn, device,scaler

def dataset():
    print('Get Dataset!')
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
    os.makedirs("/mnt/data_lab513/dhsang/model/temp",exist_ok=True)
    os.makedirs("./trial_info",exist_ok=True)
    os.makedirs(os.path.join(args.model_save,"trial_" + args.trial),exist_ok=True)
    os.makedirs(os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"train"),exist_ok=True)
    os.makedirs(os.path.join("/mnt/data_lab513/dhsang/output","trial_" + args.trial,"evaluate"),exist_ok=True)
    save_info_path = os.path.join("./trial_info","trial_"+str(args.trial))
    with open(save_info_path+".txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.ignore_warnings:
        warnings.filterwarnings("ignore")
    os.makedirs(args.log_dir, exist_ok=True)
    print('trial:',args.trial)
    print('SEED:',args.seed)
    set_seed(args.seed)
    cross_validate()

