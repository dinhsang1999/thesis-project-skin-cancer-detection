import torch
import os
from src.utils import calculate_metrics
from tqdm import tqdm

def epoch_train(dataloader, model, loss_fn, device,optimizer,scaler,use_meta,max_norm):
    """
    Train the model one time (forward propagation, loss calculation,
        back propagation, update parameters)
    Args:
        dataloader (torch.utils.data.dataloader.DataLoader):
            Dataset holder, load training data in batch.
        model (src_Tran.model.NeuralNetwork): Model architecture
        loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
        optimizer (torch.optim.sgd.SGD): Optimization algorithm
        device (str): Device for training (GPU or CPU)
        epochs (int): Number of epochs
    """
    model.train()

    train_loss = 0
    correct = 0
    total_train=0

    out_pred = torch.FloatTensor().to(device)
    out_gt = torch.IntTensor().to(device)

    for batch, (X, y) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        if use_meta:
            data, meta = X
            data,meta,y = data.to(device),meta.to(device),y.to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(data.float(),meta.float())
                loss = loss_fn(pred, y)
            out_gt = torch.cat((out_gt, y), 0)
            out_pred = torch.cat((out_pred, pred), 0)
            y = y.type(torch.long)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast():          
                pred = model(X.float())
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_norm != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            out_gt = torch.cat((out_gt, y), 0)
            out_pred = torch.cat((out_pred, pred), 0)
            y = y.type(torch.long)        
            train_loss += loss.item()
            # print(train_loss),exit()

        # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # total_train += y.nelement()
        # train_accuracy = 100. * correct / total_train
        # print('\tTraining batch {} Loss: {:.6f}, Accurancy:{:.3f}%'.format(
        #         batch + 1, loss.item(), train_accuracy))
    accuracy, precision, recall, f1,auc_score,report,balance_accuracy,acc_each_class,_,_,_,_,_,_,sensitivity,specificity = calculate_metrics(
            out_gt, out_pred)
        
    print('Training set: Average loss: {:.6f}, Average accuracy: ({:.3f}%)\n'.format(
            train_loss / (batch + 1), 100 * accuracy))

    total_loss = train_loss / (batch + 1)

    lr = optimizer.param_groups[0]["lr"] 

    return accuracy,total_loss, precision, recall, f1,auc_score,report,balance_accuracy,acc_each_class,lr,out_gt,out_pred,sensitivity,specificity



def epoch_evaluate(dataloader, model, loss_fn, device,use_meta):
    """
    Evaluate the model one time (forward propagation, loss calculation)

    Args:
        dataloader (torch.utils.data.dataloader.DataLoader):
            Dataset holder, load training and testing data in batch.
        model (src_Tran.model.NeuralNetwork): Model architecture
        loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
        device (str): Device for training (GPU or CPU)
        epoch (int): Number of epochs
    """
    out_pred = torch.FloatTensor().to(device) #ToDO del.to(device)
    out_gt = torch.IntTensor().to(device)

    val_loss = 0
    correct = 0

    model.eval()

    with torch.no_grad():
        batch_count = 0
        for X, y in tqdm(dataloader):
            batch_count += 1
            if use_meta:
                data, meta = X
                data, meta, y = data.to(device), meta.to(device), y.to(device)
                pred = model(data.float(),meta.float())
                loss = loss_fn(pred, y)
            else:
                X, y = X.to(device), y.to(device)
                pred = model(X.float())

            out_gt = torch.cat((out_gt, y), 0)
            out_pred = torch.cat((out_pred, pred), 0)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = val_loss/ batch_count
    accuracy, precision, recall, f1,auc_score,report,balance_accuracy,acc_each_class,top_2_acc,top_3_acc,top_4_acc,top_5_acc,top_2_accuracy_score_inclass,top_3_accuracy_score_inclass,sensitivity,specificity = calculate_metrics(
    out_gt, out_pred)

    print('Validation set: Average loss: {:.6f}, Average accuracy:({:.3f}%)\n'.format(avg_loss,100.* accuracy))

    return accuracy,avg_loss, precision, recall, f1,auc_score,report,balance_accuracy,acc_each_class,top_2_acc,top_3_acc,top_4_acc,top_5_acc,out_gt,out_pred,top_2_accuracy_score_inclass,top_3_accuracy_score_inclass,sensitivity,specificity 
