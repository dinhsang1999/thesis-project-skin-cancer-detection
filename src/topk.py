import numpy as np
import torch

#TODO: Add check value within y_true & y_score

def  top_k_accuracy_score_inclass(y_true,y_score,k=1,softmax=True):
    '''
        Args:
            y_true: array-like of shape (n_samples,)
                True label from zeros (Example: [0,1,2])
                torchTensor or numpy.array
            y_score: array-like of shape (n_samples,n_classes)
                to_categorical
                torchTensor or numpy.array
            k: int, default: 1
                Number of most likely outcomes considered to find the correct label
            softmax: use softmax on y_score
                if y_score is numpy.array, then softmax should False, and wait my update
        Returns:
            list_score : float
                The top-k accuracy score of each class
        See also
        -------
        accuracy_score
    
    Examples
    ----------------
    >>> y_true = torch.tensor([0,1,2,2,1,0])
    >>> y_score = torch.tensor([[0.5, 0.2, 0.3], # 0 in top 2
                                [0.1, 0.5, 0.4], # 1 in top 2
                                [0.3, 0.2, 0.5], # 2 not in top 2
                                [0.5, 0,2, 0.3], # 2 not in top 2
                                [0.7, 0.1, 0.2], # 1 not in top 2
                                [0.8, 0.2, 0.0]])# 0 in top 2
    >>> result = top_k_accuracy_score_inclass(y_true,y_score,k=2)
    [1.0,0.5,0]

    '''

    if (not isinstance(y_true,torch.Tensor)) and (not isinstance(y_true,np.ndarray)):
        raise ValueError(
        f"y-true type must be 'torch.Tensor' or 'np.ndarray', got '{type(y_true)}' instead"
        )
    if (not isinstance(y_score,torch.Tensor)) and (not isinstance(y_score,np.ndarray)):
        raise ValueError(
        f"y-true type must be 'torch.Tensor' or 'np.ndarray', got '{type(y_score)}' instead"
        )
    if len(y_true) != len(y_score):
        raise ValueError(
        f"y-true and y-score are not the same length, got y-true is '{len(y_true)}',y-score is '{len(y_score)}'"
        )
    if max(y_true) != len(y_score[0])-1:
        raise ValueError(
        f"y-true has '{max(y_true)}' class, but y-score has '{len(y_score[0])}' from output"
        )
    
    if softmax and isinstance(y_score,torch.Tensor):
        y_score = torch.nn.functional.softmax(y_score, dim=1)

    if softmax and isinstance(y_score,np.ndarray):
        y_score = y_score #FIXME: Find how to softmax a numpay shape != (1,)

    if isinstance(y_true,torch.Tensor):
        y_true = y_true.cpu().detach().numpy()

    if isinstance(y_score,torch.Tensor):
        y_score = y_score.cpu().detach().numpy()

    top_k_accuracy = []
        
    for label in range(max(y_true)+1):
        g_pred = []
        sum = 0
        for idx in range(len(y_true)):
            if y_true[idx] == label:
                g_pred.append(y_score[idx])

        g_pred = np.array([np.array(xi) for xi in g_pred])

        temp_pred = g_pred.argsort(axis=1)
        rank_pred = temp_pred.argsort(axis=1)

        for val in rank_pred:
            if val[label] > max(y_true) - k:
                sum+=1
            
        acc = sum / len(rank_pred)
        top_k_accuracy.append(acc)
    
    return top_k_accuracy


if __name__ == '__main__':
    y_true = torch.tensor([0,1,2,2,1,0])
    y_score = torch.tensor([[0.5, 0.2, 0.3], # 0 in top 2
                            [0.1, 0.5, 0.4], # 1 in top 2
                            [0.3, 0.6, 0.1], # 2 not in top 2
                            [0.5, 0.3, 0.2], # 2 not in top 2
                            [0.7, 0.1, 0.2], # 1 not in top 2
                            [0.8, 0.2, 0.0]])# 0 in top 2

    result = top_k_accuracy_score_inclass(y_true, y_score, k=2)
    print(result)