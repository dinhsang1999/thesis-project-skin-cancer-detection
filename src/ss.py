import torch
import numpy as np
from sklearn.metrics import confusion_matrix

#https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl
def ss(y_gt,y_pred):

    cm = confusion_matrix(y_gt, y_pred)

    tp = np.diag(cm)

    fp = []
    for i in range(max(y_gt)+1):
        fp.append(sum(cm[:,i]) - cm[i,i])

    fn = []
    for i in range(max(y_gt)+1):
        fn.append(sum(cm[i,:]) - cm[i,i])

    tn = []
    for i in range(max(y_gt)+1):
        temp = np.delete(cm, i, 0)   # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        tn.append(sum(sum(temp)))

    sensitivity = []
    specificity = []

    for i in range(max(y_gt)+1):
        sens = tp[i]/(tp[i]+fn[i])
        spec = tn[i]/(tn[i]+fp[i])
        sensitivity.append(sens)
        specificity.append(spec)

    return sensitivity,specificity