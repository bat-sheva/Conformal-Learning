# Auxiliary functions

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def eval_predictions(X, Y, box, data="unknown", plot=False,  printing=True):

    Y_pred = box.predict(X)
    
    if plot:
        A = confusion_matrix(Y, Y_pred)
        df_cm = pd.DataFrame(A, index = [i for i in range(K)], columns = [i for i in range(K)])
        plt.figure(figsize = (4,3))
        pal = sns.light_palette("navy", as_cmap=True)
        sn.heatmap(df_cm, cmap=pal, annot=True, fmt='g')

    class_error = np.mean(Y!=Y_pred)
    if printing:
        print("Classification error on {:s} data: {:.1f}%".format(data, class_error*100))
    return (class_error*100)

def cvm(u):
  """
  Compute the Cramer von Mises statistic for testing uniformity in distribution
  """
  n = len(u)
  u_sorted = np.sort(u)
  i_seq = (2.0*np.arange(1,1+n)-1.0)/(2.0*n)
  stat = np.sum(np.square(i_seq - u_sorted)) + 1.0/(12.0*n)
  return stat

def KL(P,Q):
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(np.multiply(P,np.log(np.divide(P,Q))),1)
    return divergence