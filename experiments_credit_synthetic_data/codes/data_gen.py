# Data generatiing model

import torch
import numpy as np


class Model_Ex1:
    def __init__(self, K, p, delta_1, delta_2, a=1):
        self.K = K
        self.p = p
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.a = a

    def sample_X(self, n, test=False):
        X = np.random.uniform(0, 1, (n,self.p))
        if test:
          X[:,6] = np.random.uniform(0, self.a, (n,))
        return X.astype(np.float32)

    def compute_prob(self, X):
        K = self.K
        X = X[:,2:]
        num_features = X.shape[1]
        P = np.zeros((X.shape[0],K))
        for i in range(X.shape[0]):
#          if (np.maximum(X[i,0],X[i,1])<np.sqrt(0.1)):
          if (X[i,0]<self.delta_1):
            # 10% of the samples should have completely unpredictable labels
            P[i,:] = 1.0/K
          else:
            # The remaining samples belong to one of two different groups (balanced)
            K_half = np.ceil(K/2).astype(int)
            if (X[i,2]<0.5):
              # Group 1: labels 0,1,..,ceiling(K/2)-1
              if (X[i,4]<self.delta_2):
                # 20% of these samples should have completely unpredictable labels (within group)
                P[i,0:K_half] = 1.0/K_half
              else:
                # The remaining samples should have labels determined by one feature
                idx = np.round(K*X[i,10]-0.5).astype(int)
                P[i,idx] = 1
            else:
              # Group 2: labels ceiling(K/2),...,K
              if (X[i,4]<self.delta_2):
                  # 20% of these samples should have completely unpredictable labels (within group)+
                P[i,K_half:K] = 1.0/(K-K_half)
              else:
                # The remaining samples should have labels determined by one feature
                idx = np.round(K*X[i,10]-0.5).astype(int)
                P[i,idx] = 1            

        prob = P
        prob_y = prob / np.expand_dims(np.sum(prob,1),1)
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1,prob_y[i]) for i in range(X.shape[0])], dtype = float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i],classes_id) for i in range(X.shape[0])], dtype = int)
        return y.astype(int)
       

       
class Oracle:
    def __init__(self, model):
        self.model = model
    
    def fit(self,X,y):
        return self

    def predict(self, X):
        return self.model.sample_Y(X)        

    def predict_proba(self, X):
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        prob = self.model.compute_prob(X)
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob


def difficulty_oracle(S_oracle, size_cutoff=1):
  size_oracle = np.array([len(S) for S in S_oracle])
  easy_idx = np.where(size_oracle<=size_cutoff)[0]
  hard_idx = np.where(size_oracle>size_cutoff)[0]
  return easy_idx, hard_idx