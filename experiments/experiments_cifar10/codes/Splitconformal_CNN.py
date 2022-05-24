import numpy as np
import pandas as pd
import random
import torch
from scipy.stats.mstats import mquantiles
import pdb
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

seed = 666
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


class ProbAccum:
    def __init__(self, prob):
        prob = np.clip(prob,1e-6,1.0-2e-6)
        prob += np.random.uniform(low=0.0, high=1e-6, size=prob.shape)
        prob /= np.sum(prob,1)[:,None]
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        #self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)        
        
    def predict_sets(self, alpha, randomize=False, epsilon=None):
        L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        idx_ra0 = np.where(L==0)[0]
        if randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        if epsilon is not None:
            epsilon[idx_ra0] = 1
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            p_remove = np.clip(p_remove, 0, 1)
            remove = np.where(epsilon < p_remove)[0]
            L[remove] -= 1
            # Make sure no prediction sets are empty
            L[L<0] = 0
        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        
        if epsilon is not None:
          # This will give exact marginal coverage
          # idx_ra0 = np.where(ranks==0)[0]
          # epsilon[idx_ra0] = 1
          alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob

        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max


class SplitConformal:
  def __init__(self, bbox=None):
    if bbox is not None:
      self.bbox = bbox
    
  def fit(self, X, Y, bbox=None):
    # Store the black-box
    if bbox is not None:
      self.bbox = bbox

    # Fit model
    self.bbox.fit(X, Y)


  def calibrate(self, calib_loader, alpha, bbox=None, return_scores=False, no_calib=False):
    if bbox is not None:
        self.bbox = bbox

    # Form prediction sets on calibration data
    # p_hat_calib = self.bbox.predict_proba(calib_loader) # first shuffle calib_loader
    p_hat_calib, Y = self.bbox.predict_proba(calib_loader, return_y_true = True)
    
    #print(cal_loader.sampler.data_source.indices[:10])
    grey_box = ProbAccum(p_hat_calib)

    n2 = len(calib_loader.dataset)
  
    epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
    scores = grey_box.calibrate_scores(Y, epsilon=epsilon)
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    tau = mquantiles(1.0-scores, prob=level_adjusted)[0]

    if return_scores:
      self.scores = scores

    # Store calibrate level
    self.alpha_calibrated = 1.0 - tau

    if no_calib:
      print('no calibration')
      self.alpha_calibrated = alpha
    print("Calibrated alpha nominal {:.3f}: {:.3f}".format(alpha, self.alpha_calibrated))


  def predict(self, data_loader, alpha=None, epsilon=None):
    n = len(data_loader.dataset)
    if epsilon is None:
      epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
    p_hat = self.bbox.predict_proba(data_loader)
    grey_box = ProbAccum(p_hat)
    if alpha is None:
      alpha = self.alpha_calibrated
    S_hat = grey_box.predict_sets(alpha, epsilon=epsilon)
    return S_hat




class ConfDataset_filter(Dataset):
    def __init__(self, dataset_XYZ, F, scale=(0.5, 0.5), ratio=(0.3, 3.3)):
        self.daztaset_XYZ = dataset_XYZ
        self.F = F
        self.filter = T.RandomErasing(p=1, scale=scale, ratio=ratio, value=0, inplace=False)
        assert len(self.dataset_XYZ) == len(self.F)
    
    def __getitem__(self, index):
        X, Y, Z = self.dataset_XYZ[index]
        F = self.F[index]
        if F == 1:
            X = self.filter(X)

        return X, Y, Z, F
      
    def __len__(self):
        return self.F.shape[0]

class ConfDataset_filter_no_Z(Dataset):
    def __init__(self, dataset_XY, F, scale=(0.5, 0.5), ratio=(0.3, 3.3)):
        self.dataset_XY = dataset_XY
        self.F = F
        self.filter = T.RandomErasing(p=1, scale=scale, ratio=ratio, value=0, inplace=False)
        assert len(self.dataset_XY) == len(self.F)
    
    def __getitem__(self, index):
        X, Y = self.dataset_XY[index]
        F = self.F[index]
        if F == 1:
          X = self.filter(X)

        return X, Y, F
      
    def __len__(self):
        return self.F.shape[0]



def blurring_images(test_dataset_to_blur, scale=(0.5,0.5), ratio=(0.3, 3.3), batch_size = 5, corrupt_percent = 0, plot = False):

  F_test = np.zeros(len(test_dataset_to_blur))
  F_test[:int(len(test_dataset_to_blur)*corrupt_percent)] = 1
  np.random.shuffle(F_test)
  test_dataset_blurred = ConfDataset_filter_no_Z(test_dataset_to_blur, F_test, scale=scale, ratio=ratio)
  
  if plot:
    indices_temp = torch.randperm(len(test_dataset_blurred))[:10]
    test_dataset_blurred_temp = torch.utils.data.Subset(test_dataset_blurred, indices_temp)
    test_loader_blurred_temp = torch.utils.data.DataLoader(test_dataset_blurred_temp, batch_size=batch_size,shuffle=False, num_workers=2)
    plot_blur(test_loader_blurred_temp)

  return(test_dataset_blurred, F_test)



def plot_blur(test_loader_blurred):

  for idx, (X_batch,_ , F_batch) in enumerate(test_loader_blurred):
    assert(len(F_batch) == len(X_batch))
    plt.figure()
    # plt.imshow(torchvision.utils.make_grid(X_batch * std[None,...,None,None] + mean[None,...,None,None]).permute(1,2,0))
    plt.imshow(torchvision.utils.make_grid(X_batch).permute(1,2,0))
    plt.title((' '*16).join([str(int(x)) for x in F_batch.data.numpy()]))
      
    if idx > 20:
      break