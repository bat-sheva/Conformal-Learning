# split-conformal functions

import torch
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles


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
          #idx_ra0 = np.where(ranks==0)[0]
          #epsilon[idx_ra0] = 1
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

  def calibrate(self, X, Y, alpha, bbox=None, return_scores=False, no_calib=False, print_alpha=True):
    if bbox is not None:
      self.bbox = bbox

    # Form prediction sets on calibration data
    p_hat_calib = self.bbox.predict_proba(X)
    grey_box = ProbAccum(p_hat_calib)

    n2 = X.shape[0]
    epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
    scores = grey_box.calibrate_scores(Y, epsilon=epsilon)
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    tau = mquantiles(1.0-scores, prob=level_adjusted)[0]

    # Store calibrate level
    self.alpha_calibrated = 1.0 - tau
    if no_calib:
      self.alpha_calibrated = alpha
    if print_alpha:
      print("Calibrated alpha nominal {:.3f}: {:.3f}".format(alpha, self.alpha_calibrated))


  def fit_calibrate(self, X, Y, alpha, bbox=None, random_state=2020, verbose=False):
    if bbox is not None:
      self.init_bbox(bbox)
    
    # Split data into training/calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

    # Fit black-box model
    self.fit(X_train, Y_train)

    self.calibrate(X_calib, Y_calib, alpha)

  def predict(self, X, alpha=None, epsilon=None):
    n = X.shape[0]
    if epsilon is None:
      epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
    p_hat = self.bbox.predict_proba(X)
    grey_box = ProbAccum(p_hat)
    if alpha is None:
      alpha = self.alpha_calibrated
    S_hat = grey_box.predict_sets(alpha, epsilon=epsilon)
    return S_hat


def evaluate_predictions(S, X, y, hard_idx=None, conditional=True, linear=False):
  # Marginal coverage
  marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    
  if conditional:
    y_hard = y[hard_idx]
    S_hard = [S[i] for i in hard_idx]

    # Evaluate conditional coverage
    wsc_coverage = np.mean([y_hard[i] in S_hard[i] for i in range(len(y_hard))])

    # Evaluate conditional size
    size_hard = np.mean([len(S[i]) for i in hard_idx])
    size_easy = np.mean([len(S[i]) for i in range(len(y)) if i not in hard_idx])   
    size_hard_median = np.median([len(S[i]) for i in hard_idx])
    size_easy_median = np.median([len(S[i]) for i in range(len(y)) if i not in hard_idx])   

    n_hard = len(hard_idx)
    n_easy = len(y) - len(hard_idx)
        
  else:
    wsc_coverage = None

  # Size and size conditional on coverage
  size = np.mean([len(S[i]) for i in range(len(y))])     
  size_median = np.median([len(S[i]) for i in range(len(y))])
  idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
  size_cover = np.mean([len(S[i]) for i in idx_cover])
  # Combine results
  out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                      'Size': [size], 'Size (median)': [size_median], 
                      'Size-hard': [size_hard], 'Size-easy': [size_easy],
                      'Size-hard (median)': [size_hard_median], 'Size-easy (median)': [size_easy_median],
                      'n-hard': [n_hard], 'n-easy': [n_easy],
                      'Size conditional on cover': [size_cover]})
  return out