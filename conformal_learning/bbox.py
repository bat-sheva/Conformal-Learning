# BBox and net 
   
import sys
import torch
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append('./third_party/')
if device.type == 'cuda': 
  import torchsort
  from torchsort import soft_rank, soft_sort 
else:
  from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
    
    
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, z_data):
        self.X_data = X_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.z_data[index]

    def __len__ (self):
        return len(self.X_data)
     
     
def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc*100
    
    
class ClassNNet(nn.Module):
    def __init__(self, num_features, num_classes, use_dropout=False):
        super(ClassNNet, self).__init__()

        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, num_classes)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm3(x)
        x = self.layer_4(x)
        x = self.relu(x)
        
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)
           
        if extract_features:
          return x, torch.cat([z2,z3],1)
        else:
          return x
          
          
          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REG_STRENGTH = 0.1
B = 50        


def soft_indicator(x, a, b=B):
  def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
  out = torch.sigmoid(b*(x-a+0.5)) - (torch.sigmoid(b*(x-a-0.5)))
  out = out / (sigmoid(b*(0.5)) - (sigmoid(b*(-0.5))) )
  return out

def soft_indexing(z, rank):
    n = len(rank)
    K = z.shape[1]
    I = torch.tile(torch.arange(K, device=device), (n,1))
    # Note: this function is vectorized to avoid a loop
    weight = soft_indicator(I.T, rank).T
    weight = weight * z
    return weight.sum(dim=1)

def compute_scores_diff(proba_values, Y_values, alpha=0.1):
    """
    Compute the conformity scores and estimate the size of 
    the conformal prediction sets (differentiable) 
    """
    n, K = proba_values.shape
    # Break possible ties at random (it helps with the soft sorting)
    proba_values = proba_values + 1e-6*torch.rand(proba_values.shape,dtype=float,device=device)
    # Normalize the probabilities again
    proba_values = proba_values / torch.sum(proba_values,1)[:,None]
    # Sorting and ranking
    ranks_array_t = soft_rank(-proba_values, regularization_strength=REG_STRENGTH)-1
    prob_sort_t = -soft_sort(-proba_values, regularization_strength=REG_STRENGTH)
    # Compute the CDF
    Z_t = prob_sort_t.cumsum(dim=1)
    # Collect the ranks of the observed labels
    ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(n,1)).flatten()
    # Compute the PDF at the observed labels
    prob_cum_t = soft_indexing(Z_t, ranks_t)
    # Compute the PMF of the observed labels
    prob_final_t = soft_indexing(prob_sort_t, ranks_t)
    # Compute the conformity scores
    scores_t = 1.0 - prob_cum_t + prob_final_t * torch.rand(n,dtype=float,device=device)
    # Note: the following part is new
    # Sort the conformity scores
    n = len(scores_t)
    scores_sorted_t = soft_sort(1.0-scores_t.reshape((1,n))).flatten()
    # Compute the 1-alpha quantile of the conformity scores
    scores_q_t = scores_sorted_t[int(n*(1.0-alpha))]
    _, sizes_t = torch.max(Z_t>=scores_q_t,1)
    sizes_t = sizes_t + 1.0
    # Return the conformity scores and the estimated sizes (without randomization) at the desired alpha
    return scores_t, sizes_t
    
    
class UniformMatchingLoss(nn.Module):
  """ Custom loss function
  """
  def __init__(self):
    """ Initialize
    Parameters
    batch_size : number of samples in each batch
    """
    super().__init__()

  def forward(self, x):
    """ Compute the loss
    Parameters
    ----------
    x : pytorch tensor of random variables (n)
    Returns
    -------
    loss : cost function value
    """
    batch_size = len(x)
    if batch_size == 0:
      return 0
    # Soft-sort the input
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
    i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
    out = torch.max(torch.abs(i_seq - x_sorted))
    return out
    
    
    
'''
Implementation of Focal Loss.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
    
class BlackBox:

    def __init__(self, num_features, num_classes,
                 family="classification", dropout=False, base_loss='CE', gamma=1):
        self.num_features = num_features
        self.num_classes = num_classes
        self.family = family
                
        # Define NNet model
        self.model = ClassNNet(num_features = num_features, num_classes=num_classes, 
                               use_dropout=dropout)

        # Detect whether CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Define loss functions
        if self.family=="classification":
          if base_loss=='CE':
            self.criterion_pred = nn.CrossEntropyLoss()
          if base_loss=='Focal':
            self.criterion_pred = FocalLoss(gamma=gamma)
        else:
          self.criterion_pred = nn.MSELoss()
        self.criterion_scores = UniformMatchingLoss()

        # How to compute probabilities
        self.layer_prob = nn.Softmax(dim=1)

    def compute_loss_scores(self, y_train_pred, y_train_batch, alpha=0.1):
        train_proba = self.layer_prob(y_train_pred)
        train_scores, train_sizes = compute_scores_diff(train_proba, y_train_batch, alpha=alpha)
        train_loss_scores = self.criterion_scores(train_scores)
        train_loss_sizes = torch.mean(train_sizes)
        return train_loss_scores, train_loss_sizes

    def fit(self, X_train, Y_train, Z_train = None, X_hout = None, Y_hout = None,
            num_epochs=10, batch_size=16, lr=0.001, 
            mu=0, mu_size=0, alpha=0.1, optimizer='Adam', cond_label=False,
            save_checkpoint_period=1, save_model=True, name=None, 
            early_stopping=False, name_CP=None, verbose=True):            
        """
        X_train : feature matrix for input data used for training
        Y_train : label vector for input data used for training
        Z_train : group membership vector for input data used for training (optional)
                  This determines how to group observations for conformal loss.
                  Group 0 does not get processed by conformal loss. Default: all 0.
        X_hout : feature matrix for input data used for hold-out evaluation (optional)
        Y_hout:  label vector for input data used for hold-out evaluation (optional)
        """

        # Process input arguments
        if save_model:
          if name is None:
            raise("Output model name file is needed.")


        # Choose the optimizer
        if optimizer=='Adam':
          optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if optimizer=='SGD':
          optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Choose the learning rate scheduler
        lr_milestones = [int(num_epochs*0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

        # Initialize loader for training data 
        X_train = torch.from_numpy(X_train).float().to(self.device)
        if self.family=="classification":
          Y_train = torch.from_numpy(Y_train).long().to(self.device)
        else:
          Y_train = torch.from_numpy(Y_train).float().to(self.device)

        if Z_train is not None:
          # Check whether conformity scores need to be evaluated on training data
          if np.sum(np.unique(Z_train)>0) > 0:
            eval_conf_train = True
          else:
            eval_conf_train = False
          Z_train = torch.from_numpy(Z_train).long().to(self.device)
        else:
          Z_train = torch.zeros(Y_train.shape).long().to(self.device)
          eval_conf_train = False
        
        train_dataset = ClassifierDataset(X_train, Y_train, Z_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, drop_last=True)

        # Initialize loader for hold-out data (if available)
        if ((X_hout is not None) and (Y_hout is not None)): 
          X_hout = torch.from_numpy(X_hout).float().to(self.device)
          if self.family=="classification":
            Y_hout = torch.from_numpy(Y_hout).long().to(self.device)
            Z_hout = torch.ones(Y_hout.shape).long().to(self.device)
          else:
            Y_hout = torch.from_numpy(Y_hout).float().to(self.device)
            Z_hout = torch.zeros(Y_hout.shape).long().to(self.device)

          estop_dataset = ClassifierDataset(X_hout, Y_hout, Z_hout)
          estop_loader = DataLoader(dataset=estop_dataset, batch_size=batch_size, 
                                    shuffle=True, drop_last=True)
        else:
          estop_loader = None
        
        # Initialize monitoring variables
        stats = {'epoch': [], 
                 'pred': [], "scores": [], "sizes": [], "loss": [], 'acc' : [],
                 'pred-estop':[], 'scores-estop':[], 'sizes-estop':[], 'loss-estop':[], 'acc-estop':[]
                 } 
        best_loss = None
        best_acc = None
        best_conf = None
        best_size = None

        # Training loop
        print("Begin training.", flush=True)
        for e in range(1, num_epochs+1):

            epoch_acc = 0
            epoch_loss_ce = 0
            epoch_loss_scores = 0
            epoch_loss_sizes = 0
            epoch_loss = 0
            epoch_acc_hout = 0
            epoch_loss_ce_hout = 0
            epoch_loss_scores_hout = 0
            epoch_loss_sizes_hout = 0
            epoch_loss_hout = 0
            
            self.model.train()

            for X_batch, Y_batch, Z_batch in train_loader:
                optimizer.zero_grad()

                # Important: make sure all the batches have the same size, because otherwise
                # the uniform matching loss will not work well.
                assert(len(Z_batch)==batch_size)

                # Compute model output
                out = self.model(X_batch)
           
                # Samples in group (Z = 0) are processed by the cross-entropy loss
                idx_ce = torch.where(Z_batch==0)[0]
                loss_ce = self.criterion_pred(out[idx_ce], Y_batch[idx_ce])
                
                # Samples in other groups are processed by the uniform matching loss
                loss_scores = torch.tensor(0.0, device=device)
                loss_sizes = torch.tensor(0.0, device=device)
                if (eval_conf_train and ((mu > 0) or (mu_size>0))):
                  Z_groups = torch.unique(Z_batch)
                  n_groups = torch.sum(Z_groups>0)
                  for z in Z_groups:
                    if z > 0:
                      idx_z = torch.where(Z_batch==z)[0]
                      n_z = len(idx_z)
                      if cond_label:
                        Y_batch_z = Y_batch[idx_z]
                        out_z = out[idx_z]
                        idx_hard_scores = torch.nonzero(Y_batch_z).squeeze(1)
                        idx_easy_scores = torch.nonzero(Y_batch_z-1).squeeze(1)
                        loss_scores_z_hard, loss_sizes_z_hard = self.compute_loss_scores(out_z[idx_hard_scores], Y_batch_z[idx_hard_scores], alpha=alpha)
                        loss_scores_z_easy, loss_sizes_z_easy = self.compute_loss_scores(out_z[idx_easy_scores], Y_batch_z[idx_easy_scores], alpha=alpha)
                        loss_scores_z = loss_scores_z_hard + loss_scores_z_easy
                        loss_sizes_z = loss_sizes_z_hard + loss_sizes_z_easy
                      else:
                        loss_scores_z, loss_sizes_z = self.compute_loss_scores(out[idx_z], Y_batch[idx_z], alpha=alpha)
                      loss_scores = loss_scores + loss_scores_z
                      loss_sizes = loss_sizes + loss_sizes_z
                  loss_scores = loss_scores / n_groups
                  loss_sizes = loss_sizes / n_groups

                # Compute total loss
                loss = loss_ce
                if mu > 0:
                   loss = loss + loss_scores * mu
                if mu_size > 0:
                   loss = loss + loss_sizes * mu_size
                
                # Take gradient step
                loss.backward()
                optimizer.step()
                
                # Store information
                acc = accuracy_point(out[idx_ce], Y_batch[idx_ce])
                epoch_acc += acc.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_scores += loss_scores.item()
                epoch_loss_sizes += loss_sizes.item()
                epoch_loss += loss.item()

            epoch_acc /= len(train_loader)
            epoch_loss_ce /= len(train_loader)
            epoch_loss_scores /= len(train_loader)
            epoch_loss_sizes /= len(train_loader)
            epoch_loss /= len(train_loader)
            
            scheduler.step()
            self.model.eval()

            if estop_loader is not None:
              for X_batch, Y_batch, Z_batch in estop_loader:
                  out = self.model(X_batch)

                  # Evaluate CE loss
                  loss_ce = self.criterion_pred(out, Y_batch)

                  # Samples in other groups are processed by the uniform matching loss
                  Z_groups = torch.unique(Z_batch)
                  loss_scores = torch.tensor(0.0, device=device)
                  loss_sizes = torch.tensor(0.0, device=device)
                  n_groups = torch.sum(Z_groups>0)
                  for z in Z_groups:
                    if z > 0:
                      idx_z = torch.where(Z_batch==z)[0]
                      loss_scores_z, loss_sizes_z = self.compute_loss_scores(out[idx_z], Y_batch[idx_z])
                      loss_scores += loss_scores_z
                      loss_sizes += loss_sizes_z
                  loss_scores /= n_groups
                  loss_sizes /= n_groups

                  # Compute total loss
                  loss = loss_ce
                  if mu > 0:
                    loss += loss_scores * mu
                  if mu_size > 0:
                    loss += mu_size * loss_sizes

                  # Store information
                  acc = accuracy_point(out, Y_batch)
                  epoch_acc_hout += acc.item()
                  epoch_loss_ce_hout += loss_ce.item()
                  epoch_loss_scores_hout += loss_scores.item()
                  epoch_loss_sizes_hout += loss_sizes.item()
                  epoch_loss_hout += loss.item()

              epoch_acc_hout /= len(estop_loader)
              epoch_loss_ce_hout /= len(estop_loader)
              epoch_loss_scores_hout /= len(estop_loader)
              epoch_loss_sizes_hout /= len(estop_loader)
              epoch_loss_hout /= len(estop_loader)

              if early_stopping:
                  
                  # Early stopping by CE loss
                  save_checkpoint = True if best_loss is not None and best_loss > epoch_loss_hout and e % save_checkpoint_period == 0 else False
                  best_loss = epoch_loss_hout if best_loss is None or best_loss > epoch_loss_hout else best_loss
                  # Save model checkpoint if requested
                  if save_checkpoint:
                      saved_state = dict(
                          best_loss=best_loss,
                          model_state=self.model.state_dict(),
                      )
                      torch.save(saved_state, name_CP+'loss')
                      if verbose:
                          print(
                              f"*** Saved checkpoint loss at epoch {e}", flush=True
                          )
                      
                  # Early stopping by accuracy
                  save_checkpoint = True if best_acc is not None and best_acc < epoch_acc_hout and e % save_checkpoint_period == 0 else False
                  best_acc = epoch_acc_hout if best_acc is None or best_acc < epoch_acc_hout else best_acc
                  # Save model checkpoint if requested
                  if save_checkpoint:
                      saved_state = dict(
                          best_acc=best_acc,
                          model_state=self.model.state_dict(),
                      )
                      torch.save(saved_state, name_CP+'acc')
                      if verbose:
                          print(
                              f"*** Saved checkpoint acc at epoch {e}", flush=True
                          )

                                
            stats['epoch'].append(e)
            stats['pred'].append(epoch_loss_ce)
            stats['scores'].append(epoch_loss_scores)
            stats['sizes'].append(epoch_loss_sizes)
            stats['loss'].append(epoch_loss)
            stats['acc'].append(epoch_acc)
            stats['acc-estop'].append(epoch_acc_hout)
            stats['pred-estop'].append(epoch_loss_ce_hout)
            stats['scores-estop'].append(epoch_loss_scores_hout)
            stats['sizes-estop'].append(epoch_loss_sizes_hout)
            stats['loss-estop'].append(epoch_loss_hout)

            
            if verbose:
                print(f'Epoch {e+0:03}: | CE: {epoch_loss_ce:.3f} | ', end='')
                if eval_conf_train > 0:
                  print(f'Scores: {epoch_loss_scores:.3f} | ', end='')
                  print(f'Sizes: {epoch_loss_sizes:.3f} | ', end='')
                print(f'Loss: {epoch_loss:.3f} | ', end='')
                print(f'Acc: {epoch_acc:.3f} | ', end='')
                if estop_loader is not None:
                  print(f'CE-ho: {epoch_loss_ce_hout:.3f} | ', end='')
                  print(f'CS-ho: {epoch_loss_scores_hout:.3f} | ', end='')
                  print(f'Sizes-ho: {epoch_loss_sizes_hout:.3f} | ', end='')
                  print(f'Loss-ho: {epoch_loss_hout:.3f} | ', end='')
                  print(f'Acc-ho: {epoch_acc_hout:.3f}', end='')
                print('',flush=True)
            
        saved_final_state = dict(stats=stats,
                                 model_state=self.model.state_dict(),
                                 )
        if save_model:
            torch.save(saved_final_state, name)
            
        return stats

    def predict(self, X_test):
        y_test = np.zeros((X_test.shape[0],))
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), 
                                         torch.from_numpy(y_test).long(),
                                         torch.ones(y_test.shape).long())
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=100)

        y_pred_list = []
        with torch.no_grad():
            self.model.eval()
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                if self.family=="classification":
                  y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                  _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                  y_pred_list.append(y_pred_tags.cpu().numpy())
                else:
                  y_pred_list.append(y_test_pred.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        return y_pred

    def predict_proba(self, X_test):
        y_test = np.zeros((X_test.shape[0],))
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), 
                                         torch.from_numpy(y_test).long(),
                                         torch.ones(y_test.shape).long())
        test_loader = DataLoader(dataset=test_dataset, batch_size=100)

        y_proba_list = []
        with torch.no_grad():
            self.model.eval()
            for X_batch, _, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_proba_softmax = torch.softmax(y_test_pred, dim = 1)
                y_proba_list.append(y_proba_softmax.cpu().numpy())
        prob = np.concatenate(y_proba_list)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob
        
        
        

        
                
