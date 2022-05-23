import pandas as pd
import numpy as np
import math 
from scipy.stats.mstats import mquantiles
from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchsort
from torchsort import soft_rank, soft_sort
import copy
import os 
from tqdm.autonotebook import tqdm
from codes.resnet import ResNet18
import pickle 
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# modify the dataset to add corruption flag
class ConfDataset(Dataset):
    def __init__(self, dataset_XY, Z):
        self.dataset_XY = dataset_XY
        self.Z = Z
    
    def __getitem__(self, index):
        X, Y = self.dataset_XY[index]
        Z = self.Z[index]
        return X, Y, Z
      
    def __len__(self):
        return self.Z.shape[0]

def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc*100

def eval_predictions(test_loader, box, data="unknown", plot=False, predict_1=False):
    if predict_1:
        Y_pred = box.predict_1(test_loader)
    else:
        Y_pred = box.predict(test_loader)
    
    Y_true = []
    for X_batch, Y_batch, _ in test_loader:
      Y_true.append(Y_batch.cpu().numpy()[0])
    
    if plot:
        A = confusion_matrix(Y, Y_pred)
        df_cm = pd.DataFrame(A, index = [i for i in range(K)], columns = [i for i in range(K)])
        plt.figure(figsize = (4,3))
        pal = sns.light_palette("navy", as_cmap=True)
        sn.heatmap(df_cm, cmap=pal, annot=True, fmt='g')

    class_error = np.mean(Y_true!=Y_pred)
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

# soft sorting and ranking
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


# Conformal Loss function
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

# FocalLoss Loss function
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
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

class FocalLossAdaptive(nn.Module):
    def __init__(self, gamma=0, size_average=False, device=None):
        super(FocalLossAdaptive, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.device = device

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(self.device)

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
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# BlackBox method
class BlackBox:

    def __init__(self, num_features, num_classes,
                 family="classification", 
                 baseloss='crossentropy',
                 model = 'resnet18',
                 dropout=False):

        self.num_features = num_features
        self.num_classes = num_classes
        self.family = family
        self.baseloss = baseloss

        if model == 'resnet18':
          self.model = ResNet18()
        elif model == 'resnet50':
          self.model = ResNet50()

        # Detect whether CUDA is available
        self.device = device
        self.model = self.model.to(self.device)

        # Define loss functions
        if self.family=="classification":
          if self.baseloss == 'crossentropy':
            self.criterion_pred = nn.CrossEntropyLoss()
          elif self.baseloss =='focalloss':
            self.criterion_pred = FocalLoss(gamma = 3, size_average = True)


        else:
          self.criterion_pred = nn.MSELoss()
          
        self.criterion_scores = UniformMatchingLoss()
                
        self.layer_prob = nn.Softmax(dim=1)

    def compute_loss_scores(self, y_train_pred, y_train_batch, alpha=0.1):
        train_proba = self.layer_prob(y_train_pred)
        train_scores, train_sizes = compute_scores_diff(train_proba, y_train_batch, alpha=alpha)
        train_loss_scores = self.criterion_scores(train_scores)
        train_loss_sizes = torch.mean(train_sizes)
        return train_loss_scores, train_loss_sizes

    def fit(self, train_loader, Z_train = None, estop_loader=None,
            num_epochs=10, batch_size=16, lr=0.001, optimizer = 'Adam', 
            mu=0, mu_size=0, alpha=0.1,  
            save_checkpoint_period=1, save_model=True, name=None, 
            early_stopping=False, name_CP=None):            

        # Process input arguments
        if save_model:
          if name is None:
            raise("Output model name file is needed.")

        # Choose the optimizer
        if optimizer == 'Adam':
          optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'SGD':
          optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Choose the learning rate scheduler
        lr_milestones = [int(num_epochs*0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)


        if Z_train is not None:
          # Check whether conformity scores need to be evaluated on training data
          if np.sum(np.unique(Z_train)>0) > 0:
            eval_conf_train = True
          else:
            eval_conf_train = False
        else:
          eval_conf_train = False
        
        
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
        print("Begin training.")
        for e in tqdm(range(1, num_epochs+1)):

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
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                Z_batch = Z_batch.to(device)
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
                  X_batch = X_batch.to(device)
                  Y_batch = Y_batch.to(device)
                  Z_batch = Z_batch.to(device)

                  out = self.model(X_batch)

                  # Evaluate CE loss
                  loss_ce = self.criterion_pred(out, Y_batch)

                  # Samples in other groups are processed by the uniform matching loss
                  Z_groups = torch.unique(Z_batch)
                  loss_scores = torch.tensor(0.0, device=device)
                  loss_sizes = torch.tensor(0.0, device=device)
                  n_groups = torch.sum(Z_groups>0)
                  #pdb.set_trace()
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
                          stats=stats,  
                          best_loss=best_loss,
                          model_state=self.model.state_dict(),
                      )
                      torch.save(saved_state, name_CP+'loss')
                      print(
                          f"*** Saved checkpoint loss at epoch {e}", flush=True
                      )
                      
                  # Early stopping by accuracy
                  save_checkpoint = True if best_acc is not None and best_acc < epoch_acc_hout and e % save_checkpoint_period == 0 else False
                  best_acc = epoch_acc_hout if best_acc is None or best_acc < epoch_acc_hout else best_acc
                  # Save model checkpoint if requested
                  if save_checkpoint:
                      saved_state = dict(
                          stats=stats,
                          best_acc=best_acc,
                          model_state=self.model.state_dict(),
                      )
                      torch.save(saved_state, name_CP+'acc')
                      print(
                          f"*** Saved checkpoint acc at epoch {e}", flush=True
                      )

                  # # Early stopping by uniformity of conformity scores
                  # save_checkpoint = True if best_conf is not None and best_conf > epoch_loss_scores_hout and e % save_checkpoint_period == 0 else False
                  # best_conf = epoch_loss_scores_hout if best_conf is None or best_conf > epoch_loss_scores_hout else best_conf
                  # # Save model checkpoint if requested
                  # if save_checkpoint:
                  #     saved_state = dict(
                  #         best_conf=best_conf,
                  #         model_state=self.model.state_dict(),
                  #     )
                  #     torch.save(saved_state, name_CP+'conf')
                  #     print(
                  #         f"*** Saved checkpoint conf at epoch {e}", flush=True
                  #     )

                  # # Early stopping by size of 90% prediction sets
                  # save_checkpoint = True if best_size is not None and best_size > epoch_loss_sizes_hout and e % save_checkpoint_period == 0 else False
                  # best_size = epoch_loss_sizes_hout if best_size is None or best_size > epoch_loss_sizes_hout else best_size
                  # # Save model checkpoint if requested
                  # if save_checkpoint:
                  #     saved_state = dict(
                  #         best_size=best_size,
                  #         model_state=self.model.state_dict(),
                  #     )
                  #     torch.save(saved_state, name_CP+'size')
                  #     print(
                  #         f"*** Saved checkpoint size at epoch {e}", flush=True
                  #     )
                                
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

    def predict(self, test_loader, return_y_true = None):

        y_pred_list = []
        y_true = []

        with torch.no_grad():
            self.model.eval()
            for X_batch, Y_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_true.append(Y_batch.cpu().numpy()[0])
                if self.family=="classification":
                  y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                  _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                  y_pred_list.append(y_pred_tags.cpu().numpy())
                else:
                  y_pred_list.append(y_test_pred.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        
        if return_y_true:
          return (y_pred, np.array(y_true))
        else:
          return y_pred

    def predict_proba(self, test_loader, return_y_true = None):

        y_proba_list = []
        y_true = []

        with torch.no_grad():
            self.model.eval()
            for X_batch, Y_batch, _ in test_loader:
                y_true.append(Y_batch.cpu().numpy()[0])
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_proba_softmax = torch.softmax(y_test_pred, dim = 1)
                y_proba_list.append(y_proba_softmax.cpu().numpy())
        prob = np.concatenate(y_proba_list)
        prob = prob / prob.sum(axis=1)[:,None]

        if return_y_true:
          return (prob, np.array(y_true))

        else:
          return prob


class BlackBox2:

    def __init__(self, num_features, num_classes, box_extract_features=None, feature_augmentation=True):
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_augmentation = feature_augmentation

        # Define the two separate black-boxes 
        self.box_1 = BlackBox(num_features, num_classes)
        if feature_augmentation:
          num_features_2 = num_features + self.box_1.model.z_dim
        else:
          num_features_2 = num_features
        self.box_2 = BlackBox(num_features_2, num_classes)
        if box_extract_features is not None:
            self.box_extract_features = box_extract_features.box_1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def fit_1(self, train_loader, estop_loader, optimizer = 'Adam', 
              num_epochs=10, batch_size=16, lr=0.001, mu=0, mu_size=0, alpha=0.1,
              save_model=False, save_checkpoint_period=1, name=None, early_stopping=False, name_CP=None):            
        
        stats = self.box_1.fit(train_loader, estop_loader=estop_loader, 
                               num_epochs=num_epochs, batch_size=batch_size, lr=lr,
                               mu=mu, mu_size=mu_size, alpha=0.1, optimizer = optimizer, 
                               save_model=save_model, name=name, early_stopping=early_stopping, 
                               save_checkpoint_period=save_checkpoint_period, name_CP=name_CP+'_init')
        return stats
            
    
    def fit_2(self, train_loader, estop_loader=None,
              Z_train = None, optimizer = 'Adam',
              num_epochs=10, batch_size=16, lr=0.001, 
              mu=0, mu_size=0, alpha=0.1,
              save_model=False, save_checkpoint_period=1, name=None, early_stopping=False, name_CP=None):            
        
        stats = self.box_2.fit(train_loader, estop_loader=estop_loader,
                               Z_train=Z_train, optimizer = optimizer, 
                               num_epochs=num_epochs, batch_size=batch_size, lr=lr, 
                               mu=mu, mu_size=mu_size, alpha=alpha,
                               save_model=save_model, name=name, early_stopping=early_stopping, 
                               save_checkpoint_period=save_checkpoint_period, name_CP=name_CP)

        return stats

    def predict_1(self, teat_loader, return_y_true = None):
        return self.box_1.predict(teat_loader, return_y_true = return_y_true)

    def predict(self, teat_loader, return_y_true = None):

        return self.box_2.predict(teat_loader, return_y_true = return_y_true)

    def predict_proba_1(self, teat_loader, return_y_true = None):
        return self.box_1.predict_proba(teat_loader, return_y_true = return_y_true)

    def predict_proba(self, teat_loader, return_y_true = None):

        return self.box_2.predict_proba(teat_loader, return_y_true = return_y_true)

