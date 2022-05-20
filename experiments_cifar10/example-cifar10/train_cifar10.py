from datetime import date
import random
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle 
from tqdm.notebook import tqdm
import torchvision.transforms as T
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import argparse

date = date.today().strftime("%m-%d-%Y")
# user = 
# print("Program Started \t user: {} \t date: {}".format(user, date))

# import self-defined or third-party packages
import torchsort
from codes import Splitconformal_CNN, black_boxes_CNN, evaluations_CNN, resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description = "UncertaintyAware")
# Hyperparameters
## data preprocessing
parser.add_argument('--seed', default = 123, type = int, help = 'seed for data splitting & model randomization')
parser.add_argument('--method', default = 'RandomErasing', type = str, help = 'corruption method')
parser.add_argument('--train_perc', default = 0.2, type = float, help = 'percentage of corruption in training data')
parser.add_argument('--cal_perc', default = 0.2, type = float, help = 'percentage of corruption in testing/calibration data')
parser.add_argument('--scale', default = 0.7, type = float, help = 'scale of corrupted images')
## model structures and model savings
parser.add_argument('--feature_augmentation', default = False, choices = [True, False], type = bool, help = 'add feature augmentation on X')
parser.add_argument('--model', default = 'resnet18', type = str, help = 'model architecture')
## learning parameters
parser.add_argument('--batch_size', default = 768, type = int, help = 'batch size')
parser.add_argument('--num_epochs', default = 2000, type = int, help = 'total number of epochs')
parser.add_argument('--lr', default = 0.1, type = float, help = 'initial learning rate')
parser.add_argument('--optimizer', default = 'Adam', choices = ['SGD', 'Adam'], help = 'which optimizer, SGD or Adam')
parser.add_argument('--mu', default = 0.1, type = float, help = 'conformal loss weights (relative to cross entropy)')
parser.add_argument('--train_alpha',default = 0.1, type = float, help = 'coverage level for training')
parser.add_argument('--mu_size', default = 0, type = float, help = 'hybrid loss parameter')
parser.add_argument('--baseloss', default = 'crossentropy', type = str, help = 'alternative loss in place of cross entropy')
## number of data points
parser.add_argument('--n_tr_samples', default = 3000, type = int, help = 'number of data points used for training')
parser.add_argument('--n_ho_samples', default = 2000, type = int, help = 'number of data points hold out for early stoping')
parser.add_argument('--n_test_samples', default = 1000, type = int, help = 'number of data points used for testing after calibration')
parser.add_argument('--n_cal_samples', default = 1000, type = int, help = 'number of data points used for calibration')
## calibration and evaluations
parser.add_argument('--cal_alpha',default = 0.1, type = float, help = 'coverage level for calibration')
parser.add_argument('--return_scores', default = False, type = bool, help = 'return conformity scores on test dataset')
parser.add_argument('--predict_prob_only', default = False, type = bool, help = 'return predicted probabilities on test dataset')
parser.add_argument('--evaluation_condition', default = 'by_flag', type = str, help = 'condition criterion for evaluating coverage/prediction set size')
parser.add_argument('--no_calib', default = False, type = bool, help = 'if no, do not calibrate/evaluate models directly on test dataset')
parser.add_argument('--bs_times', default = 1, type = int, help = 'boostraping, 1 if no bootstrap, otherwise do bootstraps and evaluate on each subset')
## results saving
parser.add_argument('--early_stopping',default = True, choices =[True, False], type = bool, help = 'use early stopping')
parser.add_argument('--save_model',default = True, choices =[True, False], type = bool, help = 'save trained model')
parser.add_argument('--save_checkpoint_period', default = 1, type = int, help = 'checkpoint save frequency')
parser.add_argument('--save_pickle', default = False, type = str, help = 'save evaluation results in pickle')

args = parser.parse_args()
        
print(args)

#----------------------------------------- Preparing Dataset ----------------------------------------

# data augmentation
augmentation = [
        transforms.ToTensor(),
        transforms.RandomErasing(p=args.train_perc, scale = (args.scale, args.scale), ratio = (0.3, 3.3), value=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
transform = transforms.Compose(augmentation)
home_dir = os.path.expanduser('~')
data_dir = home_dir + "/data"
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train = True, download=False, transform=transform)

# prepare dataloader
num_features = 3
num_classes = 10

# set randomizations
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# take a subset of full train dataset
random_indices = torch.randperm(len(train_dataset))[:args.n_tr_samples + args.n_ho_samples]
random_indices_tr = random_indices[:args.n_tr_samples]
random_indices_ho = random_indices[args.n_tr_samples:args.n_tr_samples + args.n_ho_samples]
trainset_sample = torch.utils.data.Subset(train_dataset, random_indices_tr)

# set 1/5 of the training subset for conformal loss and 4/5 for uniform loss
if args.mu > 0 or args.mu_size >0:
    Z_train = np.zeros(len(trainset_sample))
    Z_train[:len(trainset_sample)//5] = 1
    np.random.shuffle(Z_train)
    train_sample_dataset = black_boxes_CNN.ConfDataset(trainset_sample, Z_train)
else:
    Z_train = np.zeros(len(trainset_sample))
    train_sample_dataset = black_boxes_CNN.ConfDataset(trainset_sample, Z_train)


# taking hout data and add Z indicator
houtset_sample = torch.utils.data.Subset(train_dataset, random_indices_ho)
Z_hout = np.ones(len(houtset_sample))
hout_sample_dataset = black_boxes_CNN.ConfDataset(houtset_sample, Z_hout)

train_loader = torch.utils.data.DataLoader(train_sample_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
hout_loader = torch.utils.data.DataLoader(hout_sample_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

#----------------------------------------- Training The Model ----------------------------------------
file_name = "featureaug={}_batchsize={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_corruptperc={}_seed={}_baseloss={}_ntrsamples={}_model={}".format(\
            args.feature_augmentation, args.batch_size, args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, \
            args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model)

file_final = './saved_models/trained_with_blurred_new2/final_'+file_name
name_CP = './saved_models/trained_with_blurred_new2/checkpoint_'+file_name

print(file_final)

print("start training")
print("seed {}".format(args.seed))
    
# if model already traiend, load the model
if os.path.isfile(file_final):
    print('Loading model instead of training')
    print(file_final)
    bbox = black_boxes_CNN.BlackBox(num_features, 
                                num_classes, 
                                baseloss = args.baseloss)
    saved_stats = torch.load(file_final, map_location=device)
    bbox.model.load_state_dict(saved_stats['model_state'])
    stats_bbox = saved_stats['stats']


elif os.path.isfile(name_CP+'acc'):
    print('Loading checkpoint')
    bbox = black_boxes_CNN.BlackBox(num_features, 
                                num_classes, 
                                baseloss = args.baseloss)
    saved_stats = torch.load(name_CP+'acc', map_location=device)
    bbox.model.load_state_dict(saved_stats['model_state'])
    epoch = saved_stats['stats']['epoch'][-1]
    epoch_left_to_train = args.num_epochs-epoch
    stats_bbox = bbox.fit(train_loader = train_loader, 
                    estop_loader = hout_loader,
                    Z_train = Z_train,
                    batch_size = args.batch_size, 
                    num_epochs = epoch_left_to_train, 
                    optimizer = args.optimizer,
                    lr = args.lr,
                    mu = args.mu, 
                    mu_size = args.mu_size, 
                    alpha = args.train_alpha, 
                    save_model = args.save_model, 
                    save_checkpoint_period = args.save_checkpoint_period,
                    name = file_final, 
                    early_stopping = args.early_stopping, 
                    name_CP = name_CP)

# otherwise training from the start
else:
    bbox = black_boxes_CNN.BlackBox(num_features, 
                                    num_classes, 
                                    baseloss = args.baseloss)


    stats_bbox = bbox.fit(train_loader = train_loader, 
                            estop_loader = hout_loader,
                            Z_train = Z_train,
                            batch_size = args.batch_size, 
                            num_epochs = args.num_epochs, 
                            optimizer = args.optimizer,
                            lr = args.lr,
                            mu = args.mu, 
                            mu_size = args.mu_size, 
                            alpha = args.train_alpha, 
                            save_model = args.save_model, 
                            save_checkpoint_period = args.save_checkpoint_period,
                            name = file_final, 
                            early_stopping = args.early_stopping, 
                            name_CP = name_CP)





#----------------------------------------- Evaluations -----------------------------------------


seed = 666
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#----------------------------------------- Preparing test dataset ----------------------------------------
augmentation = [
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor((0.4914, 0.4822, 0.4465)), torch.Tensor((0.2023, 0.1994, 0.2010)))
        # we do not do randomearsing here, but in the blurring_images functions below, in order to keep the corruption indicator
    ]

transform = transforms.Compose(augmentation)
home_dir = os.path.expanduser('~')
data_dir = home_dir + "/data"

# download the dataset and blur it with indicator F_ind (1 for corrupt 0 for noncorrupt)
test_dataset_to_blur = torchvision.datasets.CIFAR10(root=data_dir, train = False, download=False, transform=transform)
test_dataset_blurred, F_ind = Splitconformal_CNN.blurring_images(test_dataset_to_blur, scale=(args.scale, args.scale), ratio=(0.3, 3.3), batch_size = 1, corrupt_percent = args.cal_perc, plot = False)

num_features = 3
num_classes = 10

#----------------------------------------- Loading Model ----------------------------------------
path_eval = "featureaug={}_batchsize={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_corruptperc={}_seed={}_baseloss={}_ntrsamples={}_model={}".format(args.feature_augmentation, args.batch_size, 
args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method,args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model)

# -------- full model -----------
bbox = black_boxes_CNN.BlackBox(num_features, num_classes)
mod_load = torch.load('./UncertaintyAware/saved_models/trained_with_blurred_new2/final_'+path_eval, map_location=device)
bbox.model.load_state_dict(mod_load['model_state'])

# -------- early stopping models ---------
# early stopping loss
bbox_es_loss = black_boxes_CNN.BlackBox(num_features, num_classes)
mod_load = torch.load('./UncertaintyAware/saved_models/trained_with_blurred_new2/checkpoint_'+path_eval+'loss', map_location=device)
bbox_es_loss.model.load_state_dict(mod_load['model_state'])

# early stopping acc
bbox_es_acc = black_boxes_CNN.BlackBox(num_features, num_classes)
mod_load = torch.load('./UncertaintyAware/saved_models/trained_with_blurred_new2/checkpoint_'+path_eval+'acc', map_location=device)
bbox_es_acc.model.load_state_dict(mod_load['model_state'])



#----------------------------------------- Conformal evaluation ----------------------------------------
black_boxes = [bbox, bbox_es_loss, bbox_es_acc]
black_boxes_names = ["bbox","bbox_es_loss","bbox_es_acc"]

print("begin evaluation.")

results_full = pd.DataFrame()
if args.return_scores or args.predict_prob_only:
  results_full = {}
for e in tqdm(range(args.bs_times)):

  # evaluating on subsets randomly drawn from full test data
  if args.bs_times >1:
    random_indices_t = torch.randperm(len(test_dataset_blurred))[:args.n_cal_samples + args.n_test_samples]
    random_indices_cal = random_indices_t[:args.n_cal_samples]
    random_indices_test = random_indices_t[args.n_cal_samples:args.n_cal_samples + args.n_test_samples]
    cal_sample = torch.utils.data.Subset(test_dataset_blurred, random_indices_cal)
    test_sample = torch.utils.data.Subset(test_dataset_blurred, random_indices_test)

    test_loader = torch.utils.data.DataLoader(test_sample, batch_size=1,shuffle=False, num_workers=2)
    cal_loader = torch.utils.data.DataLoader(cal_sample, batch_size=1,shuffle=False, num_workers=2)

    F_cal = F_ind[random_indices_cal]
    F_test = F_ind[random_indices_test]

  # evaluating on full test data 
  elif args.bs_times ==1: 
    evens = list(range(0, len(test_dataset_blurred), 2))
    odds = list(range(1, len(test_dataset_blurred), 2))
    cal_sample = torch.utils.data.Subset(test_dataset_blurred, evens)
    test_sample = torch.utils.data.Subset(test_dataset_blurred, odds)
    cal_loader = torch.utils.data.DataLoader(cal_sample, batch_size=1,shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_sample, batch_size=1,shuffle=False, num_workers=2)
    F_cal = [F_ind[i] for i in evens]
    F_test = [F_ind[i] for i in odds]

  for k in range(len(black_boxes)):
    if args.return_scores:
      df = evaluations_CNN.evaluation_conformal_full_realdata(test_loader, cal_loader, black_boxes[k], args.cal_alpha, \
                                              F_test, condition = args.evaluation_condition , plot_result = False, return_scores_only = True)

      if black_boxes_names[k] not in results_full:
        results_full[black_boxes_names[k]] = {
          'conformity_scores_1.0': df['scores_1'],
          'conformity_scores_0.0': df['scores_0'],
          'conformity_scores_All': df['scores_All']
        }

    if args.predict_prob_only:
      df = evaluations_CNN.evaluation_conformal_full_realdata(test_loader, cal_loader, black_boxes[k], args.cal_alpha, \
                                              F_test, condition = 'by_flag', plot_result = False, return_scores_only = False, predict_prob_only = True)
      if black_boxes_names[k] not in results_full:
        results_full[black_boxes_names[k]] = {
          'pred_prob': df['pred_prob'],
          'y_true': df['y_true']
        }
      
    else: 
      # performing calibration and evaluating 1) conditional coverage 2) size of prediction set 3) test accuracy 4) size conditional on coverage
      df = evaluations_CNN.evaluation_conformal_full_realdata(test_loader, cal_loader, black_boxes[k], args.cal_alpha, \
                                              F_test, condition = args.evaluation_condition, plot_result = False, no_calib = args.no_calib)

      # appending model name and pre-specified calibration alpha
      df['conditional_stats']['Model'] = black_boxes_names[k]
      df['conditional_stats']['calalpha_level'] = args.cal_alpha
      
      results_full = pd.concat([results_full, df['conditional_stats']]) 
      print('finished the ',e+1,' evaluation ', ' for ', black_boxes_names[k])

if args.return_scores:
  path_eval_save = "feataug={}_bs={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_trperc={}_seed={}_baseloss={}_ntrsamp={}_model={}_calalpha={}_calperc={}_confs".format(args.feature_augmentation, args.batch_size, 
  args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model, args.cal_alpha, args.cal_perc)

elif args.predict_prob_only:
  path_eval_save = "feataug={}_bs={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_trperc={}_seed={}_baseloss={}_ntrsamp={}_model={}_calalpha={}_calperc={}_probs".format(args.feature_augmentation, args.batch_size, 
  args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model, args.cal_alpha, args.cal_perc)

elif args.evaluation_condition == 'by_label_flag':
  path_eval_save = "feataug={}_bs={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_trperc={}_seed={}_baseloss={}_ntrsamp={}_model={}_calalpha={}_calperc={}_lf".format(args.feature_augmentation, args.batch_size, 
  args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model, args.cal_alpha, args.cal_perc)

elif args.no_calib:
  path_eval_save = "feataug={}_bs={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_trperc={}_seed={}_baseloss={}_ntrsamp={}_model={}_calalpha={}_calperc={}_nocalib".format(args.feature_augmentation, args.batch_size, 
  args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model, args.cal_alpha, args.cal_perc)
  
else: 
  path_eval_save = "feataug={}_bs={}_numepochs={}_lr={}_optim={}_mu={}_musize={}_tralpha={}_es={}_method={}_scales={}_trperc={}_seed={}_baseloss={}_ntrsamp={}_model={}_calalpha={}_calperc={}".format(args.feature_augmentation, args.batch_size, 
  args.num_epochs, args.lr, args.optimizer, args.mu, args.mu_size, args.train_alpha, args.early_stopping, args.method, args.scale, args.train_perc, args.seed, args.baseloss, args.n_tr_samples, args.model, args.cal_alpha, args.cal_perc)

if args.save_pickle:
  print(path_eval_save)
  with open('./UncertaintyAware/saved_models/trained_with_blurred_new2/sc_results_'+path_eval_save+'.pickle', 'wb') as f:
      pickle.dump(results_full, f)

else:
  print(path_eval_save)
  results_full.to_csv('./UncertaintyAware/saved_models/trained_with_blurred_new2/sc_results_'+path_eval_save+'.csv', index = False)
