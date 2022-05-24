import os
import sys
import torch
import random
import pandas as pd
import numpy as np

sys.path.append('../../conformal_learning/')
from bbox import BlackBox
from split_conf import SplitConformal, evaluate_predictions, ProbAccum
from auxiliary import eval_predictions, KL, cvm

sys.path.append('../../conformal_learning/third_party/')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(seed):

    #seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    gamma = 1
    mu = 0.2
    batch_size_new = 2500
    prc_score = 0.3
    
    files_dir = './saved_models_credit' 
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    df = pd.read_excel(r'default_of_credit_card_clients.xlsx')
    df.head()
    Y = np.asarray(df.iloc[1:,24]).astype(int)
    X_raw = np.asarray(df.iloc[1:,1:24]).astype(np.float32)
    X = X_raw
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    X_hout = X_train[int(0.8*X_train.shape[0]):,:]
    Y_hout = Y_train[int(0.8*X_train.shape[0]):]
    
    X_train = X_train[:int(0.8*X_train.shape[0]),:]
    Y_train = Y_train[:int(0.8*Y_train.shape[0])]
    
    X_tr_score = X_train[int((1-prc_score)*X_train.shape[0]):,:]
    Y_tr_score = Y_train[int((1-prc_score)*Y_train.shape[0]):]
    
    X_train = X_train[:int((1-prc_score)*X_train.shape[0]),:]
    Y_train = Y_train[:int((1-prc_score)*Y_train.shape[0])]
    
    num_classes = len(np.unique(Y))
    num_features=X_train.shape[1]

    out_prefix =  'seed_%a' %seed
    
    X_augmented = np.concatenate((X_train, X_tr_score),0)
    Y_augmented = np.concatenate((Y_train, Y_tr_score),0)
    
    # Train model with CE loss
                
    file_final = files_dir+'/'+'saved_model_orig_' + out_prefix
      
    if os.path.isfile(file_final):
        print('Loading model instead of training')
        box_ce = BlackBox(num_features, num_classes)
        saved_stats = torch.load(file_final, map_location=device)
        box_ce.model.load_state_dict(saved_stats['model_state'])
        stats_ce = saved_stats['stats']

    else:
        box_ce = BlackBox(num_features, num_classes)
        stats_ce = box_ce.fit(X_augmented, Y_augmented, X_hout=X_hout, Y_hout=Y_hout, 
                              batch_size=500, num_epochs=3000, lr=0.0001, mu=0,
                              save_model=True, save_checkpoint_period = 1, optimizer='Adam',
                              name=file_final, early_stopping=True, 
                              name_CP=files_dir+'/'+'checkpoint_orig_'+out_prefix, verbose=False)
                          
    # For early stopping loss
    box_orig_ho_loss = BlackBox(num_features, num_classes)
    saved_stats_ho_loss = torch.load(files_dir+'/'+'checkpoint_orig_'+out_prefix+'loss', map_location=device)
    box_orig_ho_loss.model.load_state_dict(saved_stats_ho_loss['model_state'])

    # For early stopping acc
    box_orig_ho_acc = BlackBox(num_features, num_classes)
    saved_stats_ho_acc = torch.load(files_dir+'/'+'checkpoint_orig_'+out_prefix+'acc', map_location=device)
    box_orig_ho_acc.model.load_state_dict(saved_stats_ho_acc['model_state'])
    
    
    # Train model with conformal loss


    file_final = files_dir+'/'+'saved_model_new_' + out_prefix
    name_CP = files_dir+'/'+'checkpoint_new_'+out_prefix

    Z_train = np.zeros(len(Y_train))
    Z_tr_score = np.ones(len(Y_tr_score))

    Z_augmented = np.concatenate((Z_train, Z_tr_score),0)


    if os.path.isfile(file_final):
        print('Loading model instead of training')
        box_new = BlackBox(num_features, num_classes)
        saved_stats = torch.load(file_final, map_location=device)
        box_new.model.load_state_dict(saved_stats['model_state'])
        stats_new = saved_stats['stats']
                    
                    
    else:
        box_new = BlackBox(num_features, num_classes)
        stats_new = box_new.fit(X_augmented, Y_augmented, X_hout=X_hout, Y_hout=Y_hout,
                                    Z_train=Z_augmented, cond_label=True,
                                    batch_size=batch_size_new, num_epochs=6000, lr=0.0001,
                                    mu=mu, mu_size=0, alpha=0.2, 
                                    save_model=True, save_checkpoint_period = 1, optimizer='Adam',
                                    name=file_final, early_stopping=True, 
                                    name_CP=name_CP, verbose=False)

    # For early stopping loss
    box_new_ho_loss = BlackBox(num_features, num_classes)
    saved_stats_new_loss = torch.load(files_dir+'/'+'checkpoint_new_'+out_prefix+'loss')
    box_new_ho_loss.model.load_state_dict(saved_stats_new_loss['model_state'])

    # For early stopping acc
    box_new_ho_acc = BlackBox(num_features, num_classes)
    saved_stats_new_acc = torch.load(files_dir+'/'+'checkpoint_new_'+out_prefix+'acc')
    box_new_ho_acc.model.load_state_dict(saved_stats_new_acc['model_state'])


   # Train the hybrid model
#    
#    
    file_final = files_dir+'/'+'saved_model_hybrid_' + out_prefix


    if os.path.isfile(file_final):
        print('Loading model instead of training')
        box_hybrid = BlackBox(num_features, num_classes)
        saved_stats = torch.load(file_final, map_location=device)
        box_hybrid.model.load_state_dict(saved_stats['model_state'])
        stats_hybrid = saved_stats['stats']
    else:
        box_hybrid = BlackBox(num_features, num_classes)
        stats_hybrid = box_hybrid.fit(X_augmented, Y_augmented, X_hout=X_hout, Y_hout=Y_hout,
                                    Z_train=Z_augmented, cond_label=True,
                                    batch_size=2500, num_epochs=4000, lr=0.0001,
                                    mu=0, mu_size=0.1, alpha=0.2, 
                                    save_model=True, save_checkpoint_period = 1, optimizer='Adam',
                                    name=file_final, early_stopping=True, 
                                    name_CP=files_dir+'/'+'checkpoint_hybrid_'+out_prefix, verbose=False)

    # For early stopping loss
    box_hybrid_ho_loss = BlackBox(num_features, num_classes)
    saved_stats_hybrid_loss = torch.load(files_dir+'/'+'checkpoint_hybrid_'+out_prefix+'loss')
    box_hybrid_ho_loss.model.load_state_dict(saved_stats_hybrid_loss['model_state'])

    # For early stopping acc
    box_hybrid_ho_acc = BlackBox(num_features, num_classes)
    saved_stats_hybrid_acc = torch.load(files_dir+'/'+'checkpoint_hybrid_'+out_prefix+'acc')
    box_hybrid_ho_acc.model.load_state_dict(saved_stats_hybrid_acc['model_state'])
#        
#        
#   
    # Train model with Focal loss 
    
    file_final = files_dir+'/'+'saved_model_FocalLoss_' + out_prefix
    
    if os.path.isfile(file_final):
        print('Loading model instead of training')
        box_fc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma)
        saved_stats = torch.load(file_final, map_location=device)
        box_fc.model.load_state_dict(saved_stats['model_state'])
        stats_fc = saved_stats['stats']

    else:
        box_fc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma)
        stats_fc = box_fc.fit(X_augmented, Y_augmented, X_hout=X_hout, Y_hout=Y_hout, 
                              batch_size=500, num_epochs=3000, lr=0.0001, mu=0,
                              save_model=True, save_checkpoint_period = 1, optimizer='Adam',
                              name=file_final, early_stopping=True, 
                              name_CP=files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix, verbose=False)
                              
                              
    # For early stopping loss
    box_focal_ho_loss = BlackBox(num_features, num_classes, base_loss='Focal', gamma=1)
    saved_stats_ho_loss = torch.load(files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix+'loss', map_location=device)
    box_focal_ho_loss.model.load_state_dict(saved_stats_ho_loss['model_state'])

    # For early stopping acc
    box_focal_ho_acc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=1)
    saved_stats_ho_acc = torch.load(files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix+'acc', map_location=device)
    box_focal_ho_acc.model.load_state_dict(saved_stats_ho_acc['model_state'])


    black_boxes = [box_ce, box_orig_ho_loss, 
                   box_new, box_new_ho_loss,
                   box_fc, box_focal_ho_loss,
                   box_hybrid, box_hybrid_ho_loss]
    black_boxes_names = ["Cross-entropy", "Cross-entropy ES loss",
                         "Conformal", "Conformal ES loss",
                         "Focal", "Focal ES loss",
                         "Hybrid", "Hybrid ES loss"]                    
  
    X_calib, X_test_tmp, Y_calib, Y_test_tmp = train_test_split(X_test, Y_test, test_size=0.5, random_state=seed)
    alpha = 0.2
    sc_methods = []
    for i in range(len(black_boxes)):
        print("{:s}:".format(black_boxes_names[i]))
        sc_method = SplitConformal()
        sc_method.calibrate(X_calib, Y_calib, alpha, bbox=black_boxes[i])
        sc_methods.append(sc_method)
        

    easy_idx = np.asarray(np.where((Y_test_tmp==0)))[0,:]
    hard_idx = np.asarray(np.where((Y_test_tmp==1)))[0,:]


    results = pd.DataFrame()

    for k in range(len(black_boxes)):
      
        
      p_hat_test = black_boxes[k].predict_proba(X_test_tmp)
      grey_box = ProbAccum(p_hat_test)
      epsilon_test = np.random.uniform(size=(len(Y_test_tmp,)))
      scores_test = grey_box.calibrate_scores(Y_test_tmp, epsilon=epsilon_test)

      p_hat_test_hard = black_boxes[k].predict_proba(X_test_tmp[hard_idx,:])
      grey_box = ProbAccum(p_hat_test_hard)
      epsilon_test_hard = np.random.uniform(size=(len(Y_test_tmp[hard_idx],)))
      scores_test_hard = grey_box.calibrate_scores(Y_test_tmp[hard_idx], epsilon=epsilon_test_hard)
      
      p_hat_test_easy = black_boxes[k].predict_proba(X_test_tmp[easy_idx,:])
      grey_box = ProbAccum(p_hat_test_easy)
      epsilon_test_easy = np.random.uniform(size=(len(Y_test_tmp[easy_idx],)))
      scores_test_easy = grey_box.calibrate_scores(Y_test_tmp[easy_idx], epsilon=epsilon_test_easy)
    
      sets = sc_methods[k].predict(X_test_tmp)
      res = evaluate_predictions(sets, X_test_tmp, Y_test_tmp, hard_idx, conditional=True)
      res['Model'] = black_boxes_names[k]
      res['Error'] = eval_predictions(X_test_tmp, Y_test_tmp, black_boxes[k], data="test", plot=False, printing=False)
      res['seed'] = seed
      res['cvm'] = cvm(scores_test)
      res['cvm-hard'] = cvm(scores_test_hard)
      res['cvm-easy'] = cvm(scores_test_easy)
    
      results = pd.concat([results, res])
        
    results = results.reset_index()
    return results
    
    

  
  
  
if __name__ == '__main__':
    
  # Parameters
  seed = int(sys.argv[1])
  # Output directory and filename
  out_dir = "./results_credit"
  out_file = out_dir + "_seed_" + str(seed) + ".csv"

  # Run experiment
  result = experiment(seed)

  # Write the result to output file
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  result.to_csv(out_dir + '/' + out_file, index=False, float_format="%.4f")
  print("Updated summary of results on\n {}".format(out_file))
  sys.stdout.flush()
