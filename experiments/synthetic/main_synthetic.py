import os
import sys
import torch
import random
import pandas as pd
import numpy as np

sys.path.append('../../conformal_learning/')
from bbox import BlackBox
from data_gen import Model_Ex1, Oracle, difficulty_oracle
from split_conf import SplitConformal, evaluate_predictions, ProbAccum
from auxiliary import eval_predictions, KL, cvm

sys.path.append('../../conformal_learning/third_party/')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def experiment(delta_2, seed):

    #seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Set the fixed parameters
    
    p = 100                                                # Number of features
    K = 6                                                  # Number of possible labels
    delta_1 = 0
    #delta_2 = 0.2
    gamma = 1
    a = 1
    batch_size_CE = 200
    lr_CE = 0.01
    batch_size_new = 750
    lr_new = 0.001
    mu = 0.2
    alpha = 0.1
    
    changing_param = delta_2
    changing_param_str = 'delta_2'
    
    files_dir = './saved_models_' + changing_param_str
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    num_classes = K

    data_model = Model_Ex1(K,p,delta_1, delta_2, a)           # Data generating model
    
    n_train = 2000                                         # Number of data samples
    X_train = data_model.sample_X(n_train)                 # Generate the data features
    Y_train = data_model.sample_Y(X_train)                 # Generate the data labels conditional on the features

    n_tr_score = int(n_train*0.2)                          # Number of data samples for training the new loss
    X_tr_score = data_model.sample_X(n_tr_score)           # Generate the data features
    Y_tr_score = data_model.sample_Y(X_tr_score)           # Generate the data labels conditional on the features

    n_hout = 2000                                          # Number of hold out samples
    X_hout = data_model.sample_X(n_hout)
    Y_hout = data_model.sample_Y(X_hout)

    n_test = 2000                                          # Number of test samples
    X_test = data_model.sample_X(n_test, test=True)                   # Generate independent test data
    Y_test = data_model.sample_Y(X_test) 

    num_features = X_train.shape[1]

    out_prefix = changing_param_str  + '_%a' %changing_param + '_seed_%a' %seed
    
    X_augmented = np.concatenate((X_train, X_tr_score),0)
    Y_augmented = np.concatenate((Y_train, Y_tr_score),0)
    
    oracle = Oracle(data_model)
        
        
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
                              batch_size=batch_size_CE, num_epochs=3000, lr=lr_CE, mu=0, optimizer='SGD',
                              save_model=True, save_checkpoint_period = 1,
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
                                    Z_train=Z_augmented,
                                    batch_size=batch_size_new, num_epochs=4000, lr=lr_new,
                                    mu=mu, mu_size=0, alpha=0.1,  optimizer='Adam',
                                    save_model=True, save_checkpoint_period = 1,
                                    name=file_final, early_stopping=True, 
                                    name_CP=files_dir+'/'+'checkpoint_new_cs_'+out_prefix, verbose=False)

    # For early stopping loss
    box_new_ho_loss = BlackBox(num_features, num_classes)
    saved_stats_new_loss = torch.load(files_dir+'/'+'checkpoint_new_cs_'+out_prefix+'loss')
    box_new_ho_loss.model.load_state_dict(saved_stats_new_loss['model_state'])

    # For early stopping acc
    box_new_ho_acc = BlackBox(num_features, num_classes)
    saved_stats_new_acc = torch.load(files_dir+'/'+'checkpoint_new_cs_'+out_prefix+'acc')
    box_new_ho_acc.model.load_state_dict(saved_stats_new_acc['model_state'])


    # Train hybrid model


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
                                    Z_train=Z_augmented,
                                    batch_size=750, num_epochs=4000, lr=0.01,
                                    mu=0, mu_size=0.2, alpha=0.1,  optimizer='SGD',
                                    save_model=True, save_checkpoint_period = 1,
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
#    Train model with focal loss

    file_final = files_dir+'/'+'saved_model_FocalLoss_' + out_prefix
    
    if os.path.isfile(file_final):
        print('Loading model instead of training')
        box_fc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma)
        saved_stats = torch.load(file_final, map_location=device)
        box_fc.model.load_state_dict(saved_stats['model_state'])
        stats_fc = saved_stats['stats']

    else:
        box_fc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma) # gamma 0 same as C.E.
        stats_fc = box_fc.fit(X_augmented, Y_augmented, X_hout=X_hout, Y_hout=Y_hout, 
                              batch_size=200, mu=0, num_epochs=3000, lr=0.01, optimizer='SGD',
                              save_model=True, save_checkpoint_period = 1,
                              name=file_final, early_stopping=True, name_CP=files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix, verbose=False)
                              
                              
                              
    # For early stopping loss
    box_focal_ho_loss = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma)
    saved_stats_ho_loss = torch.load(files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix+'loss', map_location=device)
    box_focal_ho_loss.model.load_state_dict(saved_stats_ho_loss['model_state'])

    # For early stopping acc
    box_focal_ho_acc = BlackBox(num_features, num_classes, base_loss='Focal', gamma=gamma)
    saved_stats_ho_acc = torch.load(files_dir+'/'+'checkpoint_FocalLoss_'+out_prefix+'acc', map_location=device)
    box_focal_ho_acc.model.load_state_dict(saved_stats_ho_acc['model_state'])


    black_boxes = [oracle, box_ce, box_orig_ho_loss, box_orig_ho_acc, 
                    box_new, box_new_ho_loss, box_new_ho_acc, box_ce, box_new,
                    box_hybrid, box_hybrid_ho_loss, box_hybrid_ho_acc,
                    box_fc, box_focal_ho_loss, box_focal_ho_acc]
    black_boxes_names = ["Oracle", 
                           "Cross-entropy", "Cross-entropy ES loss", "Cross-entropy ES acc",
                           "Conformal", "Conformal ES loss", "Conformal ES acc", "Cross-entropy no calib", "Conformal no calib",
                           "Hybrid", "Hybrid ES loss", "Hybrid ES acc",
                           "Focal", "Focal ES loss", "Focal ES acc"]
                             
  


    n_calib = 10000                                        # Number of calibration samples
    X_calib = data_model.sample_X(n_calib)             # Generate independent calibration data
    Y_calib = data_model.sample_Y(X_calib)

    sc_methods = []
    for i in range(len(black_boxes)):
      print("{:s}:".format(black_boxes_names[i]))
      sc_method = SplitConformal()
      if black_boxes_names[i] == "Cross-entropy no calib":
        sc_method.calibrate(X_calib, Y_calib, alpha, bbox=black_boxes[i], no_calib=True)
      elif black_boxes_names[i] == "Conformal no calib":
        sc_method.calibrate(X_calib, Y_calib, alpha, bbox=black_boxes[i], no_calib=True)
      else:
        sc_method.calibrate(X_calib, Y_calib, alpha, bbox=black_boxes[i])
      sc_methods.append(sc_method)

    results = pd.DataFrame()


    sc_method_oracle = SplitConformal(bbox=oracle)

    n_test = 2000                                          # Number of test samples
    X_test = data_model.sample_X(n_test, test=True)                   # Generate independent test data
    Y_test = data_model.sample_Y(X_test)

    
    prob_true = data_model.compute_prob(X_test)

    S_oracle = sc_method_oracle.predict(X_test, alpha=alpha)
    size_oracle = np.array([len(S) for S in S_oracle])
    easy_idx, hard_idx = difficulty_oracle(S_oracle)

    for k in range(len(black_boxes)):
      
        
      p_hat_test = black_boxes[k].predict_proba(X_test)
      grey_box = ProbAccum(p_hat_test)
      epsilon_test = np.random.uniform(size=(len(Y_test,)))
      scores_test = grey_box.calibrate_scores(Y_test, epsilon=epsilon_test)
    
    
      sets = sc_methods[k].predict(X_test)
      res = evaluate_predictions(sets, X_test, Y_test, hard_idx, conditional=True)  
      res['Model'] = black_boxes_names[k]
      res['Error'] = eval_predictions(X_test, Y_test, black_boxes[k], data="test", plot=False, printing=False)
      res['KL-hard'] = np.mean(KL(prob_true[hard_idx], black_boxes[k].predict_proba(X_test[hard_idx])))
      res[changing_param_str] = changing_param
      res['seed'] = seed
      res['cvm'] = cvm(scores_test)
    
      results = pd.concat([results, res])
        
    results = results.reset_index()
    return results
    
    

  
  
  
if __name__ == '__main__':
    
  # Parameters
  delta_2 = float(sys.argv[1])    # change to float for some parameters
  seed = int(sys.argv[2])
  # Output directory and filename
  out_dir = "./results_delta_2"
  out_file = out_dir + '_' + str(delta_2)+ "_seed_" + str(seed) + ".csv"
  # Run experiment
  result = experiment(delta_2, seed)

  # Write the result to output file
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  result.to_csv(out_dir + '/' + out_file, index=False, float_format="%.4f")
  print("Updated summary of results on\n {}".format(out_file))
  sys.stdout.flush()
