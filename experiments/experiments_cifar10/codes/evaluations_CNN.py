import seaborn as sns
import matplotlib.pyplot as plt

import random
import torch
import pandas as pd
import numpy as np
from codes import Splitconformal_CNN

seed = 666
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def evaluation_predictions_realdata(input, condition = 'by_label'):
  '''
  Model evaluations on different conditional criterion
  -------------------------parameters-----------------------------
  condition: 'by_label': evaluating conditional results conditioning on true label
             'by_label': evaluating conditional results conditioning on size of prediction set
             'by_flag': evaluating conditional results conditioning on pre-specified indicators (e.g. corruption flag)
  -------------------------output-----------------------------
  out: a dataframe including informations of conditional coverage, prediction set size, test accuracy etc.
  '''

  if condition == 'by_label':
    y, S = np.array(input['y_true']), np.array(input['prediction_set'])
    label = np.unique(y)

    if len(label) == 1:
      label = label[0]
    else:
      label = "All"

    # count
    count = len(y)
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # size of prediction set
    size = [len(S[i]) for i in range(len(y))]
    # average size
    size_mean = np.mean([len(S[i]) for i in range(len(y))])
    # median size
    size_median = np.median([len(S[i]) for i in range(len(y))])
    # size conditional on coverage
    # which prediction sets cover the true label
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    # average size of sets that cover true y
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # coverage indicator
    cover_indicator = [y[i] in S[i] for i in range(len(y))]
    # correlation between size of prediction set and coverage indicator
    corr_size_cover = np.corrcoef(cover_indicator, size)
    single_corr = corr_size_cover[0][1]
    # test accuracy
    test_acc = np.mean(y == input['y_pred'])
    # Combine results
    out = pd.DataFrame({'Label': [label],
                        'Count': [count],
                        'Coverage': [coverage],
                        'Size (mean)': [size_mean], 
                        'Size (median)': [size_median], 
                        'Size cover': [size_cover],
                        # 'Corr size-cover': [corr_size_cover],
                        'Single Corr':[single_corr],
                        'Test Acc':[test_acc]})
  
  elif condition == 'by_size':
    y, S, size = np.array(input['y_true']), np.array(input['prediction_set']), np.array(input['prediction_size'])

    set_size = np.unique(size)

    if len(set_size) == 1:
      set_size = set_size[0]
    else:
      set_size = "All"

    # count
    count = len(y)
    # coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    out = pd.DataFrame({'Size': [set_size],
                        'Count': [count],
                        'Coverage': [coverage]
                        })
    
  elif condition == 'by_flag':
    y, S, f = np.array(input['y_true']), np.array(input['prediction_set']), np.array(input['flag'])
    flag = np.unique(f)

    if len(flag) == 1:
      flag = flag[0]
    else:
      flag = "All"

    # count
    count = len(y)
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # size of prediction set
    size = [len(S[i]) for i in range(len(y))]
    # average size
    size_mean = np.mean([len(S[i]) for i in range(len(y))])
    # median size
    size_median = np.median([len(S[i]) for i in range(len(y))])
    # size conditional on coverage
    # which prediction sets cover the true label
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    # average size of sets that cover true y
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # coverage indicator
    cover_indicator = [y[i] in S[i] for i in range(len(y))]
    # correlation between size of prediction set and coverage indicator
    corr_size_cover = np.corrcoef(cover_indicator, size)
    single_corr = corr_size_cover[0][1]
    # test accuracy
    test_acc = np.mean(y == input['y_pred'])
    # Combine results
    out = pd.DataFrame({'Flag': [flag],
                        'Count': [count],
                        'Coverage': [coverage],
                        'Size (mean)': [size_mean], 
                        'Size (median)': [size_median], 
                        'Size cover': [size_cover],
                        # 'Corr size-cover': [corr_size_cover],
                        'Single Corr':[single_corr],
                        'Test Acc':[test_acc]})

  elif condition == 'by_label_flag': #
    y, S, f = np.array(input['y_true']), np.array(input['prediction_set']), np.array(input['flag'])
    flag = np.unique(f)
    if len(flag) == 1:
      flag = flag[0]
    else:
      flag = "All"
    print('flag is {}'.format(flag))

    label = np.unique(y)
    if len(label) == 1:
      label = label[0]
    else:
      label = "All"
    print('label is {}'.format(label))

    # count
    count = len(y)
    # Marginal coverage
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    # size of prediction set
    size = [len(S[i]) for i in range(len(y))]
    # average size
    size_mean = np.mean([len(S[i]) for i in range(len(y))])
    # median size
    size_median = np.median([len(S[i]) for i in range(len(y))])
    # size conditional on coverage
    # which prediction sets cover the true label
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    # average size of sets that cover true y
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # coverage indicator
    cover_indicator = [y[i] in S[i] for i in range(len(y))]
    # correlation between size of prediction set and coverage indicator
    corr_size_cover = np.corrcoef(cover_indicator, size)
    single_corr = corr_size_cover[0][1]
    # test accuracy
    test_acc = np.mean(y == input['y_pred'])
    # Combine results
    out = pd.DataFrame({'Flag': [flag],
                        'Label': [label],
                        'Count': [count],
                        'Coverage': [coverage],
                        'Size (mean)': [size_mean], 
                        'Size (median)': [size_median], 
                        'Size cover': [size_cover],
                        # 'Corr size-cover': [corr_size_cover],
                        'Single Corr':[single_corr],
                        'Test Acc':[test_acc]})

    
    
  return out



    

def pred_prob_on_true(selected_label, y_true, pred_prob, plot = False):
  """
  Computing the estimated probabilities for the selected label and rank of estimated probabilities
  -------------------------parameters-----------------------------
  selected_label: selected label that we want to plot
  y_true: true label 
  pred_prob: predicted probability
  -------------------------output-----------------------------
  visual_res: predicted probabilities and their ranks
  fig: visualization of the predicted probabilities and ranks
  """
  # find index of the selected label
  ind = [i for i,val in enumerate(y_true) if val==selected_label]
  # estimated probabilties for the selected true label
  prob_dist = [item[selected_label] for item in pred_prob[ind]]

  # ranks of estimated probabiltiies of the selected true label
  order = np.argsort(-pred_prob, axis=1)
  ranks = np.empty_like(order)
  for i in range(pred_prob.shape[0]):
    ranks[i, order[i]] = np.arange(len(order[i]))
  rank_dist = [item[selected_label] for item in ranks[ind]]

  visual_res = pd.DataFrame({'pred_proba': prob_dist, 
              'pred_rank': rank_dist})

  if plot == True:
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    plt.suptitle('Predicted probabilities and ranks for label {}'.format(selected_label), y=1.05)
    # sns.distplot(prob_dist, kde = False, ax = ax1)
    sns.histplot(data=visual_res, x="pred_proba", bins = 100, ax = ax1)
    sns.histplot(data=visual_res, x="pred_rank", bins = len(set(y_true)), ax = ax2)
    ax1.set_xlabel('Predicted Probabilities')
    ax2.set_xlabel('Predicted Ranks')
  return visual_res, fig



def evaluation_conformal_full_realdata(test_loader, cal_loader, model, alpha, F_test = None, condition = 'by_label', plot_result = False, no_calib=False, return_scores_only = False, predict_prob_only = False):
  """ 
  Calibrate the model and conduct evaluations
  -------------------------parameters-----------------------------
  cal_loader: calibration data - for calibrating the models
  test_loader: test data - for evaluating the (un)conditional results after model calibration (unless no_calib is True)
  model: trained models to evaluate on
  alpha: calibration (marginal coverage) level
  F_test: indicator of corruption
  condition: criterions for conditional evaluation
  plot_results: return a plot or not
  no_calib: if yes, return evaluation results without calibrating the models using calib_loader
  return_scores_only: if yes, only return conformity scores on test dataset
  predict_prob_only: if yes, only return predicted probabilities on test dataset
  -------------------------output-----------------------------
  out: a dataframe including informations of conditional coverage, prediction set size, test accuracy etc.
  """

  if return_scores_only:
    print('returning conformity scores...')
    easy_index = [i for i, j in enumerate(F_test) if float(j) == 0.0]
    hard_index = [i for i, j in enumerate(F_test) if float(j) == 1.0]
    method_test = Splitconformal_CNN.SplitConformal()
    method_test.calibrate(test_loader, alpha = alpha, bbox = model, return_scores = return_scores_only)
    print(method_test.scores[easy_index])
    return(dict({'scores_0': method_test.scores[easy_index],
                'scores_1': method_test.scores[hard_index],
                'scores_All': method_test.scores}))

  # get predicted label, predicted probabilities, and true label from bbox
  y_pred = model.predict(test_loader)
  y_pred_prob, y_true = model.predict_proba(test_loader, return_y_true = True)

  if predict_prob_only:
    print('returning predicted probabilities...')
    return(dict({'pred_prob': y_pred_prob, 
                'y_true':y_true}))

  # calibrate the bbox
  method = Splitconformal_CNN.SplitConformal()
  method.calibrate(cal_loader, alpha = alpha, bbox = model, no_calib=no_calib)
  # compute prediction set
  y_pred_S = method.predict(test_loader) 

  # computeing prediction size
  prediction_size = [len(x) for x in y_pred_S]

  if F_test is not None:
    # initialize dataframe
    df = pd.DataFrame([y_pred_S, y_pred, y_pred_prob, y_true, prediction_size, F_test]).T.\
                      rename(columns = {0:'prediction_set',
                                        1:'y_pred',
                                        2:'y_pred_prob',
                                        3:'y_true',
                                        4:'prediction_size',
                                        5:'flag'})
                      
  else: 
    df = pd.DataFrame([y_pred_S, y_pred, y_pred_prob, y_true, prediction_size]).T.\
                      rename(columns = {0:'prediction_set',
                                        1:'y_pred',
                                        2:'y_pred_prob',
                                        3:'y_true',
                                        4:'prediction_size'})

  # evaluate models based on specified conditions
  if condition == 'by_label':
    df_bylabel_conditional = pd.DataFrame(df.groupby(['y_true'])[['y_true','prediction_set','y_pred']].\
                              apply(evaluation_predictions_realdata).reset_index().drop(['y_true','level_1'], axis=1))
    df_bylabel_marginal = evaluation_predictions_realdata(df)
    out = pd.concat([df_bylabel_marginal, df_bylabel_conditional], axis=0)
    
  elif condition == 'by_size':
    df_bysize_conditional = pd.DataFrame(df.groupby(['prediction_size'])[['prediction_size','y_true','prediction_set','y_pred']].\
                              apply(evaluation_predictions_realdata, condition = 'by_size').reset_index().drop(['prediction_size','level_1'], axis=1))
    df_bysize_marginal = evaluation_predictions_realdata(df, condition = 'by_size')
    out = pd.concat([df_bysize_marginal, df_bysize_conditional], axis=0)

  elif condition == 'by_flag':
    try:
      assert(len(F_test)>0) 
    except:
      raise Exception("Corruption flag variable missing")
    df_byflag_conditional = pd.DataFrame(df.groupby(['flag'])[['prediction_size','y_true','prediction_set','y_pred', 'flag']].\
                              apply(evaluation_predictions_realdata, condition = 'by_flag').reset_index().drop(['flag','level_1'], axis=1))
    df_byflag_marginal = evaluation_predictions_realdata(df, condition = 'by_flag')
    out = pd.concat([df_byflag_marginal, df_byflag_conditional], axis=0)

  elif condition == 'by_label_flag':
    try:
      assert(len(F_test)>0) 
    except:
      raise Exception("Corruption flag variable missing")
    df_bylabelflag_conditional = pd.DataFrame(df.groupby(['y_true', 'flag'])[['prediction_size','y_true','prediction_set','y_pred', 'flag']].\
                              apply(evaluation_predictions_realdata, condition = 'by_label_flag').reset_index().drop(['flag','level_2','y_true'], axis=1))
    df_bylabelflag_marginal = pd.DataFrame(df.groupby(['y_true'])[['prediction_size','y_true','prediction_set','y_pred', 'flag']].\
                              apply(evaluation_predictions_realdata, condition = 'by_label').reset_index().drop(['y_true','level_1'], axis=1))
    df_bylabelflag_marginal['Flag'] = 'All'
    out = pd.concat([df_bylabelflag_marginal, df_bylabelflag_conditional], axis=0)


  # plot results
  if plot_result:
    print('Select a label to visualize')
    lab = int(input())
    visual_res, fig = pred_prob_on_true(lab, y_true, y_pred_prob, plot = True)
    return(dict({'conditional_stats':out,
                'hist_pred_prob_ranks': pd.DataFrame(visual_res),
                'fig': fig}))
    
  else:
    return(dict({'conditional_stats':out}))


