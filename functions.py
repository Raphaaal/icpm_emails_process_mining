from functions import * 
import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import spacy
import csv
import regex as re
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from itertools import chain
import numpy as np
import copy
import json
from spacy.pipeline import Sentencizer
import csv
import random
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from sklearn.metrics import v_measure_score, fowlkes_mallows_score
from itertools import combinations
from sklearn.metrics import f1_score, precision_recall_fscore_support
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import operator


def preprocess(emails_emb, nlp):
    # Make features categorically encoded for XGB
    emails_emb.Action = pd.Categorical(emails_emb.Action)
    emails_emb.domain = pd.Categorical(emails_emb.domain)
    # Encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(emails_emb['Action'])
    # One-hot encoder for activity
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    onehot_encoder = onehot_encoder.fit(np.array([elt for elt in emails_emb["Action"].values]).reshape(-1, 1))
    # One-hot encoder for domain
    ohe_domain = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    ohe_domain = ohe_domain.fit(np.array([elt for elt in emails_emb["domain"].values]).reshape(-1, 1))
    
    # Convert embeddings to numpy arrays
    emails_emb["subject_embedding"] = emails_emb.subject_embedding.apply(lambda x : np.matrix(x).A[0])
    emails_emb["body_embedding"] = emails_emb.body_embedding.apply(lambda x : np.matrix(x).A[0])
    emails_emb["named_entities_embedding"] = emails_emb.named_entities_embedding.apply(lambda x : np.matrix(x).A[0])

    # Extract named entities into list
    emails_emb['named_entities'] = emails_emb['named_entities'].str.replace('(',',')
    emails_emb['named_entities'] = emails_emb['named_entities'].str.replace(')',',')
    emails_emb['named_entities'] = emails_emb['named_entities'].str.split(',')

    # Extract recipients into lists
    emails_emb['To'] = emails_emb['To'].str.replace('[','')
    emails_emb['To'] = emails_emb['To'].str.replace(']','')
    emails_emb['To'] = emails_emb['To'].str.split(',')

    return emails_emb, label_encoder, onehot_encoder, ohe_domain

def filter_emails_for_split(emails_relational, filter_traces_list):
    # Remove traces from activities represented by less than 4 traces and single-email traces
    emails_relational = emails_relational[~emails_relational['Trace_ID'].isin(filter_traces_list)]
    print('Number of remaining emails: ', len(emails_relational["Email_ID"].drop_duplicates().index))
    return emails_relational

def keep_emails_for_split(emails_relational, filter_traces_list):
    # Remove traces from activities represented by less than 4 traces and single-email traces
    emails_relational = emails_relational[emails_relational['Trace_ID'].isin(filter_traces_list)]
    print('Number of remaining emails: ', len(emails_relational["Email_ID"].drop_duplicates().index))
    return emails_relational

def label_pairs_of_rows_from_ground_truth_activity(pairs_df):
    activities = pairs_df['Action'].drop_duplicates()
    if activities.shape[0] == 1:
        pairs_df['same_activity'] = 1
    else:
        pairs_df['same_activity'] = 0
    return pairs_df
    
def get_pairs_of_rows(df):
    pairs_df = []
    for index in list(combinations(df.index,2)):
        pairs_df.append(df.loc[index,:])
    pairs_df = [elt if elt.iloc[0]['Email_ID'] < elt.iloc[1]['Email_ID'] else pd.concat([elt.iloc[1], elt.iloc[0]], axis=1).transpose() for elt in pairs_df] # Keep ordered combinations
    return pairs_df

def label_pairs_of_rows_from_ground_truth_instances(pairs_df):
    trace_ids = pairs_df['Trace_ID'].drop_duplicates()
    if trace_ids.shape[0] == 1:
        pairs_df['same_instance'] = 1
    else:
        pairs_df['same_instance'] = 0
    return pairs_df

def get_pos_weight(train):
    pairs = get_pairs_of_rows(train)
    nb_neg = 0
    nb_pos = 0
    for pair in pairs:
        pair = label_pairs_of_rows_from_ground_truth_instances(pair)
        if pair['same_instance'].values[0] == 0 :
            nb_neg = nb_neg + 1 
        else:
            nb_pos = nb_pos + 1
    return nb_neg / nb_pos
    
def filter_emails_for_split(emails_relational, filter_traces_list):
    # Remove traces from activities represented by less than 4 traces and single-email traces
    emails_relational = emails_relational[~emails_relational['Trace_ID'].isin(filter_traces_list)]
    print('Number of remaining emails: ', len(emails_relational["Email_ID"].drop_duplicates().index))
    return emails_relational

def keep_emails_for_split(emails_relational, filter_traces_list):
    # Remove traces from activities represented by less than 4 traces and single-email traces
    emails_relational = emails_relational[emails_relational['Trace_ID'].isin(filter_traces_list)]
    print('Number of remaining emails: ', len(emails_relational["Email_ID"].drop_duplicates().index))
    return emails_relational
    
def split_train_test(df, instance_pct_train=0.5, train_bigger=True):
    actions = df[['Action', 'Trace_ID']].groupby('Action')['Trace_ID'].count().reset_index(name='count').sort_values('count', ascending=True)['Action'].tolist()
    scarce_actions = ['work started issue', 'reopen issue']
    frequent_actions = [x for x in actions if x not in scarce_actions]
    all_actions = scarce_actions + frequent_actions
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    used_traces_id = []
    
    for action in all_actions:
        action_traces = df[df['Action'] == action]['Trace_ID'].drop_duplicates().tolist()
        action_available_traces = [elt for elt in action_traces if elt not in used_traces_id]
        random.shuffle(action_available_traces)
        
        nb_instances = len(action_available_traces)
        nb_instances_train = int(nb_instances * instance_pct_train)
        action_traces_train = np.random.choice(action_available_traces, nb_instances_train, replace=False).tolist()
        
        train_df = pd.concat([train_df, df[df['Trace_ID'].isin(action_traces_train)]])
        test_df = pd.concat([test_df, df[ ( ~df['Trace_ID'].isin(action_traces_train) ) & ( df['Trace_ID'].isin(action_available_traces) )] ])
        
        used_traces_id = used_traces_id + action_available_traces
        
    if train_bigger:    
        # Use the biggest dataset as train, the other as test
        if train_df.shape[0] < test_df.shape[0]:
            train_df_big = copy.deepcopy(test_df)
            test_df = copy.deepcopy(train_df)
            train_df = train_df_big
    return train_df, test_df

def evaluate_instances_discovery(pairs, gt_col, pred_col):
    a = 0
    b = 0
    c = 0
    for pair in pairs:
        if (pair[gt_col].values[0] == 1) and (pair[pred_col].values[0] == 1):
            a = a+1
        if (pair[gt_col].values[0] == 0) and (pair[pred_col].values[0] == 1):
            b = b+1
        if (pair[gt_col].values[0] == 1) and (pair[pred_col].values[0] == 0):
            c = c+1
    precision = a / (a+b)
    recall = a / (a+c)    
    f_score = 2 / ((1/precision) + (1/recall))
    return (precision, recall, f_score)

# Plot functions code comes from https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
def plot_learning_curve(fitting_results, loss='mlogloss'):
    epochs = len(fitting_results['validation_0'][loss])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, fitting_results['validation_0'][loss], label='Train')
    ax.plot(x_axis, fitting_results['validation_1'][loss], label='Test')
    ax.legend()
    plt.show()
    
def plot_roc_curve(targets, preds):    
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    
def get_trace_from_insts_preds(X, X_pairs, same_instance_pred_col, trace_pred_col):
    X_copy = copy.deepcopy(X)
    X_pairs_copy = copy.deepcopy(X_pairs)
    # Create trace records from pairs
    trace_dict = {}
    trace_id_counter = 0
    for pair in X_pairs_copy:
        email1 = pair['Email_ID'].tolist()[0]
        email2 = pair['Email_ID'].tolist()[1]
        same_instance_pred = pair[same_instance_pred_col].drop_duplicates().values[0] == 1
        
        # Pair predicted as from same instance
        if same_instance_pred :
            already = False
            for trace in trace_dict.keys():
                if email1 in trace_dict[trace]['y'] or email2 in trace_dict[trace]['y'] :
                    trace_dict[trace]['y'].append(email2)
                    trace_dict[trace]['y_probas'].append(pair[same_instance_pred_col].values[0])
                    trace_dict[trace]['y'].append(email1)
                    trace_dict[trace]['y_probas'].append(pair[same_instance_pred_col].values[0])
                    already = True
            if not already:
                trace_dict[trace_id_counter] = {
                    'y': [email1, email2], 
                    'n': [], 
                    'y_probas': [pair[same_instance_pred_col].values[0], pair[same_instance_pred_col].values[0]],
                    'n_probas': []
                }
                trace_id_counter = trace_id_counter + 1
                
        # Pair predicted as from different instances
        else:
            absent_email1 = True
            absent_email2 = True
            for trace in trace_dict.keys():
                if email1 in trace_dict[trace]['y'] :
                    trace_dict[trace]['n'].append(email2)
                    trace_dict[trace]['n_probas'].append(pair[same_instance_pred_col].values[0])
                    absent_email1 = False
                if email2 in trace_dict[trace]['y'] :
                    trace_dict[trace]['n'].append(email1)
                    trace_dict[trace]['n_probas'].append(pair[same_instance_pred_col].values[0])
                    absent_email2 = False
            if absent_email1:
                trace_dict[trace_id_counter] = {
                    'y' : [email1], 
                    'n' : [email2], 
                    'n_probas': [pair[same_instance_pred_col].values[0]],
                    'y_probas': [0]
                }
                trace_id_counter = trace_id_counter + 1
            if absent_email2:
                trace_dict[trace_id_counter] = {
                    'y' : [email2], 
                    'n' : [email1],
                    'n_probas': [pair[same_instance_pred_col].values[0]],
                    'y_probas': [0]
                }
                trace_id_counter = trace_id_counter + 1
                
    # Get trace id for each email
    final_dict = {}
    email_ids = X_copy['Email_ID'].drop_duplicates().tolist()
    trace_ids = set(trace_dict.keys())
    tmp_dict = {}
    for e_id in email_ids:
        trace_scores = {}
        for trace in trace_ids:
            trace_scores[trace] = 0
            if e_id in trace_dict[trace]['y']:
                e_id_y_idxs = []
                for idx, x in enumerate(trace_dict[trace]['y']):
                    if x == e_id:
                        e_id_y_idxs.append(idx)
                y_proba = [ trace_dict[trace]['y_probas'][i] for i in e_id_y_idxs]
                for proba in y_proba:
                    trace_scores[trace] = trace_scores[trace] + proba
            if e_id in trace_dict[trace]['n']:
                trace_scores[trace] = trace_scores[trace] - trace_dict[trace]['n_probas'][trace_dict[trace]['n'].index(e_id)]
        final_dict[e_id] = max(trace_scores.items(), key=operator.itemgetter(1))[0]
        tmp_dict[e_id] = trace_scores
            
    # update df
    X_copy[trace_pred_col] = X_copy['Email_ID'].map(final_dict)
    return X_copy

def get_max_date_diff(train):
    min_date = train["date_embedding"].min()
    max_date = train["date_embedding"].max()
    return max_date-min_date

def get_weights_sklearn(train):
    train_copy = copy.deepcopy(train)
    weights_array = sklearn.utils.class_weight.compute_class_weight('balanced', classes=train_copy['Action'].drop_duplicates().tolist(), y=train_copy['Action'].values)
    acts =  train_copy['Action'].drop_duplicates().reset_index()
    acts_weights = pd.concat([acts, pd.Series(weights_array)], axis=1).reset_index()
    acts_weights['weight'] = acts_weights[0]
    acts_weights = acts_weights[['Action', 'weight']]
    train_copy = train_copy.merge(
        acts_weights[['Action', 'weight']],
        'inner',
        'Action'
    )
    return train_copy['weight'].to_numpy()

def get_avg_max_length_inst(preds_df, trace_col):
    min_length = preds_df.groupby(trace_col).agg({'date_embedding' : ['min']})
    min_length.columns = ["_".join(x) for x in min_length.columns.ravel()]
    
    max_length = preds_df.groupby(trace_col).agg({'date_embedding' : ['max']})
    max_length.columns = ["_".join(x) for x in max_length.columns.ravel()]
    
    traces = min_length.merge(max_length, on=trace_col, how='inner')
    traces['total_length'] = traces['date_embedding_max'] - traces['date_embedding_min'] 
    avg_length = traces['total_length'].mean()
    max_length = traces['total_length'].max()
    return avg_length, max_length

def get_avg_max_steps_inst(preds_df, trace_col):
    count_length = preds_df.groupby(trace_col).agg({'Email_ID' : ['count']})
    count_length.columns = ["_".join(x) for x in count_length.columns.ravel()]
    avg_length = count_length['Email_ID_count'].mean()
    max_length = count_length['Email_ID_count'].max()
    return avg_length, max_length

def get_avg_max_nb_users_inst(preds, trace_col):
    preds_df = copy.deepcopy(preds)
    trace_preds_from = preds_df.groupby(trace_col)['From'].apply(list).reset_index(name='from_list')
    trace_preds_to = preds_df.groupby(trace_col)['To'].apply(list).reset_index(name='to_list')
    trace_preds = trace_preds_from.merge(trace_preds_to, on=trace_col, how='inner')
    
    trace_preds['recipients_list'] = trace_preds['from_list'] + trace_preds['to_list'].map(lambda row : [i for sublist in row for i in sublist]) 
    trace_preds['recipients_list'] = trace_preds['recipients_list'].map(lambda x : set(x))
    trace_preds['nb_users'] = trace_preds['recipients_list'].map(lambda x : len(list(x)))

    avg_nb_users = trace_preds['nb_users'].mean()
    max_nb_users = trace_preds['nb_users'].max()
    return avg_nb_users, max_nb_users

def get_avg_max_length_act(preds, act_col, label_encoder):
    preds_df = copy.deepcopy(preds)
    if act_col == 'act_pred':
        preds_df['act_pred_label'] = preds_df[act_col].map(lambda x : label_encoder.inverse_transform(np.array([x]))[0])
        act_col = 'act_pred_label'
    preds_df = preds_df.sort_values('Email_ID')
    preds_df['previous_email'] = preds_df['Email_ID'].shift(periods=1)
    preds_df['previous_email_date_emb'] = preds_df['date_embedding'].shift(periods=1)
    preds_df['act_length'] =  preds_df['date_embedding'] - preds_df['previous_email_date_emb']
    preds_df = preds_df.dropna(subset=['act_length'])
    
    avg_length_act_df = preds_df.groupby(act_col).agg({'act_length': ['mean']})
    avg_length_act_df = avg_length_act_df.to_dict()
    
    max_length_act_df = preds_df.groupby(act_col).agg({'act_length': ['max']})
    max_length_act_df = max_length_act_df.to_dict()
    
    return avg_length_act_df, max_length_act_df

def get_avg_max_nb_users_act(preds, act_col, label_encoder):
    preds_df = copy.deepcopy(preds)
    if act_col == 'act_pred':
        preds_df['act_pred_label'] = preds_df[act_col].map(lambda x : label_encoder.inverse_transform(np.array([x]))[0])
        act_col = 'act_pred_label'
    trace_preds_from = preds_df.groupby(act_col)['From'].apply(list).reset_index(name='from_list')
    trace_preds_to = preds_df.groupby(act_col)['To'].apply(list).reset_index(name='to_list')
    trace_preds = trace_preds_from.merge(trace_preds_to, on=act_col, how='inner')
    
    trace_preds['recipients_list'] = trace_preds['from_list'] + trace_preds['to_list'].map(lambda row : [i for sublist in row for i in sublist]) 
    trace_preds['recipients_list'] = trace_preds['recipients_list'].map(lambda x : set(x))
    trace_preds['nb_users'] = trace_preds['recipients_list'].map(lambda x : len(list(x)))

    avg_nb_users_act_df = trace_preds.groupby(act_col).agg({'nb_users': ['mean']})
    avg_nb_users_act_df = avg_nb_users_act_df.to_dict()
    
    max_nb_users_act_df = trace_preds.groupby(act_col).agg({'nb_users': ['max']})
    max_nb_users_act_df = max_nb_users_act_df.to_dict()
    
    return avg_nb_users_act_df, max_nb_users_act_df
    
def plot_precision_recall_curve(targets, preds):    
    precision, recall, _ = precision_recall_curve(targets, preds)
    plt.figure()
    lw = 2
    plt.plot(precision, recall, color='darkorange',
             lw=lw, label='Precision-Recall curve')
    # plt.plot([0.0, 0.5], [1.0, 0.5], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.legend(loc="upper right")
    plt.show()
    return _
    
    
# BASELINES 

# Instances

def extract_features_from_pair(pair, max_date_diff):
    pair = label_pairs_of_rows_from_ground_truth_instances(pair)
    
    f_subject_sim = cosine_similarity(np.array([elt for elt in pair["subject_embedding"].values]))[0][1]
    
    f_body_sim = cosine_similarity(np.array([elt for elt in pair["body_embedding"].values]))[0][1]
    
    f_date_diff = [elt for elt in pair["date_embedding"].values]
    f_date_diff = (f_date_diff[1] - f_date_diff[0]) / max_date_diff

    senders = pair["From"].values
    receivers = pair["To"].values
    recipients_e1 = [senders[0], receivers[0]]
    recipients_e2 = [senders[1], receivers[1]]
    common_senders = [elt for elt in recipients_e1 if elt in recipients_e2]
    f_nb_common_senders = len(common_senders)

    domains = pair['domain'].values
    f_same_domain = 0
    if domains[0] == domains[1]:
      f_same_domain = 1
      
    email1_ne = [elt for elt in pair["named_entities"].values[0]]
    email2_ne = [elt for elt in pair["named_entities"].values[1]]
    common_ne = [elt for elt in email1_ne if (elt in email2_ne and elt != '')]
    f_nb_common_ne = len(common_ne)
    
    f_sim_ne = 0
    if (pair['named_entities_embedding'].values[0] is not np.nan) and (pair['named_entities_embedding'].values[1] is not np.nan) :
        ne_pair = np.vstack((pair['named_entities_embedding'].values[0], pair['named_entities_embedding'].values[1]))
        f_sim_ne = cosine_similarity(ne_pair)[0][1]
    
    result = np.array([f_subject_sim, f_body_sim, f_date_diff, f_nb_common_senders, f_nb_common_ne, f_same_domain, f_sim_ne])
    return result

def get_X_y_instances(train, nlp):
    # prepare data
    train_pairs = get_pairs_of_rows(train)
    X_instances = []
    y_instances = []
    max_date_diff = get_max_date_diff(train)
    for pair in train_pairs:
        X_instances.append(extract_features_from_pair(pair, max_date_diff))
        y = pair['same_instance'].values[0]
        y_instances.append(y)
    X_instances = np.array(X_instances)
    y_instances = np.array(y_instances)
    
    return train_pairs, X_instances, y_instances   

# Fitting function code comes from https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit_instances(alg, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=20, verbose=True, seed=42):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='error', early_stopping_rounds=early_stopping_rounds, seed=seed)
        if verbose:
            print("Best iteration: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    eval_set = [(X_train, y_train), (X_test, y_test)]
    alg.fit(X_train, y_train, eval_set=eval_set, eval_metric='logloss', verbose=False)
        
    #Predict :
    dtrain_predprob = alg.predict(X_train)
    dtest_predprob = alg.predict(X_test)
        
    #Print model report:
    if verbose:
        print ("\nModel Report")
        print ("F Score (Train): %f" % sklearn.metrics.f1_score(y_train, dtrain_predprob))   
        print ("F Score (Test): %f" % sklearn.metrics.f1_score(y_test, dtest_predprob))   
    
    return alg, cvresult.shape[0], sklearn.metrics.f1_score(y_test, dtest_predprob)

def hyperparameter_search_instances(train, X_train, y_train, X_test, y_test, seed=42):
    
    current_lr = 0.1
    xgb1 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    best_n_estimators = modelfit_instances(xgb1, X_train, y_train, X_test, y_test, verbose=False)[1]


    param_test1 = {
     'max_depth':range(1,10,1),
     'min_child_weight':range(1,10,1)
    }
    gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test1, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch1.fit(X_train, y_train)
    best_max_depth = gsearch1.best_params_['max_depth']
    best_min_child_weight = gsearch1.best_params_['min_child_weight']


    param_test2 = {
     'gamma':[i/10.0 for i in range(0,4)]
    }
    gsearch2 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=0, subsample=0.8, 
                colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test2, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch2.fit(X_train, y_train)
    best_gamma = gsearch2.best_params_['gamma']


    current_lr = 0.1
    xgb2 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    best_n_estimators = modelfit_instances(xgb2, X_train, y_train, X_test, y_test, verbose=False)[1]


    param_test3 = {
     'subsample':[i/10.0 for i in range(1,11)],
     'colsample_bytree':[i/10.0 for i in range(1,11)]
    }
    gsearch3 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=0.8, 
                colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test3, 
            scoring='f1',
            n_jobs=4, 
            cv=4
    )
    gsearch3.fit(X_train, y_train)
    best_subsample = gsearch3.best_params_['subsample']
    best_colsample_bytree = gsearch3.best_params_['colsample_bytree']


    param_test4 = {
        'reg_alpha':[1e-5, 1e-2, 0, 0.1, 1, 100]
    }
    gsearch4 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=best_subsample, 
                colsample_bytree=best_colsample_bytree, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test4, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch4.fit(X_train, y_train)
    best_reg_alpha = gsearch4.best_params_['reg_alpha']


    current_lr = 0.1
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    alg_lr_01 = modelfit_instances(xgb3, X_train, y_train, X_test, y_test, verbose=False)


    current_lr = 0.01
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=10000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    alg_lr_001 = modelfit_instances(xgb3, X_train, y_train, X_test, y_test, early_stopping_rounds=100, verbose=False)


    lr = 0.1
    early_stoping_rounds = 20
    best_n_estimators = 1000
    
    if alg_lr_01[2] < alg_lr_001[2]:
        lr = 0.01
        early_stoping_rounds = 100
        best_n_estimators = 10000
        
    bst = XGBClassifier(
        learning_rate = lr,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    
    return bst, early_stoping_rounds
    
def supervized_baseline_instances_structured(train, eval_set, nlp, seed=42, save_path="baseline_inst.pickle.dat"):
    
    train_pairs, X_train, y_train = get_X_y_instances(train, nlp)
    test_pairs, X_test, y_test = get_X_y_instances(eval_set, nlp)
    
    # Train XGB classifier
    hyperparams = hyperparameter_search_instances(train, X_train, y_train, X_test, y_test, seed)
    clf_instances = modelfit_instances(hyperparams[0], X_train, y_train, X_test, y_test, early_stopping_rounds=hyperparams[1], verbose=False)[0]
    
    pickle.dump(clf_instances, open(save_path, "wb"))
    
    return clf_instances
    
# Activities 

def get_X_y_activities(train, label_encoder, ohe_domain):
    # Convert to arrays
    subject_emb = np.array([elt for elt in train["subject_embedding"].values])
    body_emb = np.array([elt for elt in train["body_embedding"].values])
    X =( body_emb + subject_emb ) /2

    f_domain = np.array(train['domain'].values).reshape(-1, 1)
    f_domain_dummy = ohe_domain.transform(f_domain)

    X = np.hstack((X, f_domain_dummy))

    y = label_encoder.transform(train['Action'])

    return X, y

# Fitting function code comes from https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit_activities(alg, train, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=20, verbose=True, seed=42):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train, weight=get_weights_sklearn(train))
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, seed=seed)
        if verbose:
            print("Best iteration: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    eval_set = [(X_train, y_train), (X_test, y_test)]
    alg.fit(X_train, y_train, eval_set=eval_set, eval_metric='mlogloss', verbose=False)
        
    #Predict :
    dtrain_predprob = alg.predict(X_train)
    dtest_predprob = alg.predict(X_test)
    
    score_train = precision_recall_fscore_support(y_train, dtrain_predprob, average='micro')[2]
    score_test = precision_recall_fscore_support(y_test, dtest_predprob, average='micro')[2]
        
    #Print model report:
    if verbose:
        print ("\nModel Report")
        print ("F Score (Train): %f" % score_train)   
        print ("F Score (Test): %f" % score_test)   
    
    return alg, cvresult.shape[0], score_test

def hyperparameter_search_activities(label_encoder, train, X_train, y_train, X_test, y_test, seed=42):
    current_lr = 0.1

    xgb1 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.3,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        seed=seed, 
        num_class=len(label_encoder.classes_)
    )
    best_n_estimators = modelfit_activities(xgb1, train, X_train, y_train, X_test, y_test, verbose=False)[1]
    
    param_test1 = {
     'max_depth':range(1,10,1),
     'min_child_weight':range(1,10,1)
    }
    gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                colsample_bytree=0.3, objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test1, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch1.fit(X_train, y_train)
    best_max_depth = gsearch1.best_params_['max_depth']
    best_min_child_weight = gsearch1.best_params_['min_child_weight']
    
    
    param_test2 = {
     'gamma':[i/100.0 for i in range(0,40)]
    }
    gsearch2 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=0, subsample=0.8, 
                colsample_bytree=0.3, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test2, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch2.fit(X_train, y_train)
    best_gamma = gsearch2.best_params_['gamma']
    
    
    xgb2 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=0.8,
        colsample_bytree=0.3,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    best_n_estimators = modelfit_activities(xgb2, train, X_train, y_train, X_test, y_test, verbose=False)[1]
    
    
    param_test3 = {
     'subsample':[i/10.0 for i in range(1,11)],
     'colsample_bytree':[i/10.0 for i in range(1,11)]
    }
    gsearch3 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=0.8, 
                colsample_bytree=0.8, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test3, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch3.fit(X_train, y_train)
    best_subsample = gsearch3.best_params_['subsample']
    best_colsample_bytree = gsearch3.best_params_['colsample_bytree']
    
    
    param_test4 = {
        'reg_alpha':[1e-5, 1e-2, 0, 0.1, 1, 100]
    }
    gsearch4 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=best_subsample, 
                colsample_bytree=best_colsample_bytree, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test4, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch4.fit(X_train, y_train)
    best_reg_alpha = gsearch4.best_params_['reg_alpha']
    
    
    current_lr = 0.1
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    alg_lr_01 = modelfit_activities(xgb3, train, X_train, y_train, X_test, y_test, verbose=False)
    
    
    xgb3 = XGBClassifier(
        learning_rate = 0.01,
        n_estimators=10000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    alg_lr_001 = modelfit_activities(xgb3, train, X_train, y_train, X_test, y_test, early_stopping_rounds=100, verbose=False)
    
    
    lr = 0.1
    early_stoping_rounds = 20
    best_n_estimators = 1000
    
    if alg_lr_01[2] < alg_lr_001[2]:
        lr = 0.01
        early_stoping_rounds = 100
        best_n_estimators = 10000
        
    bst = XGBClassifier(
        learning_rate = lr,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective='multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )

    return bst, early_stoping_rounds
    
def supervized_baseline_activities_structured(train, test, label_encoder, ohe_domain, seed=42, save_path="baseline_act.pickle.dat"):
    train_df = copy.deepcopy(train)

    # Convert to arrays
    X_emb, y = get_X_y_activities(train, label_encoder, ohe_domain)
    X_emb_eval, y_eval = get_X_y_activities(test, label_encoder, ohe_domain)
    
    # Train XGB classifier
    hyperparams = hyperparameter_search_activities(label_encoder, train, X_emb, y, X_emb_eval, y_eval, seed)
    clf_activities = modelfit_activities(hyperparams[0], train, X_emb, y, X_emb_eval, y_eval, early_stopping_rounds=hyperparams[1], verbose=False)[0]
    
    pickle.dump(clf_activities, open(save_path, "wb"))
    
    return clf_activities
    
# RELATIONAL

# Instances

def extract_features_from_pair_relational(pair, activity_col, onehot_encoder, max_date_diff):
    pair = label_pairs_of_rows_from_ground_truth_instances(pair)
    
    f_subject_sim = cosine_similarity(np.array([elt for elt in pair["subject_embedding"].values]))[0][1]
    
    f_body_sim = cosine_similarity(np.array([elt for elt in pair["body_embedding"].values]))[0][1]
    
    f_date_diff = [elt for elt in pair["date_embedding"].values]
    f_date_diff = (f_date_diff[1] - f_date_diff[0]) / max_date_diff

    senders = pair["From"].values
    receivers = pair["To"].values
    recipients_e1 = [senders[0], receivers[0]]
    recipients_e2 = [senders[1], receivers[1]]
    common_senders = [elt for elt in recipients_e1 if elt in recipients_e2]
    f_nb_common_senders = len(common_senders)

    domains = pair['domain'].values
    f_same_domain = 0
    if domains[0] == domains[1]:
      f_same_domain = 1
    
    email1_ne = [elt for elt in pair["named_entities"].values[0]]
    email2_ne = [elt for elt in pair["named_entities"].values[1]]
    common_ne = [elt for elt in email1_ne if elt in email2_ne]
    f_nb_common_ne = len(common_ne)
    
    f_sim_ne = 0
    if (pair['named_entities_embedding'].values[0] is not np.nan) and (pair['named_entities_embedding'].values[1] is not np.nan) :
        ne_pair = np.vstack((pair['named_entities_embedding'].values[0], pair['named_entities_embedding'].values[1]))
        f_sim_ne = cosine_similarity(ne_pair)[0][1]
    
    old_features = np.array([f_subject_sim, f_body_sim, f_date_diff, f_nb_common_senders, f_nb_common_ne, f_same_domain, f_sim_ne])
    
    f_activities = np.array(pair[activity_col].values).reshape(-1, 1)
    f_activities_dummy = onehot_encoder.transform(f_activities).flatten()

    result = np.hstack((old_features, f_activities_dummy))

    return result

def get_X_y_instances_rel(train, activity_col, nlp, onehot_encoder):
    # prepare data
    train_pairs = get_pairs_of_rows(train) 
    X_instances = []
    y_instances = []
    max_date_diff = get_max_date_diff(train)
    for pair in train_pairs: 
        X_instances.append(extract_features_from_pair_relational(pair, activity_col, onehot_encoder, max_date_diff))
        y = pair['same_instance'].values[0]
        y_instances.append(y)

    X_instances = np.array(X_instances)
    y_instances = np.array(y_instances)
    
    return train_pairs, X_instances, y_instances

# Fitting function code comes from https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit_instances_rel(alg, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=20, verbose=True):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='error', early_stopping_rounds=early_stopping_rounds)
        if verbose:
            print("Best iteration: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    eval_set = [(X_train, y_train), (X_test, y_test)]
    alg.fit(X_train, y_train, eval_set=eval_set, eval_metric='logloss', verbose=False)
        
    #Predict in two steps:
    dtrain_predprob = alg.predict(X_train)
    dtest_predprob = alg.predict(X_test)
        
    #Print model report:
    if verbose:
        print ("\nModel Report")
        print ("F Score (Train): %f" % sklearn.metrics.f1_score(y_train, dtrain_predprob))
        print ("F Score (Test): %f" % sklearn.metrics.f1_score(y_test, dtest_predprob))   
    
    return alg, cvresult.shape[0], sklearn.metrics.f1_score(y_test, dtest_predprob)

def hyperparameter_search_instances_rel(train, X_train, y_train, X_test, y_test, seed):
    
    current_lr = 0.1
    xgb1 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    best_n_estimators = modelfit_instances_rel(xgb1, X_train, y_train, X_test, y_test, verbose=False)[1]


    param_test1 = {
     'max_depth':range(1,10,1),
     'min_child_weight':range(1,10,1)
    }
    gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test1, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch1.fit(X_train, y_train)
    best_max_depth = gsearch1.best_params_['max_depth']
    best_min_child_weight = gsearch1.best_params_['min_child_weight']


    param_test2 = {
     'gamma':[i/10.0 for i in range(0,4)]
    }
    gsearch2 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=0, subsample=0.8, 
                colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test2, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch2.fit(X_train, y_train)
    best_gamma = gsearch2.best_params_['gamma']


    current_lr = 0.1
    xgb2 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    best_n_estimators = modelfit_instances_rel(xgb2, X_train, y_train, X_test, y_test, verbose=False)[1]


    param_test3 = {
     'subsample':[i/10.0 for i in range(1,11)],
     'colsample_bytree':[i/10.0 for i in range(1,11)]
    }
    gsearch3 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=0.8, 
                colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=seed
            ), 
            param_grid=param_test3, 
            scoring='f1',
            n_jobs=4, 
            cv=4
    )
    gsearch3.fit(X_train, y_train)
    best_subsample = gsearch3.best_params_['subsample']
    best_colsample_bytree = gsearch3.best_params_['colsample_bytree']


    param_test4 = {
        'reg_alpha':[1e-5, 1e-2, 0, 0.1, 1, 100]
    }
    gsearch4 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=best_subsample, 
                colsample_bytree=best_colsample_bytree, objective='binary:logistic', nthread=4, scale_pos_weight=get_pos_weight(train), seed=0
            ), 
            param_grid=param_test4, 
            scoring='f1',
            n_jobs=4, 
            cv=5
    )
    gsearch4.fit(X_train, y_train)
    best_reg_alpha = gsearch4.best_params_['reg_alpha']


    current_lr = 0.1
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    alg_lr_01 = modelfit_instances_rel(xgb3, X_train, y_train, X_test, y_test, verbose=False)


    current_lr = 0.01
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=10000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    alg_lr_001 = modelfit_instances_rel(xgb3, X_train, y_train, X_test, y_test, early_stopping_rounds=100, verbose=False)


    lr = 0.1
    early_stoping_rounds = 20
    best_n_estimators = 1000
    
    if alg_lr_01[2] < alg_lr_001[2]:
        lr = 0.01
        early_stoping_rounds = 100
        best_n_estimators = 10000
        
    bst = XGBClassifier(
        learning_rate = lr,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=get_pos_weight(train),
        seed=seed
    )
    
    return bst, early_stoping_rounds
    
def supervized_relational_instances_structured(train, test, nlp, onehot_encoder, seed=42, save_path="relational_inst.pickle.dat"):
    
    train_pairs, X_train, y_train = get_X_y_instances_rel(train, 'Action', nlp, onehot_encoder)
    test_pairs, X_test, y_test = get_X_y_instances_rel(test, 'Action', nlp, onehot_encoder)
    
    # Train XGB classifier
    hyperparams = hyperparameter_search_instances_rel(train, X_train, y_train, X_test, y_test, seed)
    clf_instances = modelfit_instances_rel(hyperparams[0], X_train, y_train, X_test, y_test, early_stopping_rounds=hyperparams[1], verbose=False)[0]
    
    pickle.dump(clf_instances, open(save_path, "wb"))
    
    return clf_instances
    
# Activities

def get_previous_actions_feature_gen(email_id, train, trace_col, act_col, onehot_encoder, ohe_domain, generation=1):
    date_email = train[train['Email_ID'] == email_id]['date_embedding'].values[0]
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=False).values
    
    next_emails_ids = [elt for elt in trace_emails_ids if elt < email_id]
    
    prev_action = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_action = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)][act_col].values
        prev_domain = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['domain'].values
    if len(prev_action) > 0:
        encoding = onehot_encoder.transform(np.array(prev_action).reshape(-1, 1))
        f_domain = np.array(prev_domain).reshape(-1, 1)
        f_domain_dummy = ohe_domain.transform(f_domain)
        res= np.hstack((np.array(encoding[0]), f_domain_dummy[0]))
        return res
    else:
        res = np.hstack((np.zeros(train['Action'].cat.categories.shape[0]), np.zeros(train['domain'].cat.categories.shape[0])))
        return res

def get_prev_date_diff_gen(email_id, train, trace_col, generation=1):
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=True).values
    next_emails_ids = [elt for elt in trace_emails_ids if elt < email_id]
    date_email = train[train['Email_ID'] == email_id]['date_embedding'].values[0]
    
    prev_date_diff = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_date_diff = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['date_embedding'].values
    if len(prev_date_diff) > 0:
        return prev_date_diff[0] + date_email
    else:
        return 0
    
def get_next_date_diff_gen(email_id, train, trace_col, generation=1):
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=True).values
    next_emails_ids = [elt for elt in trace_emails_ids if elt > email_id]
    date_email = train[train['Email_ID'] == email_id]['date_embedding'].values[0]
    
    prev_date_diff = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_date_diff = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['date_embedding'].values
    if len(prev_date_diff) > 0:
        return prev_date_diff[0] - date_email
    else:
        return 0
    
def get_next_embedding_gen(email_id, train, trace_col, generation=1):
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=True).values
    
    next_emails_ids = [elt for elt in trace_emails_ids if elt > email_id]
    
    prev_emb = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_emb = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['subject_embedding'].values
    if len(prev_emb) > 0:
        return prev_emb[0]
    else:
        return np.zeros(train['subject_embedding'].values[0].shape)
    
def get_prev_embedding_gen(email_id, train, trace_col, generation=1):
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=True).values
    
    next_emails_ids = [elt for elt in trace_emails_ids if elt < email_id]
    
    prev_emb = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_emb = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['subject_embedding'].values
    if len(prev_emb) > 0:
        return prev_emb[0]
    else:
        return np.zeros(train['subject_embedding'].values[0].shape)
     
def get_next_actions_feature_gen(email_id, train, trace_col, act_col, onehot_encoder, ohe_domain, generation=1):
    date_email = train[train['Email_ID'] == email_id]['date_embedding'].values[0]
    email_trace = train[train['Email_ID'] == email_id][trace_col].values[0]
    trace_emails_ids = train[train[trace_col] == email_trace]['Email_ID'].sort_values(ascending=True).values

    next_emails_ids = [elt for elt in trace_emails_ids if elt > email_id]
    
    prev_action = []
    if generation < len(next_emails_ids):
        gen_email_id = next_emails_ids[generation-1]
        prev_action = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)][act_col].values
        prev_domain = train[(train[trace_col] == email_trace) & (train['Email_ID'] == gen_email_id)]['domain'].values
    if len(prev_action) > 0:
        encoding = onehot_encoder.transform(np.array(prev_action).reshape(-1, 1))
        f_domain = np.array(prev_domain).reshape(-1, 1)
        f_domain_dummy = ohe_domain.transform(f_domain)

        return np.hstack((np.array(encoding[0]), f_domain_dummy[0]))
    
    else:
        return np.hstack((np.zeros(train['Action'].cat.categories.shape[0]), np.zeros(train['domain'].cat.categories.shape[0])))
    
def get_X_y_activities_rel(train, trace_col, act_col, label_encoder, onehot_encoder, ohe_domain, generations=15):
    # Convert to arrays
    subject_emb = np.array([elt for elt in train["subject_embedding"].values])
    body_emb = np.array([elt for elt in train["body_embedding"].values])
    tmp_X = (body_emb + subject_emb) / 2
    f_domain = np.array(train['domain'].values).reshape(-1, 1)
    f_domain_dummy = ohe_domain.transform(f_domain)

    tmp_X = np.hstack((tmp_X, f_domain_dummy))
    
    for i in range(0, generations+1):
        prev_feature_name = 'prev_actions_embedding_' + str(i)
        next_feature_name = 'next_actions_embedding_' + str(i)
        
        prev_emb_name = 'prev_embedding_' + str(i)
        next_emb_name = 'next_embedding_' + str(i)
        
        prev_date_diff = 'prev_date_diff_' + str(i)
        next_date_diff = 'next_date_diff_' + str(i)
        
        train[prev_feature_name] = train['Email_ID'].map(lambda x : get_previous_actions_feature_gen(x, train, trace_col, act_col, onehot_encoder, ohe_domain, i))
        train[next_feature_name] = train['Email_ID'].map(lambda x : get_next_actions_feature_gen(x, train, trace_col, act_col, onehot_encoder, ohe_domain, i))
        X_prev_action_gen = np.array([elt for elt in train[prev_feature_name].values])
        X_next_action_gen = np.array([elt for elt in train[next_feature_name].values])
        
        train[prev_emb_name] = train['Email_ID'].map(lambda x : get_prev_embedding_gen(x, train, trace_col, i))
        train[next_emb_name] = train['Email_ID'].map(lambda x : get_next_embedding_gen(x, train, trace_col, i))
        X_prev_emb = np.array([elt for elt in train[prev_emb_name].values])
        X_next_emb = np.array([elt for elt in train[next_emb_name].values])
        
        train[prev_date_diff] = train['Email_ID'].map(lambda x : get_prev_date_diff_gen(x, train, trace_col, i))
        train[next_date_diff] = train['Email_ID'].map(lambda x : get_next_date_diff_gen(x, train, trace_col, i))
        X_prev_date_diff = np.array([elt for elt in train[prev_date_diff].values]).reshape(-1,1)
        X_next_date_diff = np.array([elt for elt in train[next_date_diff].values]).reshape(-1,1)

        tmp_X = np.hstack((tmp_X, X_prev_action_gen, X_next_action_gen, X_prev_emb, X_next_emb, X_prev_date_diff, X_next_date_diff))
        
    X = tmp_X
    y = label_encoder.transform(train['Action'])

    return X, y

def nb_generations(train):
    emails_by_trace = train.groupby("Trace_ID").agg({'Email_ID' : ['count']})
    emails_by_trace.columns = ["_".join(x) for x in emails_by_trace.columns.ravel()]
    nb_gen = emails_by_trace['Email_ID_count'].max()
    return nb_gen
  
# Fitting function code comes from https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit_activities_rel(alg, train, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=20, verbose=True, seed=42):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train, weight=get_weights_sklearn(train))
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, seed=seed)
        if verbose:
            print("Best iteration: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    eval_set = [(X_train, y_train), (X_test, y_test)]
    alg.fit(X_train, y_train, eval_set=eval_set, eval_metric='mlogloss', verbose=False)
        
    #Predict :
    dtrain_predprob = alg.predict(X_train)
    dtest_predprob = alg.predict(X_test)
    
    score_train = precision_recall_fscore_support(y_train, dtrain_predprob, average='micro')[2]
    score_test = precision_recall_fscore_support(y_test, dtest_predprob, average='micro')[2]
        
    #Print model report:
    if verbose:
        print ("\nModel Report")
        print ("F Score (Train): %f" % score_train)   
        print ("F Score (Test): %f" % score_test)   
    
    return alg, cvresult.shape[0], score_test


def hyperparameter_search_activities_rel(label_encoder, train, X_train, y_train, X_test, y_test, seed):
    current_lr = 0.1

    xgb1 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.3,
        objective= 'multi:softmax',
        nthread=4,
        seed=seed,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_)
    )
    best_n_estimators = modelfit_activities_rel(xgb1, train, X_train, y_train, X_test, y_test, verbose=False)[1]
    
    
    param_test1 = {
     'max_depth':range(1,10,1),
     'min_child_weight':range(1,10,1)
    }
    gsearch1 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                colsample_bytree=0.3, objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test1, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch1.fit(X_train, y_train)
    best_max_depth = gsearch1.best_params_['max_depth']
    best_min_child_weight = gsearch1.best_params_['min_child_weight']
    
    
    param_test2 = {
     'gamma':[i/100.0 for i in range(0,40)]
    }
    gsearch2 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=0, subsample=0.8, 
                colsample_bytree=0.3, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test2, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch2.fit(X_train, y_train)
    best_gamma = gsearch2.best_params_['gamma']
    
    
    xgb2 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=0.8,
        colsample_bytree=0.3,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    best_n_estimators = modelfit_activities_rel(xgb2, train, X_train, y_train, X_test, y_test, verbose=False)[1]
    
    
    param_test3 = {
     'subsample':[i/10.0 for i in range(1,11)],
     'colsample_bytree':[i/10.0 for i in range(1,11)]
    }
    gsearch3 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=0.8, 
                colsample_bytree=0.8, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test3, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch3.fit(X_train, y_train)
    best_subsample = gsearch3.best_params_['subsample']
    best_colsample_bytree = gsearch3.best_params_['colsample_bytree']
    
    
    param_test4 = {
        'reg_alpha':[1e-5, 1e-2, 0, 0.1, 1, 100]
    }
    gsearch4 = GridSearchCV(
            estimator=XGBClassifier(
                learning_rate=current_lr, n_estimators=best_n_estimators, max_depth=best_max_depth, min_child_weight=best_min_child_weight, gamma=best_gamma, subsample=best_subsample, 
                colsample_bytree=best_colsample_bytree, objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=seed
            ), 
            param_grid=param_test4, 
            scoring='f1_micro',
            n_jobs=4, 
            cv=2
    )
    gsearch4.fit(X_train, y_train)
    best_reg_alpha = gsearch4.best_params_['reg_alpha']
    
    
    current_lr = 0.1
    xgb3 = XGBClassifier(
        learning_rate = current_lr,
        n_estimators=1000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    alg_lr_01 = modelfit_activities_rel(xgb3, train, X_train, y_train, X_test, y_test, verbose=False)
    
    
    xgb3 = XGBClassifier(
        learning_rate = 0.01,
        n_estimators=10000,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )
    alg_lr_001 = modelfit_activities_rel(xgb3, train, X_train, y_train, X_test, y_test, early_stopping_rounds=100, verbose=False)
    
    
    lr = 0.1
    early_stoping_rounds = 20
    best_n_estimators = 1000
    
    if alg_lr_01[2] < alg_lr_001[2]:
        lr = 0.01
        early_stoping_rounds = 100
        best_n_estimators = 10000
        
    bst = XGBClassifier(
        learning_rate = lr,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_child_weight=best_min_child_weight,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        reg_alpha=best_reg_alpha,
        objective='multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        num_class=len(label_encoder.classes_),
        seed=seed
    )

    return bst, early_stoping_rounds
    
def supervized_relational_activities_structured(train, test, label_encoder, onehot_encoder, ohe_domain, seed=42, save_path="relational_act.pickle.dat"):
    nb_gen = nb_generations(train) +1 
    X_train, y_train = get_X_y_activities_rel(train,'Trace_ID', 'Action',  label_encoder, onehot_encoder, ohe_domain, nb_gen)
    X_test, y_test = get_X_y_activities_rel(test, 'Trace_ID', 'Action', label_encoder, onehot_encoder, ohe_domain, nb_gen)

    # Train XGB classifier
    hyperparams = hyperparameter_search_activities_rel(label_encoder, train, X_train, y_train, X_test, y_test, seed)
    clf_act = modelfit_activities_rel(hyperparams[0], train, X_train, y_train, X_test, y_test, early_stopping_rounds=hyperparams[1], verbose=False)[0]
    
    pickle.dump(clf_act, open(save_path, "wb"))
    
    return clf_act, train, test

def unsupervized_baseline_instances(emails_df, verbose=False):
    emails_emb = emails_df.copy()
    
    # Add date epoch to features and normalize
    X_subject = np.array([elt for elt in emails_emb["subject_embedding"].values])
    X_body = np.array([elt for elt in emails_emb["body_embedding"].values])
    X_date = np.array([elt for elt in emails_emb["date_embedding"].values]).reshape(emails_emb.shape[0], 1)
    
    # Standardize
    X = np.hstack((X_subject, X_body, X_date))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    X_cosine_sim = cosine_similarity(X)
    avg_cosine_sim = 0
    for i in range(X_cosine_sim.shape[0]):
        for j in range(i+1, X_cosine_sim.shape[1]):
            avg_cosine_sim = avg_cosine_sim + X_cosine_sim[i][j]
    avg_cosine_sim = avg_cosine_sim / ((X_cosine_sim.shape[0]*X_cosine_sim.shape[0])/2)
    max_d = 1 - avg_cosine_sim
    if verbose:
        print("Max distance btw. clusters: ", max_d)

    # Hierarchical clustering of emails based on embeddings cosine similarity with stop condition
    Z = linkage(X, metric='cosine', method='complete')
    clusters = fcluster(Z, max_d, criterion='distance')
    
    return clusters