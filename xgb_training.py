# ------------------------------------------------------------------------- #
# Transaction Fraud Detection Prototype
# ------------------------------------------------------------------------- #

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
#from sklearn.model_selection import StratifiedKFold

root_dir = "S:/can/bank/Departments/Risk/Risk Analytics/rezenro/Projects/transaction_fraud/data/"
train_data = pd.read_csv(root_dir + "training.csv")

X = train_data.loc[:,'amount':].values
y = train_data.loc[:,'fraud_flag'].values
groups = train_data.loc[:, 'group'].values     

feature_names = train_data.columns.tolist()
feature_names = feature_names[feature_names.index('amount'):]

gkf = GroupKFold(n_splits=4)
#gkf = StratifiedKFold(n_splits=4, random_state=42)


#%% ------------------------------------------------------------------------ #
# XGBoost parameters

param = dict()
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['scale_pos_weight'] = 540
#param['updater'] = 'grow_gpu_hist'
#param['max_depth'] = 6
#param['min_child_weight'] = 100
#param['gamma'] = 100
#param['reg_lambda'] = 10
#param['reg_alpha'] = 500
#param['subsample'] = 0.8
#param['colsample_bytree'] = 0.5
#param['colsample_bylevel'] = 0.3
#param['learning_rate'] = 0.2

#%% ------------------------------------------------------------------------ #
# Model Training

start_time = time.time()


train_results = list()    
train_monitor = dict()

train_index, test_index = list(gkf.split(X, y, groups))[1]

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature_names)
dtest  = xgb.DMatrix(X_test,  label = y_test,  feature_names = feature_names)
evallist = [(dtrain,'train'), (dtest,'test')]

print("\nStarting training... \n")

classifier = xgb.train(param, 
                       dtrain, 
                       num_boost_round = 100, 
                       evals = evallist, 
                       verbose_eval=5,
                       early_stopping_rounds = 10,
                       evals_result = train_monitor)


dpred = xgb.DMatrix(X_test, feature_names = feature_names)
y_pred = classifier.predict(dpred)

for threshold in np.arange(0, 1, 0.05):
    y_thres = y_pred.copy()
    y_thres[y_thres <  threshold] = 0
    y_thres[y_thres >= threshold] = 1
    if y_thres.sum() < 70:
        break

fpr, tpr, _ = roc_curve(y_test, y_pred)

y_pred[y_pred <  threshold] = 0
y_pred[y_pred >= threshold] = 1

fold_results = dict()
fold_results['Type'] = 'Test'
fold_results['n_samples'] = len(y_test)
fold_results['n_frauds'] = y_test.sum()
fold_results['n_flagged'] = y_pred.sum()
fold_results['true_positives'] = (y_pred[y_test == 1] == y_test[y_test == 1]).sum()
train_results.append(fold_results)

dpred = xgb.DMatrix(X_train, feature_names = feature_names)
y_pred = classifier.predict(dpred)

y_pred[y_pred <  threshold] = 0
y_pred[y_pred >= threshold] = 1

fold_results = dict()
fold_results['Type'] = 'Train'
fold_results['n_samples'] = len(y_train)
fold_results['n_frauds'] = y_train.sum()
fold_results['n_flagged'] = int(y_pred.sum())
fold_results['true_positives'] = (y_pred[y_train == 1] == y_train[y_train == 1]).sum()
train_results.append(fold_results)
    
results = pd.DataFrame(train_results, columns = ['Type','n_samples','n_frauds','n_flagged','true_positives'])
results = results.sort_values(['Type'])

print("\nTraining Results:\n")

plt.plot(train_monitor['train']['auc'])
plt.plot(train_monitor['test']['auc'])
plt.title('Training AUC')
plt.ylabel('AUC')
plt.xlabel('Boost Round')
plt.legend(['train', 'test'], loc='lower right')
plt.minorticks_on()
plt.ylim((0.5,1.05))
plt.grid(b=True, which='major', color='black', linestyle='--')
plt.show()

plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(['ROC curve (area = %0.3f)' % train_monitor['test']['auc'][-1]], loc='lower right')
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='--')
plt.show()

xgb.plot_importance(classifier,max_num_features=15)
plt.show()

print('\n Results\n')
print(results)

print("\nTotal processing time: {:.2f} minutes".format((time.time()-start_time)/60))   

#%% ------------------------------------------------------------------------ #
# Cross Validation

start_time = time.time()

threshold = 0.85
num_rounds = 15
    
train_results = list()    

print("\nStarting cross validation... \n")
    
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dtrain = xgb.DMatrix(X_train, y_train)
    classifier = xgb.train(param, dtrain, num_boost_round = num_rounds)

    dpred = xgb.DMatrix(X_train)
    y_pred = classifier.predict(dpred)
    
    y_pred[y_pred <  threshold] = 0
    y_pred[y_pred >= threshold] = 1
    
    fold_results = dict()
    fold_results['Samples Set'] = 'Train - Split {}'.format(len(train_results)//2 + 1)
    fold_results['n_samples'] = len(y_train)
    fold_results['n_frauds'] = y_train.sum()
    fold_results['n_flagged'] = int(y_pred.sum())
    fold_results['true_positives'] = (y_pred[y_train == 1] == y_train[y_train == 1]).sum()
    train_results.append(fold_results)

    dpred = xgb.DMatrix(X_test)
    y_pred = classifier.predict(dpred)
    
    y_pred[y_pred <  threshold] = 0
    y_pred[y_pred >= threshold] = 1
    
    fold_results = dict()
    fold_results['Samples Set'] = 'Test - Split {}'.format(len(train_results)//2 + 1)
    fold_results['n_samples'] = len(y_test)
    fold_results['n_frauds'] = y_test.sum()
    fold_results['n_flagged'] = int(y_pred.sum())
    fold_results['true_positives'] = (y_pred[y_test == 1] == y_test[y_test == 1]).sum()
    train_results.append(fold_results)
    
    print("Finished processing Split #{:.0f}".format(len(train_results)/2))

results = pd.DataFrame(train_results, columns = ['Samples Set','n_samples','n_frauds','n_flagged','true_positives'])
results = results.sort_values(['Samples Set'])

print("\nTraining Results:\n")
print(results)

print('\nTotal true positives: {}'.format(results[results['Samples Set'].str.contains('Test')==True].true_positives.sum()))
print('Total flagged: {}'.format(results[results['Samples Set'].str.contains('Test')==True].n_flagged.sum()))


print("\nTotal processing time: {:.2f} minutes".format((time.time()-start_time)/60))    
    

#%% ------------------------------------------------------------------------ #
# Scratch pad

file_name = root_dir + "transactions_all.csv"
transactions = pd.read_csv(file_name, parse_dates = ['date'], low_memory = False)  

trans_test = train_data.loc[test_index,:]

trans_flagged = trans_test[y_pred == 1]
trans_true    = trans_test[y_test == 1]

trans_true    = transactions[transactions.trans_id.isin(trans_true.trans_id.tolist())]
trans_flagged = transactions[transactions.trans_id.isin(trans_flagged.trans_id.tolist())]

fig = plt.figure(figsize=(20, 10))
ax = xgb.plot_importance(classifier,max_num_features=15)
fig.axes.append(ax)

fig.set_size_inches(18.5, 10.5, forward=True)
fig.tight_layout()
plt.show()

classifier.get_fscore()
classifier.get_score()














