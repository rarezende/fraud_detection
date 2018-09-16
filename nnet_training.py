# ------------------------------------------------------------------------- #
# Transaction Fraud Detection Prototype
# ------------------------------------------------------------------------- #

import os
os.environ['PYTHONHASHSEED'] = '42'
import numpy as np
np.random.seed(42)
import random as rn
rn.seed(42)

import time
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

startTime = time.time()

K.tf.set_random_seed(42)
tf_config = K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
K.set_session(K.tf.Session(graph=K.tf.get_default_graph(), config=tf_config))

root_dir = "C:/Users/rarez/Documents/Data Science/transaction_fraud/"

train_data = pd.read_csv(root_dir + "data/training.csv")

X = train_data.loc[:,'amount':].values
y = train_data.loc[:,'fraud_flag'].values
groups = train_data.loc[:, 'group'].values     

gkf = GroupKFold(n_splits=4)

# ---------------------------------------------------------------------------- #
# Definition of neural network model

def create_nnet(input_dim):
    
    lr = 0.0005
    dropout_rate = 0.30

    reg_lambda = list()
    reg_lambda.insert(0, 0.0003)
    reg_lambda.insert(1, 0.0003)
    reg_lambda.insert(2, 0.0001)
    reg_lambda.insert(3, 0.0001)
    
    nnet = Sequential()
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[0]), input_dim = input_dim))
    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[0])))
    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[1])))
    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[1])))
    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[2])))
    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1024, activation="relu", kernel_regularizer=l2(reg_lambda[3])))
    nnet.add(Dropout(dropout_rate))     
#    nnet.add(Dense(256, activation="relu", kernel_regularizer=l2(reg_lambda[3])))
#    nnet.add(Dropout(dropout_rate))     
#    nnet.add(Dense(256, activation="relu", kernel_regularizer=l2(reg_lambda[3])))
#    nnet.add(Dropout(dropout_rate))     
    nnet.add(Dense(1, activation="sigmoid"))
    nnet.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
    
    return nnet

# ---------------------------------------------------------------------------- #
# Callbacks

checkpointer = ModelCheckpoint(filepath = root_dir + "data/best_model.hdf5", save_best_only = True)

lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', 
                               factor = 0.5,
                               patience = 1, 
                               cooldown = 0,
                               min_lr = 1e-7,
                               verbose = True)

earlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0.0005, patience = 25)

# ---------------------------------------------------------------------------- #
# Model training

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

train_index, test_index = list(gkf.split(X_scaled, y, groups))[1]

X_train, y_train = X_scaled[train_index], y[train_index]  
X_test , y_test  = X_scaled[test_index] , y[test_index]

create_model = 1
epochs = 30

if create_model==1:
    model = create_nnet(X_train.shape[1])
else: 
    model = load_model(root_dir + "data/best_model.hdf5")


history = model.fit(X_train, y_train, 
                    batch_size = 128, 
                    epochs = epochs, 
                    callbacks = [lr_reducer, checkpointer, earlystopper], 
                    validation_data = (X_test, y_test),
                    verbose = True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.minorticks_on()
plt.ylim((0.0,0.05))
plt.grid(b=True, which='major', color='black', linestyle='--')
fileName = root_dir + "train_charts/" + "Loss_" + time.strftime("%m%d%H%M") + ".png"
plt.savefig(fileName)
plt.show()

y_true = y_test
y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

threshold = 0.025
y_pred[y_pred <  threshold] = 0
y_pred[y_pred >= threshold] = 1

print('\nTotal true positives: {}'.format((y_pred[y_test == 1] == y_test[y_test == 1]).sum()))
print('Total flagged: {}'.format(int(y_pred.sum())))

print("\nFinal learning rate: {:.8f}".format(K.get_value(model.optimizer.lr)))
print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))    


#%% ----------------------------------------------------------------------- #
# Model Performance

transactions = pd.read_csv(root_dir + "data/transactions.csv", parse_dates = ['date'], low_memory = False) 
trans_features = pd.read_csv(root_dir + "data/train.csv")
model = load_model(root_dir + "data/best_model.hdf5")

dt = pd.to_datetime('2018-05-03 00:00:00')
dt_plus1 = dt + pd.to_timedelta(24, unit='h')

trans_date = transactions[(transactions.date >= dt) & (transactions.date < dt_plus1)]

trans_feat_date = trans_features[trans_features.trans_id.isin(trans_date.trans_id.tolist())].copy()

X = scaler.transform(trans_feat_date.loc[:,'amount':].values)
X_rec = model.predict(X)

trans_feat_date['rec_mse'] = np.mean(np.square(X - X_rec), axis=1)

fraud_ids = trans_feat_date[trans_feat_date.rec_mse>threshold].trans_id.tolist()
model_flagged = trans_date[trans_date.trans_id.isin(fraud_ids) & (abs(trans_date.amount)>200)]

true_positives = model_flagged[model_flagged.fraud_flag == True]

view = trans_date[trans_date.fraud_flag == True]

len(trans_date)
len(model_flagged)
len(true_positives)

#%% ----------------------------------------------------------------------- #
# Scratch pad

view = train_data[train_data.fraud_flag == True]










