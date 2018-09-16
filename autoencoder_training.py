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
from keras.regularizers import l1
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

startTime = time.time()

K.tf.set_random_seed(42)
tf_config = K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
K.set_session(K.tf.Session(graph=K.tf.get_default_graph(), config=tf_config))

root_dir = "S:/can/bank/Departments/Risk/Risk Analytics/rezenro/Projects/transaction_fraud/"
train_data = pd.read_csv(root_dir + "data/training.csv")
fraud_transactions = train_data[train_data.fraud_flag==True]
train_data = train_data[train_data.fraud_flag==False]

def create_autoencoder(input_dim):
    
    lr = 0.001
    reg_lambda = 0.0
    dropout_rate = 0.1
    
    autoencoder = Sequential()
    #autoencoder.add(Dropout(dropout_rate, input_shape=(input_dim,)))     
    autoencoder.add(Dense(20, activation="relu", input_dim = input_dim, kernel_regularizer=l1(reg_lambda)))
    autoencoder.add(Dropout(dropout_rate))     
    autoencoder.add(Dense(6, activation="relu"))
    autoencoder.add(Dense(20, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="linear", kernel_regularizer=l1(reg_lambda)))
    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
    
    return autoencoder


checkpointer = ModelCheckpoint(filepath = root_dir + "data/best_model.hdf5", save_best_only = True)

lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', 
                               factor = 0.5,
                               patience = 10, 
                               cooldown = 0,
                               min_lr = 1e-6,
                               verbose = True)

earlystopper = EarlyStopping(monitor = 'val_loss', 
                             min_delta = 0.001, 
                             patience = 25)

X = train_data.loc[:,'amount':].values

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test = train_test_split(X_scaled, test_size=0.20, random_state=42)

create_model = 1
epochs = 100

if create_model==1:
    model = create_autoencoder(X_train.shape[1])
else: 
    model = load_model(root_dir + "data/best_model.hdf5")


history = model.fit(X_train, X_train, 
                    batch_size = 32, 
                    epochs = epochs, 
                    callbacks = [lr_reducer, checkpointer, earlystopper], 
                    validation_data = (X_test, X_test),
                    verbose = True)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.minorticks_on()
plt.ylim((0.01,0.35))
plt.grid(b=True, which='major', color='black', linestyle='--')
fileName = root_dir + "train_charts/" + "Loss_" + time.strftime("%m%d%H%M") + ".png"
plt.savefig(fileName)
plt.show()

y_true = X_test
y_pred = model.predict(X_test)
mse_normal = pd.DataFrame({'mse': np.mean(np.square(y_pred - y_true), axis=1)})

y_true = scaler.transform(fraud_transactions.loc[:,'amount':].values)
y_pred = model.predict(y_true)
mse_fraud = fraud_transactions.copy()
mse_fraud['mse'] = np.mean(np.square(y_pred - y_true), axis=1)

print("\nNormal transactions MSE")
print(mse_normal.mse.describe())
print("\nFraud transactions MSE")
print(mse_fraud.mse.describe())

quantile = 0.985
threshold = mse_normal.mse.quantile(quantile)
detection_rate = mse_fraud[mse_fraud.mse > threshold].mse.count()/len(mse_fraud)

print("\nPercentage of frauds detected with {:.1f}% threshold: {:.0f}%".format(100*quantile, 100*detection_rate))

print("Final learning rate: {:.5f}".format(K.get_value(model.optimizer.lr)))
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
view = mse_fraud[mse_fraud.mse > threshold]

len(trans_date)
len(model_flagged)
len(true_positives)

#%% ----------------------------------------------------------------------- #
# Scratch pad

view = train_data[train_data.fraud_flag == True]










