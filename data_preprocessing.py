# ------------------------------------------------------------------------- #
# Transaction Fraud Detection Prototype
# ------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

root_dir = "S:/can/bank/Departments/Risk/Risk Analytics/rezenro/Projects/transaction_fraud/data/"

#%% ----------------------------------------------------------------------- #

def preprocess_transactions(input_trans_fname, preproc_trans_fname):
    
    transactions = pd.read_csv(input_trans_fname, low_memory = False)
    
    file_name = root_dir + "fraud_transactions.csv"
    fraud_trans = pd.read_csv(file_name, low_memory = False)
    fraud_trans['ENTRY_DATE'] = fraud_trans['ENTRY_DATE'].str.upper()
    
    # We are not interested in transaction fees
    transactions = transactions[transactions.TRANSACTION_CODE.str.contains('FEE')==False].copy()
    
    transactions['trans_id'] = transactions[['INSTRUMENT_ID',
                                             'ENTRY_DATE',
                                             'TR_SEQUENCE',
                                             'TIME_SEQ']].astype(str).sum(axis=1)
    
    fraud_trans['trans_id'] = fraud_trans[['INSTRUMENT_ID',
                                           'ENTRY_DATE',
                                           'TR_SEQUENCE',
                                           'TIME_SEQ']].astype(str).sum(axis=1)
    
    fraud_ids = fraud_trans.trans_id.tolist()
    
    transactions['fraud_flag'] = transactions.trans_id.isin(fraud_ids).astype(int)
    
    transactions['date'] = pd.to_datetime(transactions.ENTRY_DATE + ' ' + transactions.TIME_SEQ)
    
    transactions['group'] = np.random.randint(1,21,len(transactions))
    
    fraud_codes = ['XMLMDC','UGWEBETSF','UGWEBBILL']
    transactions.loc[transactions.TRANSACTION_CODE.isin(fraud_codes)==False,'TRANSACTION_CODE'] = 'OTHERCODE'
    
    transactions.loc[(transactions.date.dt.hour >= 0 ) & (transactions.date.dt.hour <  6), 'timeslot'] = 'TIMESLOT_1'
    transactions.loc[(transactions.date.dt.hour >= 6 ) & (transactions.date.dt.hour < 12), 'timeslot'] = 'TIMESLOT_2'
    transactions.loc[(transactions.date.dt.hour >= 12) & (transactions.date.dt.hour < 18), 'timeslot'] = 'TIMESLOT_3'
    transactions.loc[(transactions.date.dt.hour >= 18) & (transactions.date.dt.hour < 24), 'timeslot'] = 'TIMESLOT_4'
    
    code_onehot = pd.get_dummies(transactions.TRANSACTION_CODE)
    teller_onehot = pd.get_dummies(transactions.TELLER)
    timeslot_onehot = pd.get_dummies(transactions.timeslot)
    
    transactions = pd.concat([transactions, code_onehot, teller_onehot, timeslot_onehot], axis = 1)
    
    transactions.rename(columns={'TRANSACTION_AMOUNT':'amount',
                                 'BALANCE': 'balance',
                                 'INSTRUMENT_ID':'account_id',
                                 'TRANSACTION_CODE':'code',
                                 'TELLER':'teller'}, inplace=True)
    
    transactions = transactions[['account_id','code','teller','date','timeslot',
                                 'trans_id','group','fraud_flag','amount','balance',
                                 'UGWEBBILL','UGWEBETSF','XMLMDC','OTHERCODE',
                                 'UGMLBWEB','UGMOBILE',
                                 'TIMESLOT_1','TIMESLOT_2','TIMESLOT_3','TIMESLOT_4']]
    
    transactions.to_csv(preproc_trans_fname, index = False)
    
    return

#%% ----------------------------------------------------------------------- #
# Preprocess transactions

input_trans_fname = root_dir + "retail_trans_0415_to_0528.csv"
preproc_trans_fname = root_dir + "trans_preproc.csv"
preprocess_transactions(input_trans_fname, preproc_trans_fname)

input_trans_fname = root_dir + "retail_trans_fraud_accounts.csv"
preproc_trans_fname = root_dir + "trans_preproc_fraud_accounts.csv"
preprocess_transactions(input_trans_fname, preproc_trans_fname)



#%% ----------------------------------------------------------------------- #
# Scratch pad

transactions = pd.read_csv(input_trans_fname, low_memory = False)

view = transactions[transactions.TRANSACTION_CODE == 'UGWEBETRE']
view.groupby(['TRANSACTION_DECRIPTION']).TRANSACTION_DECRIPTION.count()

transactions.groupby(['TRANSACTION_CODE']).TRANSACTION_CODE.count()

file_name = root_dir + "trans_preproc.csv"
trans_preproc = pd.read_csv(file_name, parse_dates = ['date'], low_memory = False)  
trans_preproc = trans_preproc[trans_preproc.fraud_flag == False]

file_name = root_dir + "trans_preproc_fraud_accounts.csv"
trans_preproc_fraud_accounts = pd.read_csv(file_name, parse_dates = ['date'], low_memory = False)  
trans_preproc_fraud_accounts = trans_preproc_fraud_accounts[trans_preproc_fraud_accounts.fraud_flag == True]

trans_preproc_all = pd.concat([trans_preproc, trans_preproc_fraud_accounts])

#trans_preproc_all.to_csv(root_dir + "trans_preproc_all.csv", index = False)


#%%






