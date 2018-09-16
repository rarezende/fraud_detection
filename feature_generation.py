# ------------------------------------------------------------------------- #
# Transaction Fraud Detection Prototype
# ------------------------------------------------------------------------- #

import time
import pandas as pd
import multiprocessing    

def calculate_aggregate(past_trans, cur_trans, period, agg_filter, feat_values, feat_list):
    
    field, value = agg_filter
    
    if field == '':
        name_suffix = str(period) + 'H'
        trans_agg = past_trans[past_trans.date >= (cur_trans.date - pd.to_timedelta(period, unit='h'))]
    else: 
        name_suffix = field + '_' +  value + '_' + str(period) + 'H'
        if value == 'SAME':
            trans_agg = past_trans[(past_trans[field] == cur_trans[field]) &
                                   (past_trans.date >= (cur_trans.date - pd.to_timedelta(period, unit='h')))]
        else:
            trans_agg = past_trans[(past_trans[field] == value) &
                                   (past_trans.date >= (cur_trans.date - pd.to_timedelta(period, unit='h')))]
    
    feat_name = 'number_' + name_suffix
    feat_list.append(feat_name)
    feat_values[feat_name] = len(trans_agg)
    
    feat_name = 'amount_' + name_suffix
    feat_list.append(feat_name)
    feat_values[feat_name] = trans_agg.amount.abs().sum()

    return feat_values, feat_list
    

def generate_features(transactions, feat_sets, trans_ids):
    
    past_trans_period = pd.to_timedelta(feat_sets[-1]['period'], unit='h')
    
    features = list()
    for trans_id in trans_ids:
        cur_trans  = transactions[transactions.trans_id == trans_id].squeeze()
        past_trans = transactions[(transactions.account_id == cur_trans.account_id) & 
                                  (transactions.trans_id   != cur_trans.trans_id)   & 
                                  (transactions.date <= cur_trans.date)             &
                                  (transactions.date >= (cur_trans.date - past_trans_period))].copy()
    
        feat_values = dict()
        feat_list = transactions.columns.tolist()
        feat_list = feat_list[feat_list.index('trans_id'):]
        
        for feat in feat_list:
            feat_values[feat] = cur_trans[feat]
        
        for feat_set in feat_sets:
            for agg_filter in feat_set['agg_filters']:
                feat_values, feat_list = calculate_aggregate(past_trans, 
                                                             cur_trans, 
                                                             feat_set['period'], 
                                                             agg_filter, 
                                                             feat_values, 
                                                             feat_list)
        
        features.append(feat_values)

    if 'feat_list' in locals():
        features_df = pd.DataFrame(features, columns = feat_list)
    else: 
        features_df = pd.DataFrame()
        
    print("{} finished processing {} transactions".format(multiprocessing.current_process().name, 
                                                          len(features_df)), 
                                                          flush=True)
        
    return features_df
    
    
def process_transactions(transactions, feat_sets, start_date):
    
    # Make sure that we have the minimum period for the aggregation of past transactions
    cutoff_date = transactions.date.min() + pd.to_timedelta(feat_sets[-1]['period'], unit='h')
    cutoff_date = max(cutoff_date, start_date)
    
    trans_ids = transactions[transactions.date > cutoff_date].trans_id.tolist()
    
    print("Processing {} transactions:\n".format(len(trans_ids)))    

    # Adjust the load on the CPU
    num_processes = multiprocessing.cpu_count() - 1
    
    num_batches = 4*num_processes
    batch_size = int(len(trans_ids)/num_batches)
    
    arg_list = list()
    for j in range(0, num_batches):
        arg_list.append([transactions.copy(), feat_sets, trans_ids[j*batch_size:(j+1)*batch_size]])
        
    arg_list.append([transactions.copy(), feat_sets, trans_ids[num_batches*batch_size:]])
    
    # Map
    pool = multiprocessing.Pool(processes = num_processes)
    result = pool.starmap_async(generate_features, arg_list)
    # Reduce
    train_data = pd.concat(result.get())
    pool.close()
    pool.join()
    
    return train_data

#%% ------------------------------------------------------------------------ #

def create_training_data():
    
    startTime = time.time()

    root_dir = "S:/can/bank/Departments/Risk/Risk Analytics/rezenro/Projects/transaction_fraud/data/"
    
    feat_sets = list()
    feat_sets.append({'period': 0.001, 'agg_filters':[('',''),('code','SAME')]}) # 3.6 seconds
    feat_sets.append({'period': 0.001, 'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 0.01,  'agg_filters':[('',''),('code','SAME')]}) # 36 seconds
    feat_sets.append({'period': 0.01,  'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 0.05,  'agg_filters':[('',''),('code','SAME')]}) # 3 minutes
    feat_sets.append({'period': 0.05,  'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 0.1,   'agg_filters':[('',''),('code','SAME')]}) # 6 minutes
    feat_sets.append({'period': 0.1,   'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 0.5,   'agg_filters':[('',''),('code','SAME')]}) # 30 minutes
    feat_sets.append({'period': 0.5,   'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 1,     'agg_filters':[('',''),('code','SAME')]}) 
    feat_sets.append({'period': 1,     'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 6,     'agg_filters':[('',''),('code','SAME')]}) 
    feat_sets.append({'period': 6,     'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 12,    'agg_filters':[('',''),('code','SAME')]}) 
    feat_sets.append({'period': 12,    'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 24,    'agg_filters':[('',''),('code','SAME')]}) 
    feat_sets.append({'period': 24,    'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 48,    'agg_filters':[('',''),('code','SAME'),('timeslot','SAME')]}) 
    feat_sets.append({'period': 48,    'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 72,    'agg_filters':[('',''),('code','SAME'),('timeslot','SAME')]}) 
    feat_sets.append({'period': 72,    'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 168,   'agg_filters':[('',''),('code','SAME'),('timeslot','SAME'),('teller','SAME')]}) # 1 Week
    feat_sets.append({'period': 168,   'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 336,   'agg_filters':[('',''),('code','SAME'),('timeslot','SAME'),('teller','SAME')]}) # 2 Weeks
    feat_sets.append({'period': 336,   'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    feat_sets.append({'period': 504,   'agg_filters':[('',''),('code','SAME'),('timeslot','SAME'),('teller','SAME')]}) # 3 Weeks
    feat_sets.append({'period': 504,   'agg_filters':[('code','XMLMDC'),('code','UGWEBETSF'),('code','UGWEBBILL')]})
    
    file_name = "trans_preproc.csv"
    print("\nProcessing file {}".format(file_name))
    transactions = pd.read_csv(root_dir + file_name, parse_dates = ['date'], low_memory = False)  
    train_data_normal = process_transactions(transactions, feat_sets, pd.to_datetime('2018-05-21'))

    file_name = "trans_preproc_fraud_accounts.csv"
    print("\nProcessing file {}".format(file_name))
    transactions = pd.read_csv(root_dir + file_name, parse_dates = ['date'], low_memory = False)  
    train_data_fraud = process_transactions(transactions, feat_sets, pd.to_datetime('2017-05-28'))
    
    file_name = root_dir + "fraud_transactions.csv"
    fraud_trans = pd.read_csv(file_name, low_memory = False)
    fraud_trans['ENTRY_DATE'] = fraud_trans['ENTRY_DATE'].str.upper()
    fraud_trans['trans_id'] = fraud_trans[['INSTRUMENT_ID','ENTRY_DATE','TR_SEQUENCE','TIME_SEQ']].astype(str).sum(axis=1)
    fraud_trans.rename(columns={'GROUP':'group'}, inplace=True)
    
    fraud_trans_groups = fraud_trans[['trans_id','group']]
    
    train_data_fraud = train_data_fraud.drop(['group'], axis = 1)
    
    train_data_fraud = pd.merge(fraud_trans_groups, train_data_fraud, how = 'inner', on = 'trans_id')
    
    train_data_normal = train_data_normal[train_data_normal.fraud_flag==False]

    training = pd.concat([train_data_normal, train_data_fraud])
    training = training.sample(frac=1, random_state = 42).copy()
    
    training.to_csv(root_dir + "training_new.csv", index = False)

    print("\nTotal processing time: {:.2f} minutes".format((time.time()-startTime)/60))    
    
    return


# -------------------------------------------------------------------------------------- #
# Main module function
# -------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    create_training_data()        
