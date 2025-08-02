from train_functions.fncs import select_feature, cluster_feature, cotrain
from config import cols, labels
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import LeaveOneGroupOut
from train_functions.MLPLogistic import MLPLogistic
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def personal_model_train(type, epochs, lr, hidden, reduce, personal, inner_epochs, inner_rounds, inner_lr, t, seed, inf_cnt=-1) :
    df = pd.read_csv(f'./datas/{type}.csv',index_col=0)
    if reduce : df = select_feature(df, cols[type], reduce)
    idxs = cluster_feature(df, cols[type])
    torch.manual_seed(seed)
    np.random.seed(seed)
    logo = LeaveOneGroupOut()
    groups = df['pnum']
    mid_accs,accs,f1s,train_times,inf_times = [],[],[],[],[]
    unique_groups = np.unique(groups)
    group_count = len(unique_groups)
    for i,(train_idx, test_idx) in enumerate(logo.split(df, df[labels[type]], groups)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_data = df.drop(columns=cols[type])
        y_data = df[labels[type]]
        x_train, x_test = np.array(x_data.iloc[train_idx]), np.array(x_data.iloc[test_idx])
        y_train, y_test = np.array(y_data.iloc[train_idx]).reshape(-1,1), np.array(y_data.iloc[test_idx]).reshape(-1,1)
        
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]

        indices = np.arange(x_test.shape[0])
        np.random.shuffle(indices)
        x_test, y_test = x_test[indices], y_test[indices]

        x_pred, y_pred = x_test[:personal],y_test[:personal]
        x_test, y_test = x_test[personal:],y_test[personal:]
        x_test, y_test = x_test[:inf_cnt],y_test[:inf_cnt]
        
        x_trains = [x_train[:,idx] for idx in idxs]
        x_tests = [x_test[:,idx] for idx in idxs]
        x_preds = [x_pred[:,idx] for idx in idxs]

        models = [MLPLogistic(x_trains[j].shape[1], hidden, epochs, lr, seed) for j in range(2)]
        
        mid_acc,acc,f1,train_time,inf_time = cotrain(type, x_trains, y_train, x_preds ,x_tests, y_test, models, inner_epochs, inner_lr, inner_rounds, t, seed, i)
        
        mid_accs.append(mid_acc)
        accs.append(acc)
        f1s.append(f1)
        train_times.append(train_time)
        inf_times.append(inf_time)

        bar = '[' + '=' * int((i+1) * 31 / group_count) + ' ' * (31 - int((i+1) * 31 / group_count)) + ']'
        percent = ((i+1) / group_count) * 100
        sys.stdout.write(f'\rProgress: {bar} {percent:.2f}% ({(i+1)}/{group_count})')
        sys.stdout.flush()
    print('')
    return mid_accs,accs,f1s,train_times,inf_times