import torch, warnings, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import LeaveOneGroupOut
from config import cols, labels
from train_functions.MLPLogistic import MLPLogistic
from train_functions.fncs import select_feature

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def general_model_train(type, epochs, lr, hidden, reduce, personal, seed) :
    df = pd.read_csv(f'./datas/{type}.csv',index_col=0)
    if reduce : df = select_feature(df, cols[type], reduce)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logo = LeaveOneGroupOut()
    groups = df['pnum']
    accs,f1s,train_times,inf_times = [],[],[],[]
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
        
        x_test, y_test = x_test[personal:],y_test[personal:]
        
        x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)

        x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)

        model = MLPLogistic(x_train.shape[1], hidden, epochs, lr, seed)     
        st = time.time()
        model.fit(x_train, y_train)
        ed = time.time()
        train_times.append(ed-st)

        st = time.time()
        proba = model.forward(torch.Tensor(x_test))
        ed = time.time()
        inf_times.append(ed-st)

        pred = np.where(np.array(proba.cpu().detach())>=0.5,1,0)
        accs.append(accuracy_score(y_test,pred))
        f1s.append(f1_score(y_test,pred))
        bar = '[' + '=' * int((i+1) * 31 / group_count) + ' ' * (31 - int((i+1) * 31 / group_count)) + ']'
        percent = ((i+1) / group_count) * 100
        sys.stdout.write(f'\rProgress: {bar} {percent:.2f}% ({(i+1)}/{group_count})')
        sys.stdout.flush()
    print('')
    return accs,f1s,train_times,inf_times
