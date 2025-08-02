import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
import json, os, warnings, torch, time
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def select_feature(df, cols, k) :
    X = df.drop(columns=cols)
    y = df.iloc[:, -1]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    df_selected = pd.concat([df[selected_features], df[cols]], axis=1)
    return df_selected

def cluster_feature(df, cols):
    n_clusters = 2
    X = df.drop(columns=cols)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X.T)
    cluster_labels = kmeans.labels_
    idxs = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]
    return idxs

def cotrain(type, x_bases, y_base, x_preds, x_tests, y_test, models, inner_epochs, inner_lr, inner_rounds, t, seed, num):
    x_tests = [torch.as_tensor(x_test, dtype=torch.float32, device=device) for x_test in x_tests]
    x_preds = [torch.as_tensor(x_pred, dtype=torch.float32, device=device) for x_pred in x_preds]
    x_bases = [torch.as_tensor(x_base, dtype=torch.float32, device=device) for x_base in x_bases]
    y_base = torch.as_tensor(y_base, dtype=torch.float32, device=device)
    pos = f'./model_datas/{type}/{seed}/'
    if not os.path.exists(pos): os.makedirs(pos)
    pos = f'./model_saves/{type}/{seed}/'
    if not os.path.exists(pos): os.makedirs(pos)
    pos = f'./model_datas/{type}/{seed}/{num}_{x_preds[0].shape[0]}.json'
    if not os.path.exists(pos) :
        d = {}
        d['x_preds'] = [x_preds[0].tolist(),x_preds[1].tolist()]
        d['x_tests'] = [x_tests[0].tolist(),x_tests[1].tolist()]
        d['y_test'] = y_test.tolist()
        with open(pos,'w') as f :
            json.dump(d, f)
    for i in range(2) :
        pos = f'./model_saves/{type}/{seed}/{num}_{i}.json'
        if os.path.exists(pos) :
            with open(pos, 'r') as f:
                weights = json.load(f)
                w1 = np.array(weights['w1'])
                w2 = np.array(weights['w2'])
                models[i].hidden.weight = nn.Parameter(torch.tensor(w1[:-1,:].T,dtype=torch.float32 ,device=device))
                models[i].hidden.bias = nn.Parameter(torch.tensor(w1[-1,:],dtype=torch.float32 ,device=device))
                models[i].output.weight = nn.Parameter(torch.tensor(w2[:-1,:].T,dtype=torch.float32 ,device=device))
                models[i].output.bias = nn.Parameter(torch.tensor(w2[-1,:],dtype=torch.float32 ,device=device))
        else : 
            models[i].fit(x_bases[i], y_base)
            torch.set_printoptions(precision=10)
            w1 = models[i].hidden.weight.T.cpu().detach().numpy()
            b1 = models[i].hidden.bias.view(1,-1).cpu().detach().numpy()
            w2 = models[i].output.weight.T.cpu().detach().numpy()
            b2 = models[i].output.bias.view(1,-1).cpu().detach().numpy()
            w1 = np.concatenate([w1,b1],axis=0).tolist()
            w2 = np.concatenate([w2,b2],axis=0).tolist()
            d = {'w1':w1,'w2':w2}
            with open(pos,'w') as f :
                json.dump(d, f)
    
    
    def make_acc() :
        probas = [model.forward(x_test).cpu().detach().numpy() for model, x_test in zip(models, x_tests)]
        pred = np.where(np.sum(probas,axis=0) >= 0.5*len(models), 1.0, 0.0)
        return accuracy_score(y_test, pred)
    
    def make_ret() :
        st = time.time()
        probas = [model.forward(x_test).cpu().detach().numpy() for model, x_test in zip(models, x_tests)]
        ed = time.time()
        pred = np.where(np.sum(probas,axis=0) >= 0.5*len(models), 1.0, 0.0)
        return accuracy_score(y_test, pred),f1_score(y_test, pred), ed-st
    
    for model in models : 
        model.epochs = inner_epochs
        model.lr = inner_lr

    accs = []
    train_time = 0
    accs.append(make_acc())
    for round in range(1,inner_rounds+1):
        st = time.time()
        probas = [model.forward(x_pred).cpu().detach().numpy() for model, x_pred in zip(models, x_preds)]

        labels = [np.where(proba>=0.5,1,0) for proba in probas]
        labels[0],labels[1] = labels[1],labels[0]

        masks = [((proba >= t) | (proba <= 1-t)).flatten() for proba in probas]
        masks[0],masks[1] = masks[1],masks[0] 
        
        x_trains = [x_pred[mask] for mask,x_pred in zip(masks,x_preds)]
        y_trains = [label[mask] for mask,label in zip(masks,labels)]
        
        x_trains = [torch.as_tensor(x_train, dtype=torch.float32, device=device) for x_train in x_trains]
        y_trains = [torch.as_tensor(y_train, dtype=torch.float32, device=device) for y_train in y_trains]

        for i in range(len(models)):
            models[i].fit(x_trains[i], y_trains[i])
        ed = time.time()
        train_time += ed-st
        if round % 5 == 0 : 
            accs.append(make_acc())
    
    acc, f1, inf_time = make_ret()

    return accs, acc,f1, train_time, inf_time


def show_figure(data) :
    plt.plot(data)