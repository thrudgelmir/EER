import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1D CNN 정의
class CNN1D(nn.Module):
    def __init__(self, input_length=143):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.AvgPool1d(2)
        reduced_length = input_length // 2 // 2 // 2
        self.fc = nn.Linear(reduced_length * 128, 2)
        self.lrn = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.lrn(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.lrn(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.softmax(self.fc(x), dim=1)

# softmax 10개 단위 concat
def create_concat_features(cnn_outputs, window=10):
    return np.array([
        cnn_outputs[i*window:(i+1)*window].flatten()
        for i in range(len(cnn_outputs) // window)
    ])

# voting 기반 라벨 생성
def create_voted_labels(y, window=10):
    y = y.reshape(-1)  # 또는 y = y.ravel()
    return np.array([
        np.bincount(y[i*window:(i+1)*window]).argmax()
        for i in range(len(y) // window)
    ])
# 전체 학습 및 평가 파이프라인
def train_cnn_svm_pipeline(x_train, y_train, x_test, y_test, epoch=10):
    device = 'cpu'
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNN1D(input_length=x_train.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # CNN 학습
    model.train()
    for _ in range(epoch):
        optimizer.zero_grad()
        pred = model(x_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        cnn_train_out = model(x_train_tensor).cpu().numpy()
        cnn_test_out = model(torch.tensor(x_test, dtype=torch.float32).to(device)).cpu().numpy()

    # SVM 학습을 위한 재구성
    x_train_concat = create_concat_features(cnn_train_out)
    y_train_concat = create_voted_labels(y_train)
    x_test_concat = create_concat_features(cnn_test_out)
    y_test_concat = create_voted_labels(y_test)

    # SVM 학습 및 평가
    clf = SVC(kernel='linear')
    clf.fit(x_train_concat, y_train_concat)
    y_pred = clf.predict(x_test_concat)
    acc = accuracy_score(y_test_concat, y_pred)

    print(f"SVM Accuracy (CNN features + 10-window concat): {acc:.4f}")
    return acc
