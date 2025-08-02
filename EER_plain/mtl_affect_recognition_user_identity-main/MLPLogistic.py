import torch
import torch.nn as nn

class MLPLogistic(nn.Module):
    def __init__(self, input_dim, hidden_dim, epochs, lr):
        super().__init__()
        self.epochs, self.lr = epochs, lr
        self.hidden = nn.Linear(input_dim, hidden_dim)  # 중간 레이어 (32 크기)
        self.output = nn.Linear(hidden_dim, 1)  # Logistic Regression 출력 레이어
        self.relu = nn.ReLU()  # ReLU 활성화 함수

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.hidden(x)  # 중간 레이어 통과
        x = self.relu(x)  # ReLU 활성화 함수 적용
        x = self.output(x)  # 출력 레이어 통과
        return torch.sigmoid(x)  # Logistic Regression이므로 sigmoid 사용
    
    def fit(self, X_train, y_train):
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train).view(-1,1)  # torch.Tensor()로 2D tensor로 변환
        criterion = nn.BCELoss()  # BCEWithLogitsLoss 사용
        ret = []
        for epoch in range(self.epochs):
            self.train()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()  # 기울기 계산
            with torch.no_grad():  # 옵티마이저 없이 수동으로 가중치 갱신
                for param in self.parameters():
                    param -= self.lr * param.grad  # learning rate 적용 (예: 0.001)
            self.zero_grad()
            