import torch
import torch.nn as nn

class MLPLogisticFL(nn.Module):
    def __init__(self, input_dim, hidden_dim, epochs, lr):
        super().__init__()
        self.epochs, self.lr = epochs, lr
        self.hidden = nn.Linear(input_dim, hidden_dim)  # 중간 레이어 (32 크기)
        self.output = nn.Linear(hidden_dim, 2)  # Logistic Regression 출력 레이어
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.criterion = nn.CrossEntropyLoss()
        self.t = 0.5
        self.T = 2

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.hidden(x)  # 중간 레이어 통과
        x = self.relu(x)  # ReLU 활성화 함수 적용
        x = self.output(x)  # 출력 레이어 통과
        return x  # Logistic Regression이므로 sigmoid 사용
    
    def fit(self, X_train, X_train2, y_train):
        X_train = torch.Tensor(X_train)
        X_train2 = torch.Tensor(X_train2)
        y_train = torch.LongTensor(y_train).view(-1)
          # BCEWithLogitsLoss 사용
        for epoch in range(self.epochs):
            self.train()

            outputs = self.forward(X_train)
            loss = self.criterion(outputs, y_train)

            with torch.no_grad():
                pseudo_probs = self.forward(X_train2) 
                pseudo_probs /= self.T
                pseudo_probs = torch.softmax(pseudo_probs, dim=1)
                confidence, pseudo_label = torch.max(pseudo_probs, dim=1)
                mask = confidence >= self.t
                selected_X = X_train2[mask]
                selected_y = pseudo_label[mask]


            outputs2 = self.forward(selected_X)
            loss2 = self.criterion(outputs2, selected_y)

            loss += loss2
            loss.backward()  # 기울기 계산
            with torch.no_grad():  # 옵티마이저 없이 수동으로 가중치 갱신
                for param in self.parameters():
                    param -= self.lr * param.grad  # learning rate 적용 (예: 0.001)
            self.zero_grad()
            