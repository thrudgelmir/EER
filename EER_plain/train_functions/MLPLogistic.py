import torch
import torch.nn as nn

class MLPLogistic(nn.Module):
    def __init__(self, input_dim, hidden_dim, epochs, lr, seed):
        torch.manual_seed(seed)
        super().__init__()
        self.epochs, self.lr = epochs, lr
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return torch.sigmoid(x)
    
    def fit(self, X_train, y_train):
        criterion = nn.BCELoss()
        for epoch in range(self.epochs):
            self.train()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            with torch.no_grad():
                for param in self.parameters():
                    param -= self.lr * param.grad
            self.zero_grad()
