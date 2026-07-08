import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os

class Model(nn.Module):
    def __init__(self, in_features=57, h1=8, h2=9, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def main():
    data_path = 'data/spambase_csv.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df['class'].values
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train_raw, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    
    model = Model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 500
    model.train()
    for epoch in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch} | Train Loss: {loss.item():.5f}')
            
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth successfully.")

if __name__ == '__main__':
    main()
