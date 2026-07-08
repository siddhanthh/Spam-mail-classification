import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle

class Model(nn.Module):
    def __init__(self, in_features=57, h1=32, h2=16, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)
    
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
    
    # Evaluate model accuracy on test set
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred_class = (y_test_pred >= 0.5).float()
        
        correct = (y_test_pred_class == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f"\n--- Model Evaluation ---")
        print(f"Test Set Accuracy: {accuracy:.2%}")
        
        # Calculate true positives, false positives, false negatives
        tp = ((y_test_pred_class == 1) & (y_test == 1)).sum().item()
        fp = ((y_test_pred_class == 1) & (y_test == 0)).sum().item()
        fn = ((y_test_pred_class == 0) & (y_test == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-Score: {f1:.2%}")

if __name__ == '__main__':
    main()
