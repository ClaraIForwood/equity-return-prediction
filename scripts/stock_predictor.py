"""
Stock Market Next-Day Closing Price Prediction (AAPL) using LSTM (PyTorch)

Pipeline:
1) Download data (yfinance)
2) Clean + keep Close
3) Scale data (MinMax)
4) Create sliding-window sequences
5) Train/Test split (time-based)
6) PyTorch Dataset/DataLoader
7) Build LSTM model
8) Train
9) Predict
10) Inverse transform
11) Evaluate (RMSE/MAE + directional accuracy)
12) Plot results
"""

# ------------------------
# 1) Imports
# ------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ------------------------
# 2) Download data
# ------------------------
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2021-01-01"

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    raise ValueError("No data downloaded. Check ticker or internet connection.")

print(data.head())


# ------------------------
# 3) Clean + keep Close
# ------------------------
data = data[["Close"]].dropna()
close_prices = data.values  # shape: (N, 1)

print("\nClose shape:", close_prices.shape)
print("Missing values:\n", data.isnull().sum())


# ------------------------
# 4) Train/Test split FIRST (avoid leakage when scaling)
# ------------------------
lookback = 60
split_ratio = 0.8

# We'll split on the raw series index, then scale using TRAIN only
split_index = int(len(close_prices) * split_ratio)

train_prices = close_prices[:split_index]
test_prices = close_prices[split_index - lookback:]  # include lookback overlap for sequences

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_prices)
test_scaled = scaler.transform(test_prices)


# ------------------------
# 5) Create sequences (sliding windows)
# ------------------------
def make_sequences(scaled_array: np.ndarray, lookback: int):
    """
    scaled_array: shape (N, 1)
    returns:
      X: (samples, lookback, 1)
      y: (samples, 1)
    """
    X, y = [], []
    for i in range(lookback, len(scaled_array)):
        X.append(scaled_array[i - lookback:i])
        y.append(scaled_array[i])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, lookback)
X_test, y_test = make_sequences(test_scaled, lookback)

print("\nX_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


# ------------------------
# 6) PyTorch tensors + DataLoaders
# ------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

batch_size = 32

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)


# ------------------------
# 7) Build LSTM model
# ------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)   # (batch, seq_len, hidden)
        out = out[:, -1, :]     # last timestep (batch, hidden)
        out = self.fc(out)      # (batch, 1)
        return out


model = StockLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ------------------------
# 8) Train loop
# ------------------------
epochs = 30
train_losses, test_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    running_train_loss = 0.0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * Xb.size(0)

    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Evaluate (test set as validation)
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            running_test_loss += loss.item() * Xb.size(0)

    avg_test_loss = running_test_loss / len(test_loader.dataset)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")


# ------------------------
# 9) Plot loss curves
# ------------------------
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over Epochs (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ------------------------
# 10) Predict on test set
# ------------------------
model.eval()
preds_list = []

with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb)
        preds_list.append(preds.cpu().numpy())

preds_scaled = np.vstack(preds_list)       # (test_samples, 1)
y_test_scaled = y_test                     # (test_samples, 1)


# ------------------------
# 11) Inverse transform (back to real prices)
# ------------------------
preds_real = scaler.inverse_transform(preds_scaled)
y_test_real = scaler.inverse_transform(y_test_scaled)


# ------------------------
# 12) Metrics + plot predictions
# ------------------------
rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
mae = mean_absolute_error(y_test_real, preds_real)

print("\n--- Test Metrics ---")
print("RMSE:", rmse)
print("MAE :", mae)

# Directional accuracy (optional)
true_diff = y_test_real[1:] - y_test_real[:-1]
pred_diff = preds_real[1:] - preds_real[:-1]
directional_acc = np.mean((true_diff > 0) == (pred_diff > 0))
print("Directional Accuracy:", directional_acc)

# Plot predicted vs true
plt.figure(figsize=(12, 5))
plt.plot(y_test_real, label="True Close")
plt.plot(preds_real, label="Predicted Close")
plt.title(f"{ticker} Next-Day Close Prediction (Test Set)")
plt.xlabel("Time (test samples)")
plt.ylabel("Price")
plt.legend()
plt.show()
