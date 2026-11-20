import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. TRADITIONAL MACHINE LEARNING MODELS
# ==========================================

def get_svm_model(C=1.0, kernel='rbf'):
    return SVC(C=C, kernel=kernel, probability=True, random_state=42)

def get_random_forest_model(n_estimators=100, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


# ==========================================
# 2. DEEP LEARNING SEQUENCE MODELS (PyTorch)
# ==========================================

class RadarRNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=4, num_layers=1):
        super(RadarRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, h_n = self.rnn(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


class RadarLSTM(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, num_classes=4, num_layers=1):
        super(RadarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


class RadarCNN1D(nn.Module):
    def __init__(self, input_dim=3, num_classes=4, hidden_dim=64):
        super(RadarCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len=100, input_dim=3) -> Transpose to (batch_size, 3, 100)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1) # Shape: (batch_size, hidden_dim)
        out = self.fc(x)
        return out