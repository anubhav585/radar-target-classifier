import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb  
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import RadarLSTM, RadarCNN1D

def train_lstm():
    # 1. Initialize a W&B run
    wandb.init(
        project="radar-target-classifier",
        config={  "learning_rate": 0.001,
            "epochs": 10, "batch_size": 32,"hidden_dim": 64,"model_type": "LSTM"} )
    config = wandb.config

    print("--- Loading Data for W&B LSTM Tracking ---")
    df = pd.read_csv("data/raw/astra_dataset.csv")
    feature_cols = [str(i) for i in range(300)]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Reshape into 3D tensors: (samples, 100 timesteps, 3 features)
    X = X.reshape(-1, 100, 3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale data per timestep/channel if needed, or convert directly to tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model, loss, and optimizer
    # model = RadarLSTM(input_dim=3, hidden_dim=config.hidden_dim,num_classes=4)
    model = RadarCNN1D(input_dim=3, hidden_dim=config.hidden_dim,num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 2. Watch model gradients and architecture with W&B
    wandb.watch(model, criterion, log="all", log_freq=10)

    print("--- Training LSTM with W&B Tracking ---")
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        # Evaluate accuracy on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        test_accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # 3. Log metrics to W&B dashboard
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "test_accuracy": test_accuracy
        })

    # ==========================================
    # INFERENCE LATENCY BENCHMARK
    # ==========================================
    model.eval()
    sample_inputs, _ = next(iter(test_loader)) # Grab one test batch
    
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(sample_inputs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    print(f"\n--- INFERENCE BENCHMARK ---")
    print(f"Inference Latency for batch of {sample_inputs.size(0)}: {latency_ms:.2f} ms")
    
    # Log final latency to W&B
    wandb.log({"inference_latency_ms": latency_ms})

    wandb.finish()


if __name__ == "__main__":
    train_lstm()