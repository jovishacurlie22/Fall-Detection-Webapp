# ============================================================
# LSTM Fall Detection Training (FINAL - FRAME SEQUENCE VERSION)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 📁 Path to generated sequences
DATA_PATH = "/content/drive/MyDrive/fall_project/dataset_sequences"

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001


# ------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------
class FallDataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        mapping = {
            "nonfall": 0,
            "fall": 1
        }

        for folder, label in mapping.items():
            path = os.path.join(root_dir, folder)

            if not os.path.exists(path):
                print(f"⚠️ Missing folder: {path}")
                continue

            for file in os.listdir(path):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(path, file))
                    self.labels.append(label)

        print(f"\n✅ Total samples loaded: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("❌ No data found. Run generate_sequences.py first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx])

        # shape: (30, 99)
        seq = torch.tensor(seq, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return seq, label


# ------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------
class FallLSTM(nn.Module):

    def __init__(self, input_size=99, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out


# ------------------------------------------------------------
# Training Function
# ------------------------------------------------------------
def train():

    dataset = FallDataset(DATA_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FallLSTM().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Training started...\n")

    for epoch in range(EPOCHS):

        total_loss = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # 📁 Save model to Drive
    SAVE_PATH = "/content/drive/MyDrive/fall_project/fall_lstm.pt"

    torch.save(model.state_dict(), SAVE_PATH)

    print(f"\n✅ Model saved to: {SAVE_PATH}")


# ------------------------------------------------------------
if __name__ == "__main__":
    train()
