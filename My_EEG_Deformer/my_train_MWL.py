import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# les paramètres spécifiques au dataset MWL
SEED = 2023
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.25  # comme précisé dans l'article pour MWL
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './data_processed/data_eeg_MWL_MW/'  # Dossier contenant les fichiers subX.pkl pour MWL
SAVE_PATH = './save/logs_MWL_Deformer/'  # Dossier de sauvegarde des résultats MWL

print(f"GPU disponible : {torch.cuda.is_available()} — Utilisé : {DEVICE}")

# on met tout à la même seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataset utilisé pour entraîner et tester le modèle
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        y_array = np.array(y)
        self.y = torch.tensor(y_array.reshape(-1), dtype=torch.long)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim == 3 and x.shape[0] != 1:
            x = x[:, 0, :]
        x = x.unsqueeze(0)
        y = self.y[idx]
        return x, y


# encodeur convolutionnel pour extraire les features de bas niveau
class ShallowEncoder(nn.Module):
    def __init__(self, in_channels, dim, kernel_length):
        super().__init__()
        self.temporal = nn.Conv2d(1, dim, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2))
        self.spatial = nn.Conv2d(dim, dim, kernel_size=(in_channels, 1))
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.activation(x)

# branche coarse avec attention
class CoarseBranch(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.attn(x)

# branche fine avec convolution locale
class FineBranch(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.conv(x))

# bloc HCT qui combine les deux branches
class HCTBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.coarse = CoarseBranch(dim, heads, dropout)
        self.fine = FineBranch(dim, dropout)

    def forward(self, x):
        x_coarse = self.coarse(x)
        x_fine = self.fine(x.transpose(1, 2)).transpose(1, 2)
        return x_coarse + x_fine, x_fine

# module DIP pour combiner les features intermédiaires
class DIP(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size=1) for _ in range(num_blocks)])
        self.act = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, fine_features):
        outs = []
        for i, f in enumerate(fine_features):
            out = self.pool(self.act(self.proj[i](f)))
            outs.append(out.squeeze(-1))
        return torch.stack(outs, dim=1).sum(dim=1)

# modèle complet EEG-Deformer
class EEGDeformer(nn.Module):
    def __init__(self, in_channels=29, seq_len=384, num_classes=3, dropout=0.5):
        super().__init__()
        dim = 64
        num_blocks = 4

        self.encoder = ShallowEncoder(in_channels, dim, kernel_length=64)
        self.proj = nn.Conv2d(dim, dim, (1, 1))
        self.hct_blocks = nn.ModuleList([HCTBlock(dim, heads=4, dropout=dropout) for _ in range(num_blocks)])
        self.dip = DIP(dim, num_blocks)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        x = x.squeeze(2).transpose(1, 2)

        fine_features = []
        for block in self.hct_blocks:
            x, fine = block(x)
            fine_features.append(fine.transpose(1, 2))
        x = self.dip(fine_features)
        return self.classifier(x)

# fonction pour charger tous les sujets depuis les fichiers .pkl
def load_all_subjects(directory):
    data = {}
    for file in os.listdir(directory):
        if file.endswith('.pkl') and file.startswith('sub'):
            subject_id = int(file.replace('sub', '').replace('.pkl', ''))
            with open(os.path.join(directory, file), 'rb') as f:
                content = pickle.load(f)
                data[subject_id] = (content['data'], content['label'])
    return data

# fonction principale pour entraîner avec la stratégie Leave-One-Subject-Out
def train():
    all_data = load_all_subjects(DATA_DIR)
    os.makedirs(SAVE_PATH, exist_ok=True)

    for test_id in sorted(all_data):
        print(f"Entraînement avec sujet {test_id} mis de côté")
        X_train, y_train, X_test, y_test = [], [], [], []

        for sid, (X, y) in all_data.items():
            if sid == test_id:
                X_test, y_test = X, y
            else:
                X_train.extend(X)
                y_train.extend(y)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

        train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE)

        model = EEGDeformer(in_channels=29, seq_len=384, num_classes=3, dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            print(f"Epoch {epoch+1}/{EPOCHS}")
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = torch.argmax(model(x_batch), dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total
        print(f"Accuracy sujet {test_id} : {acc:.4f}")

if __name__ == '__main__':
    train()
