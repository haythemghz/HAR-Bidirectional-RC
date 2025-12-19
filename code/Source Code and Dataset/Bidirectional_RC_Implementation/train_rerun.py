
"""
Rerun Training Script for 10 Epochs + Visualization
Uses Advanced Readout (Maxout/KAF)
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from model import BidirectionalRC
from data_utils import load_skeleton_data, normalize_sequences, prepare_data_for_training, compute_sequence_lengths, create_dummy_data

# ==========================================
# Re-using classes from train.py
# ==========================================

class SkeletonDataset(Dataset):
    """Dataset class for skeleton sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return list(sequences), torch.LongTensor(labels)

def train_epoch(model, dataloader, optimizer, device, lambda1=0.001, lambda2=0.0001, lambda3=0.01):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)
        logits = model(sequences, training=True) # Explicitly set training=True for gradients
        loss, components = model.compute_total_loss(logits, labels, lambda1, lambda2, lambda3)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += components['ce_loss']
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, avg_ce_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)
            # training=False for evaluation
            logits = model(sequences, training=False)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    return accuracy, all_preds, all_labels

def main():
    # Hardcoded settings for this specific rerun task
    data_path = "c:\\Users\\Dell\\Desktop\\HAR_with_RC\\Source Code and Dataset\\CZU-MHAD-skeleton_mat"
    reservoir_size = 1000
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    fusion_strategy = 'concat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './results_rerun'
    
    print(f"Rerunning on {device} for {epochs} epochs...")
    print(f"Data path: {data_path}")
    
    # Load Data
    sequences, labels, metadata = load_skeleton_data(data_path)
    print(f"Loaded {len(sequences)} samples, {metadata['num_classes']} classes")
    
    # Normalize
    sequences, scaler = normalize_sequences(sequences, method='standard')
    
    # Process lengths
    length_stats = compute_sequence_lengths(sequences)
    T_max = int(length_stats['percentile_95'])
    print(f"T_max: {T_max}")
    
    # Split
    X_train, X_test, y_train, y_test = prepare_data_for_training(sequences, labels, test_size=0.2, random_state=42)
    
    # Init Model
    input_dim = X_train[0].shape[1]
    model = BidirectionalRC(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        num_classes=metadata['num_classes'],
        fusion_strategy=fusion_strategy,
        random_seed=42
    )
    model.T_max = T_max
    model = model.to(device)
    
    # Phase 1: Reservoir
    print("Phase 1: Reservoir processing...")
    R_samples = []
    model.eval()
    with torch.no_grad():
        for X in tqdm(X_train, desc="Reservoir"):
            R = model.forward_reservoir(X)
            R_samples.append(R)
            
    # Phase 2: Fit (This now initializes AdvancedReadout)
    print("Phase 2: Fitting dimensionality reduction (Initializing AdvancedReadout)...")
    model.fit_dimensionality_reduction(R_samples)
    model.fitted = True
    print(f"Readout type: {type(model.readout)}")
    
    # Phase 3: Training
    print("Phase 3: Training...")
    train_dataset = SkeletonDataset(X_train, y_train)
    test_dataset = SkeletonDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        train_loss, train_ce, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")
        
    # ==========================================
    # Visualization using IN-MEMORY model
    # ==========================================
    print("\nGenerating Visualizations...")
    
    # 1. Confusion Matrix
    _, all_preds, all_labels_viz = evaluate(model, test_loader, device)
    
    cm = confusion_matrix(all_labels_viz, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix (Acc: {test_acc:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_rerun.png'))
    plt.close()
    print("Saved confusion_matrix_rerun.png")
    
    # 2. t-SNE
    print("Extracting features for t-SNE...")
    model.eval()
    all_features = []
    # Use subset for speed if needed, but 20% of 880 is ~176 samples, fast enough
    # Use X_test
    
    # Important: Extract features using model.extract_features
    # We can do this sample by sample
    with torch.no_grad():
        for x in tqdm(X_test, desc="Extracting features"):
            # extract_features returns numpy array if training=False
            f = model.extract_features(x, training=False)
            all_features.append(f)
            
    all_features = np.stack(all_features)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Action Features')
    plt.savefig(os.path.join(save_dir, 'tsne_plot_rerun.png'))
    plt.close()
    print("Saved tsne_plot_rerun.png")
    
    print("Rerun Complete!")

if __name__ == '__main__':
    main()
