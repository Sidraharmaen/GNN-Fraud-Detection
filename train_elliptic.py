"""
train_elliptic.py — Training a GraphSAGE model on the Elliptic Bitcoin Dataset.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from graph_builder import build_graph

# ────────────────────────────────────────────────────────────────────────
# 1. Model Definition: GraphSAGE
# ────────────────────────────────────────────────────────────────────────

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels_1)
        self.conv2 = SAGEConv(hidden_channels_1, hidden_channels_2)
        # Final classification layer
        self.lin = torch.nn.Linear(hidden_channels_2, out_channels)
        self.dropout = 0.3

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# ────────────────────────────────────────────────────────────────────────
# 2. Main Training Script
# ────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cpu')
    print(f"🖥️ Using Device: {device}")

    # Build graph data
    data = build_graph()
    data = data.to(device)

    # Filter labeled nodes
    labeled_indices = data.labeled_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    labels = data.y[data.labeled_mask].cpu().numpy()

    # Split into train and validation (80/20) for labeled nodes
    train_idx, val_idx = train_test_split(
        labeled_indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create manual masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    # Handle Class Imbalance with Loss Weights
    # Count occurrences in training set
    train_labels = data.y[train_mask]
    label_0_count = (train_labels == 0).sum().item()
    label_1_count = (train_labels == 1).sum().item()
    
    # Weights proportional to 1/count
    # Larger weight for class 1 (fraud) since it's smaller
    weights = torch.tensor([1.0 / label_0_count, 1.0 / label_1_count], dtype=torch.float).to(device)
    # Normalize weights
    weights = weights / weights.sum() * 2.0
    print(f"⚖️ Loss Weights: Legitimate (0)={weights[0]:.4f}, Fraud (1)={weights[1]:.4f}")

    # Initialize Model
    model = GraphSAGEModel(
        in_channels=data.num_node_features,
        hidden_channels_1=128,
        hidden_channels_2=64,
        out_channels=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss(weight=weights)

    best_f1 = 0
    epochs = 30
    history = []

    print("\n🚀 Starting Training Loop (30 Epochs)...")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # Training Step
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation Step
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out[val_mask].argmax(dim=1)
            y_true = data.y[val_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            
            f1 = f1_score(y_true, y_pred, average='macro')
            # Save history
            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'val_f1': f1
            })

        # Save Best Model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val F1 (Macro): {f1:.4f}")

    # Save metrics to CSV
    pd.DataFrame(history).to_csv('metrics.csv', index=False)
    print("\n📊 Training history saved to metrics.csv")
    print("-" * 50)
    print(f"✅ Training Complete! Best Val F1: {best_f1:.4f}")
    print("💾 Best model saved to: best_model.pt")

if __name__ == "__main__":
    main()
