"""
GNN Fraud Detection - Training Script
Trains a Graph Attention Network (GAT) on transaction graph data.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx

# ─────────────────────────────────────────────
# 1.  Model Definition
# ─────────────────────────────────────────────

class FraudGAT(torch.nn.Module):
    """Graph Attention Network for fraud detection."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ─────────────────────────────────────────────
# 2.  Synthetic Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_data(num_nodes: int = 500, num_edges: int = 2000,
                             fraud_ratio: float = 0.1) -> Data:
    """
    Generate a synthetic transaction graph for demonstration.
    Replace this with your real dataset loader.
    """
    np.random.seed(42)

    # Node features: [amount, frequency, avg_amount, hour, is_new_account]
    features = np.random.randn(num_nodes, 5).astype(np.float32)

    # Labels: 1 = fraud, 0 = legitimate
    labels = (np.random.rand(num_nodes) < fraud_ratio).astype(int)

    # Random edges (transactions between accounts)
    edge_src = np.random.randint(0, num_nodes, num_edges)
    edge_dst = np.random.randint(0, num_nodes, num_edges)
    edge_index = np.stack([edge_src, edge_dst], axis=0)

    # Train / val / test masks
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data = Data(
        x          = torch.tensor(features),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        y          = torch.tensor(labels, dtype=torch.long),
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )
    return data


# ─────────────────────────────────────────────
# 3.  Training Loop
# ─────────────────────────────────────────────

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out  = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    acc  = (pred == data.y[mask]).float().mean().item()
    return acc, pred.cpu().numpy(), data.y[mask].cpu().numpy()


# ─────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────

def main():
    print("🔍 GNN Fraud Detection — Training")
    print("=" * 40)

    # Data
    data = generate_synthetic_data()
    print(f"Nodes : {data.num_nodes}")
    print(f"Edges : {data.num_edges}")
    print(f"Features : {data.num_node_features}")
    print(f"Fraud rate : {data.y.float().mean():.1%}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FraudGAT(
        in_channels=data.num_node_features,
        hidden_channels=32,
        out_channels=2,
    ).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Training
    train_losses = []
    val_accs     = []
    epochs       = 100

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        val_acc, _, _ = evaluate(model, data, data.val_mask)
        train_losses.append(loss)
        val_accs.append(val_acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    # Test
    test_acc, preds, labels = evaluate(model, data, data.test_mask)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Legitimate", "Fraud"]))

    # Save model
    torch.save(model.state_dict(), "models/fraud_gat.pt")
    print("💾 Model saved to models/fraud_gat.pt")

    # Plot training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, color="#6C63FF", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(val_accs, color="#FF6584", linewidth=2)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/training_curve.png", dpi=150)
    print("📊 Training curve saved to models/training_curve.png")


if __name__ == "__main__":
    main()
