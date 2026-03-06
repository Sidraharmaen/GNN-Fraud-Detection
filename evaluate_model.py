"""
evaluate_model.py — Evaluating the GraphSAGE model on the Elliptic Dataset.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split
from torch_geometric.nn import SAGEConv
from graph_builder import build_graph

# ────────────────────────────────────────────────────────────────────────
# 1. Model Definition (Must match train_elliptic.py)
# ────────────────────────────────────────────────────────────────────────

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels_1)
        self.conv2 = SAGEConv(hidden_channels_1, hidden_channels_2)
        self.lin = torch.nn.Linear(hidden_channels_2, out_channels)
        self.dropout = 0.3

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# ────────────────────────────────────────────────────────────────────────
# 2. Main Evaluation
# ────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cpu')
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("⏳ Loading graph and model...")
    data = build_graph().to(device)
    
    model = GraphSAGEModel(
        in_channels=data.num_node_features,
        hidden_channels_1=128,
        hidden_channels_2=64,
        out_channels=2
    ).to(device)
    
    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
    model.eval()

    # Get evaluation indices (same 20% validation split as training)
    labeled_indices = data.labeled_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    labels = data.y[data.labeled_mask].cpu().numpy()
    _, val_idx = train_test_split(
        labeled_indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Predict
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        # Class probabilities
        probs = torch.exp(out[val_idx])[:, 1].cpu().numpy()
        # Predictions
        preds = out[val_idx].argmax(dim=1).cpu().numpy()
        y_true = data.y[val_idx].cpu().numpy()

    print("\n" + "="*50)
    print("📈 Model Evaluation Results")
    print("="*50)

    # 1. Classification Report (P, R, F1 for both classes)
    report = classification_report(y_true, preds, target_names=['Legit (0)', 'Fraud (1)'])
    print("\nClassification Report:")
    print(report)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    print(f"📊 Saved Confusion Matrix to {output_dir}/confusion_matrix.png")

    # 3. ROC-AUC Score
    auc = roc_auc_score(y_true, probs)
    print(f"\nROC-AUC Score: {auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png')
    print(f"📊 Saved ROC Curve to {output_dir}/roc_curve.png")

    # 4. Training Loss Curve
    if os.path.exists('metrics.csv'):
        history = pd.read_csv('metrics.csv')
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['loss'], label='Loss', color='red')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('NLL Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['val_f1'], label='Val F1', color='blue')
        plt.title('Validation Macro F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_curves.png')
        print(f"📊 Saved Training Curves to {output_dir}/training_curves.png")
    else:
        print("⚠️ metrics.csv not found, skipping loss curve plot.")

    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
