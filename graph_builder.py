"""
graph_builder.py — Process Elliptic Bitcoin Dataset into PyTorch Geometric format.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def build_graph():
    data_dir = '/Users/macbook/Desktop/GNN Fraud detection/data'
    
    print("⏳ Loading CSV files...")
    # Load data
    df_features = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_features.csv'), header=None)
    df_classes = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_classes.csv'))
    df_edges = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_edgelist.csv'))

    # Rename columns for convenience
    df_features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
    
    print("🛠️ Mapping Transaction IDs to indices...")
    # Map txId to 0...N-1 indices
    nodes = df_features['txId'].values
    map_id = {j: i for i, j in enumerate(nodes)}
    
    # 1. Node Features (X)
    # Dropping txId and timestep from the features matrix for training
    x = torch.tensor(df_features.iloc[:, 2:].values, dtype=torch.float)
    
    # 2. Edge Index (Edge List)
    # Map the txIds in edgelist to our 0...N-1 indices
    edge_index = df_edges.copy()
    edge_index['txId1'] = edge_index['txId1'].map(map_id)
    edge_index['txId2'] = edge_index['txId2'].map(map_id)
    
    # Remove any edges that might have broken mapping (shouldn't happen with this dataset)
    edge_index = edge_index.dropna()
    
    edge_index = torch.tensor(edge_index.values.T, dtype=torch.long)
    
    # 3. Labels (Y)
    # Mapping: '1' (illicit) -> 1, '2' (licit) -> 0, 'unknown' -> -1
    def map_class(c):
        if c == '1': return 1
        if c == '2': return 0
        return -1
    
    labels = df_classes['class'].apply(map_class).values
    y = torch.tensor(labels, dtype=torch.long)
    
    # 4. Create Masks
    # We only care about nodes that aren't 'unknown' (-1)
    # Usually we split labeled nodes into train/test. 
    # For now, let's just identify which are labeled.
    labeled_mask = (y != -1)
    
    # ────────────────────────────────────────────────────────────────────────
    # Construct PyTorch Geometric Data object
    # ────────────────────────────────────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Add masks to the data object
    data.labeled_mask = labeled_mask
    
    # Stats
    total_nodes = data.num_nodes
    total_edges = data.num_edges
    fraud_nodes = (y == 1).sum().item()
    legit_nodes = (y == 0).sum().item()
    unknown_nodes = (y == -1).sum().item()

    print("\n" + "="*50)
    print("🚀 Graph Construction Complete")
    print("="*50)
    print(f"1. Total number of nodes: {total_nodes:,}")
    print(f"2. Total number of edges: {total_edges:,}")
    print(f"3. Node Classification counts:")
    print(f"   - Fraudulent (1):  {fraud_nodes:,}")
    print(f"   - Legitimate (0):  {legit_nodes:,}")
    print(f"   - Unknown nodes:   {unknown_nodes:,}")
    print(f"4. PyTorch Geometric Data object: READY ✅")
    print(f"   Structure: {data}")
    print("="*50)
    
    return data

if __name__ == "__main__":
    data_obj = build_graph()
    
    # Optional: Save the torch object for training
    # torch.save(data_obj, 'data/elliptic_graph.pt')
    # print("\n💾 Saved to data/elliptic_graph.pt")
