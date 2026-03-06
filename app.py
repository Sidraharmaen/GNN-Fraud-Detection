"""
app.py — Premium GNN Fraud Detection Dashboard for Elliptic Bitcoin Dataset.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

# ── 1. Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Bitcoin Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── 2. Premium Design System (CSS) ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall Background */
    .stApp {
        background-color: #0b0e14;
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 24px;
        flex: 1;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk Status Colors */
    .risk-low { color: #00ff88; text-shadow: 0 0 15px rgba(0, 255, 136, 0.3); }
    .risk-medium { color: #ffbb00; text-shadow: 0 0 15px rgba(255, 187, 0, 0.3); }
    .risk-high { color: #ff3e3e; text-shadow: 0 0 15px rgba(255, 62, 62, 0.3); }

    /* Custom Header */
    .main-title {
        background: linear-gradient(90deg, #5c7cff, #ff5cbb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #666;
        margin-bottom: 30px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Model Definition ──────────────────────────────────────────
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

# ── 4. Data Loading Logic ────────────────────────────────────────

@st.cache_resource
def load_elliptic_defaults():
    data_dir = '/Users/macbook/Desktop/GNN Fraud detection/data'
    df_features = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_features.csv'), header=None)
    df_features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]
    df_edges = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_edgelist.csv'))
    return df_features, df_edges

def process_custom_data(df_features, df_edges):
    # Ensure correct columns
    if df_features.shape[1] < 167:
        st.error(f"❌ Uploaded features file has only {df_features.shape[1]} columns. Expected at least 167 (txId, timestep + 165 features).")
        return None
    
    # Standardize column names for the app
    df_features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(df_features.shape[1]-2)]
    
    nodes = df_features['txId'].values
    map_id = {j: i for i, j in enumerate(nodes)}
    rev_map_id = {i: j for i, j in enumerate(nodes)}
    
    # Features (X) - take exactly 165 features for the model
    x = torch.tensor(df_features.iloc[:, 2:167].values, dtype=torch.float)
    
    # Edges
    edge_index_raw = df_edges.copy()
    edge_index_raw.columns = ['txId1', 'txId2']
    edge_index_raw['txId1'] = edge_index_raw['txId1'].map(map_id)
    edge_index_raw['txId2'] = edge_index_raw['txId2'].map(map_id)
    edge_index = torch.tensor(edge_index_raw.dropna().values.T, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    return data, map_id, rev_map_id, df_features

@st.cache_resource
def get_model():
    model = GraphSAGEModel(165, 128, 64, 2)
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ── 5. App Application ──────────────────────────────────────────

# Header
st.markdown("<h1 class='main-title'>₿ Fraud Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Detecting illicit transactions on the blockchain using Graph Neural Networks.</p>", unsafe_allow_html=True)

# 🔍 Sidebar - Data Source Selection
with st.sidebar:
    st.header("📂 Data Source")
    source_type = st.radio("Choose Dataset", ["Elliptic Benchmark", "Custom CSV Upload"])
    
    data_loaded = False
    
    if source_type == "Elliptic Benchmark":
        df_feat_raw, df_edges_raw = load_elliptic_defaults()
        data, map_id, rev_map_id, df_features = process_custom_data(df_feat_raw, df_edges_raw)
        data_loaded = True
    else:
        st.divider()
        st.subheader("📤 Upload Custom Files")
        feat_file = st.file_uploader("Upload Features CSV", type=['csv'], help="Cols: txId, timestep, 165 features")
        edge_file = st.file_uploader("Upload Edgelist CSV", type=['csv'], help="Cols: txId1, txId2")
        
        if feat_file and edge_file:
            df_feat_raw = pd.read_csv(feat_file, header=None if "elliptic" in feat_file.name else 'infer')
            df_edges_raw = pd.read_csv(edge_file)
            processed = process_custom_data(df_feat_raw, df_edges_raw)
            if processed:
                data, map_id, rev_map_id, df_features = processed
                data_loaded = True

    st.divider()
    if data_loaded:
        st.success(f"Graph Loaded: {data.num_nodes:,} nodes")
        st.markdown("### 🔎 Query Transaction")
        sample_ids = df_features['txId'].sample(min(5, len(df_features))).tolist()
        st.caption(f"Example IDs: {', '.join(map(str, sample_ids))}")
        search_id = st.number_input("Enter txId", value=int(sample_ids[0]) if sample_ids else 0, step=1)
    else:
        st.warning("Please upload both CSV files to proceed.")

# ── 6. Main Dashboard Content ──────────────────────────────────

if data_loaded:
    model = get_model()
    
    if search_id not in map_id:
        st.error(f"❌ ID {search_id} not found.")
    else:
        node_idx = map_id[search_id]
        
        # Inference
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = torch.exp(out[node_idx])
            fraud_prob = float(probs[1])

        # Risk UI
        if fraud_prob < 0.3: risk_lvl, risk_cls = "LOW RISK", "risk-low"
        elif fraud_prob < 0.7: risk_lvl, risk_cls = "MEDIUM RISK", "risk-medium"
        else: risk_lvl, risk_cls = "HIGH RISK", "risk-high"

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card"><div class="metric-label">Fraud Probability</div><div class="metric-value {risk_cls}">{fraud_prob*100:.1f}%</div></div>
            <div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value {risk_cls}">{risk_lvl}</div></div>
            <div class="metric-card"><div class="metric-label">Network Neighbors</div><div class="metric-value" style="color:#5c7cff">{data.edge_index.shape[1]} total edges</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Plotting & Analysis
        col_viz, col_feat = st.columns([1.2, 1])

        with col_viz:
            st.subheader("🌐 Local Neighborhood")
            subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, num_hops=1, edge_index=data.edge_index, relabel_nodes=False)
            G = nx.Graph()
            G.add_nodes_from(subset.tolist())
            G.add_edges_from(sub_edge_index.t().tolist())
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0b0e14')
            ax.set_facecolor('#0b0e14')
            pos = nx.spring_layout(G, k=0.3, seed=42)
            node_colors = ['#ff3e3e' if n == node_idx and fraud_prob > 0.5 else '#00ff88' if n == node_idx else '#2c3e50' for n in G.nodes()]
            nx.draw(G, pos, ax=ax, node_size=600, node_color=node_colors, with_labels=False, edge_color='#444', edgecolors='white')
            st.pyplot(fig)

        with col_feat:
            st.subheader("📊 Feature Analysis")
            node_feats = data.x[node_idx].numpy()
            global_mean = data.x.mean(dim=0).numpy()
            z_scores = (node_feats - global_mean) / (data.x.std(dim=0).numpy() + 1e-9)
            top_indices = np.argsort(np.abs(z_scores))[-10:][::-1]
            importance_df = pd.DataFrame({'Feature': [f"F_{i}" for i in top_indices], 'Z-Score': z_scores[top_indices]}).sort_values('Z-Score')
            st.bar_chart(importance_df, y='Z-Score', x='Feature', color='#ff5cbb')
            
        with st.expander("📋 Raw Record Data"):
            st.dataframe(df_features[df_features['txId'] == search_id].T)
else:
    st.info("💡 Select the Elliptic Benchmark or upload your own transaction data in the sidebar to begin.")

st.divider()
st.caption("Powered by PyTorch Geometric & Streamlit. Elliptic Data Set © Kaggle.")
