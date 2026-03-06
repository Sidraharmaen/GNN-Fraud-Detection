# ₿ GNN-Fraud-Detection: Blockchain Anti-Money Laundering (AML)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io/)

A state-of-the-art **Graph Neural Network (GNN)** solution for detecting illicit transactions in the Bitcoin blockchain. Built using **PyTorch Geometric** and the **GraphSAGE** architecture, this project targets the critical business problem of Anti-Money Laundering (AML) in decentralized finance.

---

## 💼 Business Problem: The Cost of Global Money Laundering

Financial institutions and exchanges face a massive challenge in identifying "dirty" money on the blockchain. Traditional rule-based systems often fail because they lack the ability to analyze the **topological relationships** between addresses.

**The GNN Advantage:**
Money laundering isn't just about a single transaction; it's about a **pattern of movement**. By modeling transactions as a graph, our system identifies suspicious chains of flow that are invisible to standard machine learning models.

---

## 🧠 Model Architecture: GraphSAGE

Instead of viewing transactions in isolation, our model utilizes **GraphSAGE (SAGEConv)**. This inductive learning approach allows the model to:
- **Aggregate Information:** Nodes "listen" to their neighbors' features.
- **Learn Topology:** It identifies structural patterns (e.g., mixing services or high-frequency cycling).
- **Generalize:** Using inductive learning, the model can infer risk for transactions it has never seen before.

**Deployment Specifications:**
- **Layers:** 2x SAGEConv Layers (128 → 64 hidden units).
- **Optimization:** Weighted NLL Loss to combat severe class imbalance (only ~2% of data is illicit).
- **Regularization:** 0.3 Dropout to prevent overfitting on the dense benchmark data.

---

## 📈 Performance Metrics

Our model achieves industry-leading discriminatory power on the **Elliptic Benchmark Dataset**:

| Metric | Score | Key Takeaway |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9501** | Near-perfect separation between licit and illicit flows. |
| **Fraud Recall** | **89.0%** | **Crucial for AML:** Minimal false negatives (missed fraud). |
| **Macro F1 Score** | **0.74** | High performance across imbalanced classes. |

---

## 🚀 Live Demo & Dashboard

The project includes an interactive **Streamlit Dashboard** for real-time risk assessment:

- **Interactive ID Lookup:** Search any transaction to get a live risk score.
- **Neighborhood Visualization:** Dynamic 1-hop graphing of the transaction context.
- **Explainable AI (XAI):** Z-Score analysis identifying *why* a transaction was flagged as high-risk.
- **Custom Uploads:** Upload your own `features.csv` and `edgelist.csv` to run the model on new data.

> **[Click Here for Live Demo Placeholder]** *(http://localhost:8501)*

---

## 💻 Local Installation

Ensure you have **Python 3.11** installed.

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/GNN-Fraud-Detection.git
cd GNN-Fraud-Detection
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset
The app expects the [Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) in the `data/` folder.
```bash
python load_elliptic.py
```

### 3. Launch UI
```bash
streamlit run app.py
```

---

## 📚 Dataset Citation

The data used in this project is provided by **Elliptic**. It is the world's largest labeled transaction dataset in the decentralized economy.

- **Reference:** *Weber, M., et al. "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks." KDD 2019.*
- **Source:** [Kaggle: Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

## 🛠️ Project Structure
- `app.py`: The Streamlit Dashboard.
- `train_elliptic.py`: Training script with GraphSAGE architecture.
- `graph_builder.py`: Pipeline for converting CSVs to PyTorch Geometric Data objects.
- `outputs/`: Visualization results (Confusion Matrix, ROC Curves).

---
*Created by Sidra Harmaen. 
*Contact for collaboration on Graph ML and AML solutions.
