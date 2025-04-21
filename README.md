# Neural Networks and Graph Learning – Self-Study Index

This repository documents a structured learning journey from classical neural networks to graph neural networks (GNNs). Projects progress from image classification using MLP/CNN/RNN to graph analysis with NetworkX, and finally to advanced GNN models (GCN, GraphSAGE, GAT) using PyTorch Geometric (PyG) and DGL.

---

## Part I: Image-based Neural Networks (MNIST)

Foundations of deep learning with regular grid-structured data.

- Load and Visualize the MNIST Dataset  
- Train a Multilayer Perceptron (MLP)  
- Train a Convolutional Neural Network (CNN)  
- Train a Recurrent Neural Network (RNN)  
- Visualize Training Curves and Evaluate Accuracy  

---

## Part II: Graph Structure and Representation Basics

Graph theory and network analysis with NetworkX.

- Create Simple Undirected and Directed Graphs  
- Use Built-in Karate Club Graph  
- Calculate Graph Statistics:  
  - Average Degree  
  - Clustering Coefficient  
  - Closeness Centrality (Manual and NetworkX)  
- Create Bipartite Graphs  
- Generate Adjacency Matrix  
- Generate Incidence Matrix  
- Compute Laplacian Matrix (Unnormalized and Normalized)  

---

## Part III: Node Embedding via Supervised Edge Classification

Learn node representations using simple graph-based tasks.  
**Project: Karate Club Node Embedding**

- Load and Visualize the Karate Club Graph  
- Initialize Random Embeddings  
- PCA Projection of Initial Embeddings  
- Create Positive and Negative Edges  
- Train Node Embeddings via Edge Classification  
- Visualize Final Embeddings  

---

## Part IV: Graph Convolutional Networks (GCN)

Graph-level learning for classification tasks.  
**Project: MUTAG Graph Classification with GCN**

- Load and Analyze the MUTAG Dataset  
- Check Graph Properties: Isolated Nodes, Self-Loops, Directionality  
- Implement a 3-layer GCN with Global Mean Pooling  
- Train and Evaluate GCN using Batched Graphs  
- Track Training/Test Accuracy over Epochs  

---

## Part V: GraphSAGE for Link Prediction

Edge-level prediction on citation networks.  
**Project: GraphSAGE on the Cora Graph**

- Preprocess Graph for Link Prediction  
- Create Train/Test Edges and Negative Samples  
- Build Two-layer GraphSAGE Model (mean aggregator)  
- Compare Dot Product and MLP Predictors  
- Train with Binary Cross-Entropy Loss  
- Evaluate with AUC Score  

---

## Part VI: Graph Attention Networks (GAT)

Node classification with attention mechanisms.  
**Project: GAT on Cora Dataset**

- Implement GAT using PyTorch Geometric (GATConv)  
- Re-implement GAT Manually with DGL  
- Compare High-level and Low-level Approaches  
- Train GAT on Cora and Evaluate Node Classification Accuracy  
