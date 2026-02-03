# Project 3A – Similarity Learning in Deep Learning  
**Supervised and Unsupervised Similarity Methods**

**Author:** Théa Chaduteau  
**Program:** FICM – Mines Nancy (GIMA ID Big Data)  
**Date:** February 2025



## 1. Project Overview

This project explores **similarity learning** through both **supervised** and **unsupervised** approaches in deep learning.  
The objective is to go beyond classical fixed similarity metrics (Euclidean, cosine, etc.) by designing models that **learn similarity directly from data**, across different data modalities.

Three complementary approaches are implemented and compared:

1. **Supervised similarity learning** using a Siamese Network with Triplet Loss  
2. **Unsupervised similarity learning** using an enhanced Convolutional Autoencoder  
3. **Algorithmic similarity** using Levenshtein distance, with scalability analysis and LSH-based extensions  

The project combines **theoretical foundations**, **model implementation**, **experimental evaluation**, and **critical analysis of limitations**.



## 2. Objectives

- Study and formalize different definitions of similarity:
  - Feature-based
  - Semantic
  - Relational
  - Transformation-based
- Implement **state-of-the-art similarity learning models**
- Compare **supervised vs unsupervised** similarity learning
- Evaluate scalability and robustness on real datasets
- Analyze trade-offs between accuracy, interpretability, and computational cost



## 3. Implemented Methods

### 3.1 Siamese Network (Supervised)

**Key characteristics**
- Shared-weight embedding network
- Triplet-based training (Anchor / Positive / Negative)
- Dynamic margin Triplet Loss
- Hard negative mining
- Attention mechanism on embeddings
- Multi-metric similarity fusion:
  - Cosine similarity
  - Euclidean distance
  - Dot product

**Datasets**
- MNIST
- Fashion-MNIST

**Results**
- Cosine similarity significantly outperforms Euclidean distance
- Best performance achieved with **learned fusion of multiple metrics**
- Well-separated embeddings (t-SNE / LDA)
- High generalization accuracy on unseen data



### 3.2 Enhanced Convolutional Autoencoder (Unsupervised)

**Motivation**
Unsupervised similarity learning is harder due to the absence of labels.  
This model aims to learn **meaningful latent embeddings** usable for clustering.

**Architecture**
- Fully convolutional encoder / decoder
- Batch normalization
- Contrastive loss + reconstruction loss (MSE)
- UMAP for dimensionality reduction
- K-Means clustering in latent space

**Key improvements over classical autoencoders**
- Convolutional layers preserve spatial structure
- Contrastive loss enforces semantic separation
- Hierarchical feature extraction
- Robust latent clustering

**Datasets**
- Custom image sets (dogs, trees)
- MNIST (validation of scalability)

**Results**
- Clear cluster separation in latent space
- Better performance than linear autoencoders
- Stable unsupervised embeddings



### 3.3 Levenshtein-Based Similarity (Algorithmic)

**Approach**
- Custom implementation of Levenshtein distance
- Applied to:
  - Textual / categorical sequences
  - Image features extracted via HOG

**Clustering**
- Full distance matrix
- DBSCAN clustering

**Limitations**
- Quadratic time complexity
- High memory usage
- Not scalable to large datasets

**Proposed Improvement**
- Locality Sensitive Hashing (LSH)
- Approximate nearest neighbors
- Significant scalability gains at the cost of approximation
