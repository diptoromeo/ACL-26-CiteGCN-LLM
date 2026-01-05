# ACL-26-CiteGCN-LLM
**Citation-Aware Research Paper Classification via Graph Convolutional Networks and Large Language Models**

This repository contains the official implementation of **CiteGCN-LLM**, a unified framework for research paper classification and recommendation that jointly models citation structure and textual semantics using Graph Convolutional Networks (GCNs) and Large Language Models (LLMs).

The framework constructs heterogeneous citation graphs and integrates them with transformer-based textual representations to achieve state-of-the-art performance on scholarly datasets.

## 📌 **Overview**

Scholarly documents are inherently interconnected through citations, authorship, and shared terminology. Purely text-based models fail to capture these relational signals, while graph-only methods lack deep semantic understanding.

**CiteGCN-LLM bridges this gap by:**

  - Modeling citation topology using GCNs

  - Extracting contextual semantics from abstracts using pretrained LLMs

  - ointly optimizing both branches under a unified multi-label objective

## 🧠 **Key Contributions**

- **Two novel heterogeneous citation graphs**

  - RPCG-1: Paper–Word–Citation graph (TF-IDF + PMI + citations)

  - RPCG-2: Paper–Author–Citation graph (co-authorship + citations)

- **Unified GCN–LLM architecture**

  - Structural logits from GCN

  - Semantic logits from transformer encoder

  - Joint training with BCEWithLogits loss

- **Top-K TF-IDF based labeling**

  - Interpretable, personalized multi-label supervision

- **Extensive evaluation**

  - arXiv, DBLP, Elsevier, and PubMed

  - Significant gains over GCNs, Graph Transformers, and LLM-only baselines
 

## 🏗️ **Framework Architecture**

```text
Paper Abstracts  -->  LLM Encoder  -->  Semantic Logits
        |                                
        v                                
Citation Graph  -->  GCN Layers  -->  Structural Logits
        |
        +---------------- Fusion ----------------> Final Prediction
```

  - GCN captures citation and relational structure

  - LLM captures deep semantic meaning

  - Outputs are combined at the logit level


## 📊 **Graph Construction**
**RPCG-1: Paper–Word–Citation Graph**

  - Paper–Word edges weighted by TF-IDF

  - Word–Word edges weighted by Positive PMI

  - Paper–Paper edges weighted by citation counts

**RPCG-2: Paper–Author–Citation Graph**

  - Paper–Author membership edges

  - Author–Author edges via co-authorship (PMI)

  - Paper–Paper citation edges

Both graphs are normalized and processed using standard GCN message passing.

## 🏷️ Labeling Strategy

- Extract Top-K globally salient terms from paper titles using TF-IDF

- Assign each paper a multi-hot label vector

- Enables personalized and interpretable classification


## ⚙️ Training Objective

CiteGCN-LLM is trained end-to-end with a **joint binary cross-entropy loss**:

![loss](https://latex.codecogs.com/svg.image?\mathcal{L}=\mathcal{L}_{\mathrm{GCN}}+\mathcal{L}_{\mathrm{LLM}})

- Uses BCEWithLogitsLoss

- Supports independent multi-label predictions

- Single backpropagation step updates both branches


## 📈 Experimental Results

- Consistently outperforms:

	- GCN, GAT, GraphSAGE, GIN

	- Graph Transformers (Graph-BERT, Graphormer)

	- LLM-augmented GNN baselines

- Achieves 93%+ accuracy on arXiv and DBLP

- RPCG-2 shows strong gains on socially connected datasets


## 🗂️ Repository Structure

```text
.
├── data/
│   ├── arxiv/
│   ├── dblp/
│   ├── elsevier/
│   └── pubmed/
├── graph/
│   ├── build_rpcg1.py
│   └── build_rpcg2.py
├── models/
│   ├── gcn.py
│   ├── llm_encoder.py
│   └── citegcn_llm.py
├── train.py
├── evaluate.py
├── utils/
└── README.md

```

## 🔧 Requirements

- Python ≥ 3.9

- PyTorch

- PyTorch Geometric

- HuggingFace Transformers

- NumPy, SciPy, scikit-learn

- NLTK


## 🚀 Usage (High-Level)

Preprocess datasets and construct citation graphs

1. Extract LLM embeddings from abstracts

2. Train CiteGCN-LLM jointly on graph and text

3. Evaluate classification accuracy and F1 scores

_(Detailed scripts and configurations are provided in the repository.)_



## Citation

```bibtex
@inproceedings{citegcnllm2025,
  title     = {CiteGCN-LLM: Citation-Aware Research Paper Classification via Graph Convolutional Networks and Large Language Models},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACL},
  year      = {2026}
}
```

