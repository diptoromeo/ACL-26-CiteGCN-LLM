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
  Paper Abstracts ──▶ LLM Encoder ──▶ Semantic Logits
       │
       ▼
  Citation Graph ──▶ GCN Layers ──▶ Structural Logits
       │
       └────────────── Fusion ──────────────▶ Final Prediction

  - GCN captures citation and relational structure

  - LLM captures deep semantic meaning

  - Outputs are combined at the logit level

