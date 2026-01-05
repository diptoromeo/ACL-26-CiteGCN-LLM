# ACL-26-CiteGCN-LLM
Citation-Aware Research Paper Classification via Graph Convolutional Networks and Large Language Models

This repository contains the official implementation of CiteGCN-LLM, a unified framework for research paper classification and recommendation that jointly models citation structure and textual semantics using Graph Convolutional Networks (GCN) and Large Language Models (LLMs).

📌 Overview

Modern scholarly corpora exhibit rich relational structures (citations, co-authorship) that cannot be fully captured by text-only models. CiteGCN-LLM addresses this limitation by:

Constructing heterogeneous citation graphs

Injecting transformer-pooled semantic embeddings

Jointly optimizing graph-based and text-based predictors under a unified multi-label objective

The framework achieves state-of-the-art performance on multiple large-scale citation datasets.

🧠 Key Contributions

Two novel citation graph constructions

RPCG-1: Paper–Word–Citation graph (TF-IDF + PPMI + citation edges)

RPCG-2: Paper–Author–Citation graph (co-authorship + citation edges)

Joint GCN–LLM training

Structural logits from GCN

Semantic logits from a pretrained transformer

Optimized using BCEWithLogitsLoss

Top-K label generation

Automatic multi-label supervision derived from TF-IDF statistics over paper titles

Extensive evaluation

arXiv, DBLP, Elsevier, PubMed

Robust gains in Accuracy and F1
