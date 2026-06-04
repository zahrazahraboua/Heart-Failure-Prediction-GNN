# Multimodal Heart Failure Prediction using Graph Neural Networks (GNN)

This repository contains the official Python/PyTorch implementation of the Master's graduation thesis: **"Prediction of Heart Failure Diseases using Deep Learning Models"** presented at the Department of Computer Science, University of 8 May 1945 - Guelma.

This end-to-end framework proposes a **Multimodal Graph-based Deep Learning Pipeline** that fuses visual feature embeddings from Electrocardiogram (ECG) images with clinical numerical features. Relations between patient profiles are modeled into a non-Euclidean similarity graph via the $k$-Nearest Neighbors ($k$-NN) algorithm, and classification is performed utilizing a Graph Convolutional Network (GCN).



##  Pipeline & Methodology Overview

The architecture processes patient diagnostics through a specialized multimodal and graph-driven pipeline:

1. **Data Cleansing & Validation:** Filters out corrupted files and guarantees a minimum image dimension threshold ($100 \times 100$).
2. **Class Balancing via Augmentation:** Addresses dataset imbalance by expanding the minority class (`Normal`) using an `ImageDataGenerator` framework (applying rotation, shifts, zooms, and horizontal flips) to match the majority class (`Malade`).
3. **Clinical Feature Engineering:** Generates 5 critical medical variables sampled from pathological statistical distributions mapped to patient groups:
   - **EF (Ejection Fraction):** Normal distribution ($\mu=0.60$ for healthy vs. $\mu=0.38$ for pathological).
   - **BNP (Brain Natriuretic Peptide):** Lognormal distribution ($\mu=4.2$ vs. $\mu=5.8$).
   - **NYHA Functional Class:** Categorical choice (classes 1–4) weighted by heart failure severity.
   - **SBP (Systolic Blood Pressure):** Normal distribution ($\mu=125$ mmHg vs. $\mu=110$ mmHg).
   - **Age:** Continuous integer sampling ($\mu=55$ vs. $\mu=68$ years).
4. **Multimodal Feature Fusion:**
   - Visual feature extraction is performed using a pre-trained **ResNet18** backbone (with the final fully-connected classification layer removed), producing a 512-dimensional embedding tensor per ECG image.
   - Clinical vectors (5 dimensions) are normalized via `StandardScaler` and concatenated directly with the visual embeddings, outputting a comprehensive **517-dimensional** node feature vector ($x_i \in \mathbb{R}^{517}$).
5. **Graph Construction ($5$-NN):** Computes patient-to-patient structural links based on their multi-domain features using a 5-Nearest Neighbors graph topology (`kneighbors_graph(n_neighbors=5)`). Self-loops are added to stabilize spatial message-passing processing.
6. **Graph Convolutional Network Classification:** Evaluates node representations using a custom `GlobalGCN` network composed of two `GCNConv` layers, localized ReLU non-linearities, Dropout regularization ($p=0.5$), and a Softmax linear classification layer.
7. **Real-Time Single-Patient Inference:** Includes a dynamic induction pipeline that seamlessly injects an unlabelled patient into the existing graph topology, recalculates local adjacency links via $k$-NN, and predicts patient status (`Malade` vs. `Normal`).



##  Tech Stack & Key Libraries

- **Language:** Python 3
- **Deep Learning Framework:** PyTorch & PyTorch Geometric (PyG)
- **Computer Vision:** Torchvision (Pre-trained ResNet18)
- **Data Augmentation:** TensorFlow / Keras (ImageDataGenerator)
- **Scientific Computing & Preprocessing:** Scikit-Learn, Pandas, NumPy
- **Visualization:** Seaborn, Matplotlib



## 📂 Repository Tree Structure

```text
├── heart_disease_gcn.py       # Complete executable pipeline script
├── multi.ipynb                # Interactive Google Colab Jupyter Notebook
├── README.md                  # Project documentation manual
└── requirements.txt           # Python environment packages requirements
