# Structural Analysis of the Networks Formed by Wikipedia Editors

This repository contains the code developed as part of the **Master’s Thesis (TFM)** entitled:

**“Structural Analysis of the Networks Formed by Wikipedia Editors”**

The main objective of this project is to analyze the **structure of collaboration networks** that emerge among Wikipedia editors using tools from **graph theory and complex network analysis**, with particular emphasis on community detection and structural pattern identification.

The code is made publicly available in order to **ensure the reproducibility** of the results presented in the Master’s Thesis.

---

## Repository Contents

The general structure of the repository is as follows:

```plaintext
VIU_TFM/
├── data/ # Input data (raw and/or preprocessed)
│ ├── 01_wiki/ # Wikipedia-related files (e.g., list of bots)
│ ├── 02_csv/ # Temporary CSV files generated during processing
│ ├── 03_graph/ # Graph data created with NetworkX
│ └── 01_plots/ # Plots and visualizations
├── src/ # Main source code
│ ├── preprocessing/ # Data cleaning, preparation, and graph creation
│ └── analysis/ # Network metrics and analysis
├── requirements.txt # Project dependencies
└── README.md # Repository documentation


