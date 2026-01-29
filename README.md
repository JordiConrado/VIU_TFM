# Structural Analysis of the Networks Formed by Wikipedia Editors

This repository contains the code developed as part of the **Master’s Thesis (TFM)** entitled:

**“Structural Analysis of the Networks Formed by Wikipedia Editors”**

The main objective of this project is to analyze the **structure of collaboration networks** that emerge among Wikipedia editors using tools from **graph theory and complex network analysis**, with particular emphasis on community detection and structural pattern identification.

The code is made publicly available in order to **ensure the reproducibility** of the results presented in the Master’s Thesis.

---

## Repository Contents

The general structure of the repository is as follows:
VIU_TFM/
├── data/ # Input data (raw and/or preprocessed)
│ ├── 01_wiki/ # Wikipedia related files (mainly the list of BOTS from Wikipedia)
│ ├── 02_csv/ # Repository for all temporary CSV files created
│ ├── 03_graph/ # Repository for graph data created with NetworkX as part of anañysis
│ ├── 01_plots/ # Repository for plots and visualizations
├── code/ # Main source code
│ ├── Preprocessing/ # Data cleaning, preparation and graph creation
│ ├── Core_Analysis/ # Metrics and Analysis
├── requirements.txt # Project dependencies
└── README.md # Repository documentation
