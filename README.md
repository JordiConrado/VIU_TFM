# Structural Analysis of the Networks Formed by Wikipedia Editors

This repository contains the code developed as part of the **Master’s Thesis** entitled:

**“Structural Analysis of the Networks Formed by Wikipedia Editors”**

The main objective of this project is to analyze the **structure of collaboration networks** that emerge among Wikipedia editors using tools from **graph theory and complex network analysis**, with particular emphasis on community detection and structural pattern identification.

The code is made publicly available in order to **ensure the reproducibility** of the results presented in the Master’s Thesis.

---

## Repository Contents

The general structure of the repository is as follows:

```text
VIU_TFM/
├── data/               # Input data (raw and/or preprocessed)
│ ├── 01_wiki/          # Wikipedia-related files (e.g., list of bots)
│ ├── 02_csv/           # Temporary CSV files generated during processing
│ ├── 03_graph/         # Graph data created with NetworkX
│ └── 01_plots/         # Plots and visualizations
├── src/                # Main source code
│ ├── preprocessing/    # Data cleaning, preparation, and graph creation
│ └── analysis/         # Network metrics and analysis
├── requirements.txt    # Project dependencies
└── README.md           # Repository documentation

```

*(The structure may slightly differ from the final version used in the thesis.)*

---

## ⚙️ Requirements and Environment

The code was developed and tested using:

- **Python 3.x**
- Main libraries:
  - `networkx`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

To install the required dependencies:

```bash
pip install -r requirements.txt
```

---
## Running the Code
The project is organized into two main components:

1. **Preprocessing** (`src/preprocessing/`): scripts for data cleaning, preparation, and construction of collaboration networks.
2. **Analysis** (`src/analysis/`): scripts for computing network metrics, community detection, and result visualization.

Intermediate and final outputs (e.g., CSV files, graphs, and plots) are stored in the `data/` directory, organized into subdirectories.

To reproduce the results presented in the thesis, it is recommended to follow the execution order described in the thesis and ensure that the same input data are used.

## Author
Jordi Conrado 
---
## Data Source
The data used in this project was obtained from Wikipedia dumps located at: https://dumps.wikimedia.org/
Due to size constraints, the actual data files are not included in this repository. Please download the necessary dumps directly from the provided link and store them locally
in the appropriate directories before running the first script.
---
## Academic Context
This project was completed as part of the requirements for the Master’s Degree in Artificial Intelligence at **Universidad Internacional de Valencia (VIU)**.
This code was developed for academic purposes only and it is not intended for production use.
---
## Citation
If you use this code or the results from the associated Master’s Thesis in your research, please cite it as follows:
_Conrado, J. (2026). Structural analysis of the networks formed by Wikipedia editors [Source code]. GitHub. https://github.com/usuario/VIU_TFM_
---
## Acknowledgments
I would like to thank my thesis advisor and the faculty at Universidad Internacional de Valencia (VIU) for their support and guidance throughout this project.
## License
This project is licensed under the MIT License.

For more details, see the [LICENSE](LICENSE) file.
---
## Contact
For any questions or inquiries regarding this project, please contact the author via GitHub.

