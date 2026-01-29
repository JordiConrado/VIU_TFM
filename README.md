# Structural Analysis of the Networks Formed by Wikipedia Editors

This repository contains the code developed as part of the **Master’s Thesis (TFM)** entitled:

**“Structural Analysis of the Networks Formed by Wikipedia Editors”**

The main objective of this project is to analyze the **structure of collaboration networks** that emerge among Wikipedia editors using tools from **graph theory and complex network analysis**, with particular emphasis on community detection and structural pattern identification.

The code is made publicly available in order to **ensure the reproducibility** of the results presented in the Master’s Thesis.

---

## Repository Contents

The general structure of the repository is as follows:

```text
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
The code is organized into two main parts: **preprocessing** and **analysis**.
1. **Preprocessing**: This part includes scripts for cleaning and preparing the data, as well as creating the collaboration graphs. Navigate to the `src/preprocessing/` directory and run the relevant scripts in the specified order.
2. **Analysis**: This part contains scripts for computing network metrics, detecting communities, and visualizing the results. Navigate to the `src/analysis/` directory and execute the analysis scripts as needed.
3. **Data Storage**: Intermediate and final data files (e.g., CSVs, graphs, plots) are stored in the `data/` directory, organized into subdirectories for easy access.
4. **Configuration**: Some scripts may require configuration parameters (e.g., file paths, thresholds). Ensure to modify these parameters at the beginning of each script as needed.
5. **Execution Order**: It is recommended to follow the order of scripts as outlined in the thesis to ensure proper data flow and results.
6. **Documentation**: Each script contains comments and documentation to guide users through its functionality and usage.
7. **Reproducibility**: To reproduce the results presented in the thesis, ensure that you use the same input data and follow the specified execution order.
8. **Support**: For any questions or issues related to the code, please refer to the documentation within the scripts or contact the author.
9. **License**: The code is provided under the MIT License. Feel free to use and modify it for academic and research purposes, ensuring proper attribution.
10. **Updates**: This repository may be updated periodically. Check back for new features, bug fixes, or improvements.
---
## Author
- **Name**: Jordi Conrado 
---
## Data Source
The data used in this project was obtained from Wikipedia dumps located at: https://dumps.wikimedia.org/
Due to size constraints, the actual data files are not included in this repository. Please download the necessary dumps directly from the provided link and store them locally
in the appropriate directories before running the the first script.
---
## Academic Context
This project was completed as part of the requirements for the Master’s Degree in Artificial Intelligence at **Valencian International University (VIU)**.
This code was developed for academic purposes only and it is not intended for production use.
---
## Citation
If you use this code or the results from the associated Master’s Thesis in your research, please cite it as follows:
Conrado, J. (2026). Structural analysis of the networks formed by Wikipedia editors [Source code]. GitHub. https://github.com/usuario/VIU_TFM
---
## Acknowledgments
I would like to thank my thesis advisor and the faculty at Valencian International University for their support and guidance throughout this project.
## License
This project is licensed under the MIT License.

For more details, see the [LICENSE](LICENSE) file.
---
## Contact
For any questions or inquiries regarding this project, please contact the author via GitHub.

