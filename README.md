

# A Bayesian Semiparametric Framework for Factual and Counterfactual Time-Varying Case Fatality Rate Estimation

This repository contains the complete source code and simulation framework for the manuscript, "A Bayesian Semiparametric Framework for Factual and Counterfactual Time-Varying Case Fatality Rate Estimation." The paper introduces a robust Bayesian semiparametric model to estimate the time-varying Case Fatality Rate (CFR) while accounting for reporting delays and the confounding effects of non-pharmaceutical interventions (NPIs).

***

## Key Features

* **Bayesian Semiparametric Mode**l: A flexible model that separates smooth baseline trends from sharp intervention shocks.

* **Causal Inference**: Enables the estimation of counterfactual outcomes (what would have happened without NPIs).

* **Theoretical Guarantees**: The estimator is supported by a posterior contraction rate theorem, ensuring statistical consistency.

* **Simulation & Analysis Pipeline**: Includes a complete, parallelized pipeline for running Monte Carlo simulations and analyzing results.

* **Real Data Application**: Code to replicate the analysis of COVID-19 data in the UK.

  ***

## Getting Started

### Prerequisites

* Python 3.8+

* JAX

* NumPyro

* pandas

* scikit-learn

* statsmodels

* matplotlib

* joblib

* tqdm

### Running the Simulation Study

The entire simulation study can be executed by running the main script `Simulation.ipynb`. This will perform all Monte Carlo runs for the 12 scenarios defined in `config.py` and save the results to the `simulation_outputs/ directory`.

### Analyzing the Results

After the simulation is complete, you can use the `Simu_Data_Analysis.ipynb` notebook to load the saved results, generate the summary plots (as seen in the manuscript), and create the final LaTeX tables.

### UK Real Data Application

The analysis of the UK COVID-19 data can be reproduced by running the cells in the `UK_Analysis.ipynb` notebook. Please ensure the `WHO-COVID-19-global-daily-data.csv` dataset is in the root directory.

***

## Contact

For any questions, comments, or suggestions, please feel free to contact the first author or corresponding author:

* Hengtao Zhang: zhanght@gdou.edu.cn

* Yuanke Qu: quxiaoke@gdou.edu.cn

***

## Citation

If you use this code or model in your research, please cite our manuscript:

```latex
@article{zhang2025scfr,
  title={A Bayesian Semiparametric Framework for Factual and Counterfactual Time-Varying Case Fatality Rate Estimation},
  author={Zhang, Hengtao and Lee, Chun Yin and Qu, Yuanke},
  year={2025},
  journal={Working Paper}
}
```

