# EC_Project

## Explainable Multi-Objective Genetic Programming for Chronic Kidney Disease Classification Using NSGA-II

This project uses Genetic Programming combined with NSGA-II and SHAP analysis to optimize the classification of Chronic Kidney Disease (CKD) while maintaining interpretability. A multi-objective approach is used to balance classification performance and model simplicity. SHAP analysis is incorporated to provide feature-level explanations, yielding a final model that is both accurate and clinically interpretable.

---

### Project Structure

- `main.py` – Runs the project
- `config.py` – Global settings and parameters
- `data/preprocessing.py` – Dataset cleaning and imputation
- `genetic_programming/` – Genetic Programming and NSGA-II algorithm implementation
- `explain/SHAP_analysis.py` – SHAP-based feature importance explanation
- `results/` – Output logs, Pareto fronts, and performance plots

---

## Recommended Environment

- Python 3.12.4

---

## Install Dependencies

```bash
pip install -r requirements.tx
```

## Usage
```bash
python main.py
```

## Output
- Pareto front visualization (F1-score vs Complexity)
- SHAP summary plots
- CSV log

## Objectives 
- Develop interpretable GP-based classifiers for CKD detection
- Optimize F1-score and model simplicity via NSGA-II
- Enforce clinically meaningful constraints (e.g., albumin ≠ 0)
- Explain evolved models using SHAP

## References
- UCI CKD Dataset: https://doi.org/10.24432/C5G020
- NSGA-II: Deb et al., 2002
- SHAP: Lundberg & Lee, 2017 (https://github.com/slundberg/shap)

## Author
Jacob Sharon, BS, MLS(ASCP)<sup>CM</sup>
Graduate Student, Computer Science  
Missouri State University