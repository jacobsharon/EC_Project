# EC_Project

## Explainable Multi-Objective Genetic Programming for Chronic Kidney Disease Classification Using NSGA-II

This project uses Genetic Programming combined with NSGA-II and SHAP analysis to optimize the classification of Chronic Kidney Disease (CKD) while maintaining interpretability. A multi-objective approach is used to balance classification performance and model simplicity. SHAP analysis is incorporated to provide feature-level explanations, yielding a final model that is both accurate and clinically interpretable.

---

### Project Structure

- `main.py` – Entry point: runs the evolutionary loop, evaluates individuals, and generates output
- `README.md` – Project overview and usage guide
- `LICENSE` – License file
- `requirements.txt` – Python dependencies
- `settings.py` – Global constants (e.g., population size, generations)

#### data/
- `cleaned_ckd_dataset.csv` – Cleaned dataset after preprocessing
- `raw_ckd_dataset.csv` – Original dataset
- `default_attribute_values` – Reference values used for data cleaning and imputation

#### classification_model/
- `classification_model.py` – Classification model and train/test data preparation

#### genetic_programming/
- `creator_setup.py` – DEAP creator for individuals and fitness
- `evaluate_population.py` – Fitness evaluation (F1 score and complexity)
- `initialize.py` – Population initialization logic
- `primitives.py` – DEAP Primitive and Terminal set definitions
- `toolbox.py` – DEAP Genetic Operators (crossover, mutation, selection)

#### explain/
- `SHAP_analysis.py` – SHAP-based feature importance analysis

#### results/
- Auto-generated output directory (not version-controlled).
- `visualize_tree.py` - Script for rendering trees using Graphviz

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
- Enforce clinically meaningful feature constraints
- Explain evolved models using SHAP

## References
- UCI CKD Dataset: https://doi.org/10.24432/C5G020
- NSGA-II: Deb et al., 2002
- SHAP: Lundberg & Lee, 2017 (https://github.com/slundberg/shap)

## Author
Jacob Sharon, BS, MLS(ASCP)<sup>CM</sup>  
Graduate Student, Computer Science  
Missouri State University
