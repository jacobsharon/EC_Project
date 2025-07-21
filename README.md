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

### `data/`
- `raw_ckd_dataset.csv` – Original dataset  
- `cleaned_ckd_dataset.csv` – Cleaned dataset after preprocessing 
- `violated_rows_with_constraints.csv` - All rows removed during cleaning and imputation. 
- `default_attribute_values/` – Reference values for imputing missing values  

### `classification_model/`
- `classification_model.py` – Train/test data preparation with 5-fold stratified splitting  

### `genetic_programming/`
- `creator_setup.py` – DEAP creator for individuals and fitness  
- `evaluate_population.py` – Fitness evaluation (F1 score and complexity)  
- `initialize.py` – Population initialization logic  
- `primitives.py` – DEAP Primitive and Terminal set definitions  
- `toolbox.py` – DEAP Genetic Operators (crossover, mutation, selection)  

### `explain/`
- `SHAP_analysis.py` – SHAP-based feature importance analysis  

### `results/`
- `final_results/` – Output per fold (models, metrics, SHAP, etc.)  
- `visualize_tree.py` – Script for rendering trees using Graphviz  
- `elbow_locator.py` – Detects elbow point on Pareto front using KneeLocator  

---

## Steps to Run

Follow the steps below to run the project from start to finish.

### 1. Install Dependencies

Recommended environment Python 3.12.4. Then install all required packages using:

```bash
pip install -r requirements.txt
```

### 2. Set Global Parameters

Open the `settings.py` file and configure any global parameters as needed.  
Examples of parameters to set:

- `POP_SIZE`: population size for the genetic algorithm
- `GENERATIONS`: number of generations
- `RANDOM_SEED`: seed for reproducibility

### 3. Run Main Script

Execute the main script to begin training and evaluating symbolic models.  
This process may take a significant amount of time depending on your hardware and parameters:

```bash
python main.py
```

The script will perform:

- 5-fold cross-validation
- Multi-objective optimization using NSGA-II
- Evaluation of best and elbow-point models per fold
- Metric recording and result saving

### 4. Run SHAP Analysis

Once `main.py` finishes executing, run the SHAP analysis to interpret feature importance for the evolved models.  
This step may also take time depending on the complexity of the models:

```bash
python -m SHAP_analysis.py
```

The SHAP script will:

- Load saved models from each fold
- Compute SHAP values per feature
- Generate summary plots and store them in the output directory

## Objectives 
- Develop interpretable GP-based classifiers for CKD detection
- Optimize F1-score and model simplicity via NSGA-II
- Enforce clinically meaningful feature constraints
- Explain evolved models using SHAP

## References
- UCI CKD Dataset: https://doi.org/10.24432/C5G020

## Author
Jacob Sharon, BS, MLS(ASCP)<sup>CM</sup>  
Graduate Student, Computer Science  
Missouri State University
