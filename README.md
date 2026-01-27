# SmartML Dashboard

A Streamlit-based ML platform providing end-to-end capabilities for data preprocessing, model training, evaluation, and batch inference. Designed with modular architecture for maintainability and extensibility.

---

## 1. Project Overview

SmartML Dashboard is a full-stack ML application implementing the following workflows:

- **Data Pipeline**: CSV/Excel ingestion → preprocessing → feature engineering
- **EDA Module**: Statistical analysis with Plotly-based interactive visualizations
- **Training Module**: Multi-algorithm training with optional hyperparameter tuning
- **Evaluation Module**: Cross-model comparison with metrics and feature importance
- **Inference Module**: Batch prediction on new data with CSV export

**Application Type**: Single-page Streamlit application with server-side session state management.

---

## 2. Key Features

### 2.1 Data Layer

| Feature | Implementation |
|---------|----------------|
| File Upload | Streamlit `file_uploader` supporting CSV/Excel via `pandas.read_csv`/`read_excel` |
| Type Detection | `pd.DataFrame.select_dtypes` for numeric vs categorical column classification |
| Missing Value Handling | `sklearn.impute.SimpleImputer` with mean/median/mode strategies |
| Categorical Encoding | One-hot encoding (≤10 categories) via `pd.get_dummies`, label encoding otherwise |
| Normalization | `StandardScaler` (z-score) or `MinMaxScaler` (0-1 range) |

### 2.2 EDA Module

| Feature | Implementation |
|---------|----------------|
| Distributions | `plotly.express.histogram` with marginal box plots |
| Correlations | `plotly.express.imshow` correlation matrix |
| Pairplots | `seaborn.pairplot` (sampled for performance) |
| Scatter Analysis | `plotly.express.scatter` with optional color encoding |

### 2.3 Training Module

**Supported Algorithms**:

| Regression | Classification |
|------------|----------------|
| LinearRegression | LogisticRegression |
| Ridge (L2) | SVC |
| Lasso (L1) | DecisionTreeClassifier |
| SVR | RandomForestClassifier |
| DecisionTreeRegressor | GradientBoostingClassifier |
| RandomForestRegressor | XGBClassifier |
| GradientBoostingRegressor | |
| XGBRegressor | |

**Pipeline Components**:
- `train_test_split` (default 80/20, `random_state=42`)
- `GridSearchCV` for hyperparameter tuning (RF, XGBoost, SVM)
- `cross_val_score` for model validation

### 2.4 Evaluation Module

| Problem Type | Metrics |
|--------------|---------|
| Regression | MAE, MSE, RMSE, R² |
| Classification | Accuracy, Precision, Recall, F1 (weighted), AUC-ROC |

**Visualization Outputs**:
- Confusion matrix (`sklearn.metrics.confusion_matrix`)
- Feature importance via `model.feature_importances_` (tree-based) or `model.coef_` (linear)
- Residual plots for regression diagnostics

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Streamlit 1.22.0+ | UI framework, reactive state management |
| Data | Pandas 1.5.0+ | DataFrame operations, file I/O |
| ML | Scikit-learn 1.2.0+ | Algorithms, preprocessing, metrics |
| Gradient Boosting | XGBoost 1.7.3+ | High-performance tree ensembles |
| Visualization | Plotly 5.11.0+ | Interactive charts |
| Serialization | Joblib 1.2.0+ | Model persistence |

---

## 4. System Architecture

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Upload   │ │ Cleaning │ │  EDA     │ │ Training │       │
│  │ Page     │→│ Page     │→│ Page     │→│ Page     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Session State (st.session_state)            │
│  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌────────────────┐  │
│  │   df   │→ │ processor│→ │ results │→ │ problem_type   │  │
│  └────────┘  └──────────┘  └─────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Utilities                            │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ DataProcessor  │  │ EDAAnalyzer  │  │ ModelTrainer    │  │
│  │ (cleaning,     │  │ (statistical │  │ (training,      │  │
│  │  encoding)     │→ │  analysis)   │→ │  evaluation)    │  │
│  └────────────────┘  └──────────────┘  └─────────────────┘  │
│                              │                               │
│                              ▼                               │
│                      ┌─────────────────┐                     │
│                      │ ModelVisualizer │                     │
│                      │ (metrics, plots)│                     │
│                      └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **Ingestion**: File upload → `pd.read_csv/excel` → `session_state.df`
2. **Preprocessing**: `DataProcessor` instance → transformations → updated DataFrame
3. **Analysis**: `EDAAnalyzer` yields Plotly figures → rendered in Streamlit containers
4. **Training**: `ModelTrainer` → `train_test_split` → model.fit() → metrics computation
5. **Output**: Results stored in `session_state.results` → `ModelVisualizer` generates comparison charts

### 4.3 State Management

```
Session State Schema:
{
    "df": pd.DataFrame,           # Raw/processed data
    "target": str,                # Target column name
    "processor": DataProcessor,   # Preprocessing instance
    "results": dict,              # {model_name: {model, metrics, predictions, ...}}
    "problem_type": str           # "classification" or "regression"
}
```

### 4.4 Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Streamlit | Rapid UI development with native data app support; session state for server-side persistence |
| Modular ML Layer | Separation of concerns enables unit testing and future API extraction |
| Session State | Avoids database overhead for single-session workflows; maintains data across page navigations |
| Plotly | Interactive, zoomable charts with hover data; superior to static matplotlib for EDA |
| Joblib | Efficient sklearn model serialization with numpy array compatibility |

---

## 5. Machine Learning Workflow

### 5.1 Pipeline Stages

```
DATA LOAD → DATA CLEANING → EDA → MODEL TRAINING → EVALUATION
```

### 5.2 Data Processing Pipeline

`ml_utils/data_processing.py` → `DataProcessor`

| Method | Operation |
|--------|-----------|
| `__init__` | Column type detection (numeric/categorical) |
| `handle_missing_values()` | SimpleImputer or row removal |
| `encode_categorical()` | One-hot (≤10 categories) or label encoding |
| `normalize_data()` | StandardScaler or MinMaxScaler |
| `drop_columns()` | Feature removal |
| `get_summary()` | DataFrame summary (types, nulls, uniques) |

### 5.3 Training Pipeline

`ml_utils/modeling.py` → `ModelTrainer`

```
Input: X (features), y (target), problem_type
   │
   ▼
Train-Test Split (80/20, random_state=42)
   │
   ▼
User Model Selection
   │
   ▼
Optional GridSearchCV (RF, XGBoost, SVM)
   │
   ▼
model.fit(X_train, y_train)
   │
   ▼
y_pred = model.predict(X_test)
   │
   ▼
Metrics Calculation
   │
   ▼
Feature Importance Extraction
   │
   ▼
Output: results dict {model, metrics, predictions, train_time, feature_importance}
```

### 5.4 Feature Importance Extraction

| Model Family | Attribute |
|--------------|-----------|
| Tree-based (RF, GBM, XGBoost, DT) | `model.feature_importances_` |
| Linear (Linear, Ridge, Lasso, LogReg) | `model.coef_` (shape-checked) |
| SVM | `None` (no importance attribute) |

---

## 6. Setup & Installation

```bash
git clone https://github.com/Guna-Asher/smartml-pro.git
cd smartml-pro
pip install -r requirements.txt
streamlit run app.py
```

**Configuration** (`config.py`):

```python
MAX_DATASET_SIZE_MB = int(os.getenv("MAX_DATASET_SIZE_MB", "100"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RANDOM_STATE = 42
```

---

## 7. Usage

| Section | Function |
|---------|----------|
| **Data Upload** | File ingestion, target column selection, problem type auto-detection |
| **Data Cleaning** | Missing value imputation, categorical encoding, normalization |
| **Exploratory Analysis** | Distributions, correlations, pairwise relationships |
| **Model Training** | Algorithm selection, hyperparameter tuning, training execution |
| **Model Evaluation** | Metrics comparison, confusion matrices, feature importance |
| **Make Predictions** | Batch inference on new data with CSV export |

---

## 8. Project Structure

```
smartml-pro/
├── app.py                  # Streamlit entry point, session state, page routing
├── config.py               # Configuration singleton
├── requirements.txt        # Dependencies
├── ml_utils/
│   ├── data_processing.py  # DataProcessor class
│   ├── eda.py             # EDAAnalyzer class
│   ├── modeling.py        # ModelTrainer class
│   └── visualization.py   # ModelVisualizer class
└── utils/
    └── logger.py          # Logging configuration
```

---

## 9. Future Improvements

| Priority | Feature |
|----------|---------|
| High | Model persistence (joblib save/load) |
| Medium | FastAPI layer for model serving |
| Medium | MLflow integration for experiment tracking |
| Medium | Pipeline export (sklearn Pipeline serialization) |
| Low | Optuna-based Bayesian hyperparameter tuning |

---

**License**: MIT

**Author**: [Guna Asher](https://github.com/Guna-Asher)

