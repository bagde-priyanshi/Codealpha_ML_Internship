# Credit Scoring Model
### CodeAlpha ML Internship — Task 1

---

## Objective
Predict whether a customer will default on a loan using their past financial data.
This is a **binary classification** problem — output is either:
- `0` → Will NOT default (safe)
- `1` → Will default (risky)

---

## Dataset
**Give Me Some Credit** — Kaggle  
- 150,000 rows, 12 columns
- Target column: `SeriousDlqin2yrs`

| Column | Description |
|---|---|
| `SeriousDlqin2yrs` | Target — 1 if defaulted, 0 if not |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card usage ratio |
| `age` | Age of the borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Late payments (30–59 days) |
| `DebtRatio` | Debt as a fraction of income |
| `MonthlyIncome` | Monthly earnings |
| `NumberOfOpenCreditLinesAndLoans` | Total open credit lines |
| `NumberOfTimes90DaysLate` | Severely late payments |
| `NumberRealEstateLoansOrLines` | Real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Late payments (60–89 days) |
| `NumberOfDependents` | Family members dependent on borrower |

---

## Project Structure
```
Codealpha_task1/
│
├── cs-training.csv       # Dataset
├── credit_scoring.py     # Main model code
└── README.md             # This file
```


---

## Approach

### 1. Data Cleaning
- Dropped `Unnamed: 0` (useless row index)
- Filled missing `MonthlyIncome` values with **median** (safer than mean for skewed income data)
- Filled missing `NumberOfDependents` values with **median**

### 2. Train/Test Split
- 80% training data, 20% test data
- `random_state=42` for reproducibility

### 3. Models Used
- **Logistic Regression** — baseline model (draws a linear decision boundary)
- **Random Forest** — ensemble of decision trees (crowd voting approach)
- Both used `class_weight='balanced'` to handle class imbalance

### 4. Why `class_weight='balanced'`?
The dataset is heavily imbalanced:
- 93% → did NOT default
- 7% → defaulted

Without balancing, the model simply predicts "no default" for everyone and gets 94% accuracy — but catches almost 0% of actual defaulters. `class_weight='balanced'` forces the model to pay more attention to the minority class (defaulters).

---

## Results

| Model | Accuracy | Recall (Defaulters) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (balanced) ✅ | 78% | 0.65 | 0.71 |
| Decision Tree (balanced) | 90% | 0.24 | 0.59 |
| Random Forest (balanced) | 94% | 0.16 | 0.57 |

### Best Model: Logistic Regression with `class_weight='balanced'`

---

## Key Metrics Explained

| Metric | What it means |
|---|---|
| **Precision** | Of all predicted defaulters, how many actually defaulted? |
| **Recall** | Of all actual defaulters, how many did the model catch? |
| **F1-Score** | Balance between Precision and Recall |
| **ROC-AUC** | Overall ability to separate defaulters from non-defaulters (0.5 = random, 1.0 = perfect) |

> **Recall matters most here** — missing a defaulter costs the bank real money.

---

## How to Run

1. Make sure you have the required libraries:
```bash
pip install pandas scikit-learn
```

2. Place `cs-training.csv` in the same folder as `credit_scoring.py`

3. Run:
```bash
python credit_scoring.py
```

---

## Libraries Used
- `pandas` — data loading and cleaning
- `scikit-learn` — model training and evaluation
