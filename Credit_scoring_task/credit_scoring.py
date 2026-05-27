import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Step 1: Loading data...")
df = pd.read_csv('cs-training.csv')
print("\nPreview the dataset")
print(f"Shape: {df.shape}")
print(df.head())
print(df.info())

# ── 2. Clean Data ─────────────────────────────────────────────────────────────
print("\nStep 2: Cleaning data...")

# Drop useless row index column
df = df.drop(columns=['Unnamed: 0'])

# Fill missing values with median (safer than mean for skewed income data)
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

print("Done!")

# ── 3. Split Data ─────────────────────────────────────────────────────────────
print("\nStep 3: Splitting data (80% train / 20% test)...")

X = df.drop(columns=['SeriousDlqin2yrs'])   # Features (inputs)
y = df['SeriousDlqin2yrs']                  # Target (what we predict)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Done! Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 4. Model 1: Logistic Regression ──────────────────────────────────────────
print("\n── Model 1: Logistic Regression (Baseline) ──")
print("Training...")

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\nResults:")
print(classification_report(y_test, lr_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_pred):.4f}")

# ── 5. Model 2: Decision Tree ─────────────────────────────────────────────────
print("\n── Model 2: Decision Tree Classifier ──")
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\nResults:")
print(classification_report(y_test, dt_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, dt_pred):.4f}")


# ── 6. Model 3:Random Forest ─────────────────────────────────────────────────
print("\n── Model 3: Random Forest ──")
print("Training (this may take a few minutes)...")

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nResults:")
print(classification_report(y_test, rf_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_pred):.4f}")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n── Summary ──")
print(f"Logistic Regression ROC-AUC : {roc_auc_score(y_test, lr_pred):.4f}")
print(f"Decision Tree ROC-AUC       : {roc_auc_score(y_test, dt_pred):.4f}")
print(f"Random Forest ROC-AUC       : {roc_auc_score(y_test, rf_pred):.4f}")
print("\nBest Model: Logistic Regression (balanced) — higher ROC-AUC and Recall for defaulters")