from src.preprocessing import load_and_clean_data, scale_features, balance_classes
from src.modeling import train_knn, train_dtc, train_rfc
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Define the class labels (used in reports and confusion matrix)
tags = ['GALAXY', 'STAR', 'QSO']

# === Step 1: Load and clean the dataset ===
# Loads CSV, encodes classes as numbers (Galaxy=0...), and drops unnecessary columns
df = load_and_clean_data('data/star_classification.csv')

# Separate features and target label
X = df.drop('class', axis=1)
y = df['class']

# === Step 2: Split data into training and test sets ===
# We stratify to keep the same class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

# === Step 3: Scale features ===
# Standardize the features (important for KNN)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# === Step 4: Balance the training data (for KNN only) ===
# Undersample Galaxy and oversample QSO + Star so each class has ~25000 samples
X_balanced, y_balanced = balance_classes(X_train_scaled, y_train)

# === Step 5: Train models ===
# KNN is trained on balanced data
knn = train_knn(X_balanced, y_balanced)

# Decision Tree and Random Forest are trained on original (unbalanced) data,
# but with class_weight='balanced' to help compensate for imbalance
dtc = train_dtc(X_train_scaled, y_train)
rfc = train_rfc(X_train_scaled, y_train)

# === Step 6: Evaluate all models on the same test set ===
# Outputs confusion matrix and per-class metrics (TP, FP, TN, FN)
evaluate_model(y_test, knn.predict(X_test_scaled), tags, "KNN")
evaluate_model(y_test, dtc.predict(X_test_scaled), tags, "Decision Tree")
evaluate_model(y_test, rfc.predict(X_test_scaled), tags, "Random Forest")

