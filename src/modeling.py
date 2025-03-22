from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_knn(X_train, y_train, n_neighbors=15):
    """
    Trains a K-Nearest Neighbors classifier.
    
    n_neighbors: how many neighbors to look at when predicting a class.
    Works best when data is scaled and balanced.
    
    Returns:
        Trained KNN model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_dtc(X_train, y_train):
    """
    Trains a Decision Tree classifier.
    
    - Uses 'gini' to measure how "pure" the splits are.
    - class_weight='balanced' helps when some classes have more samples than others.
    
    Returns:
        Trained Decision Tree model
    """
    model = DecisionTreeClassifier(criterion='gini',class_weight='balanced', min_samples_split=2)
    model.fit(X_train, y_train)
    return model

def train_rfc(X_train, y_train):
    """
    Trains a Random Forest classifier (a bunch of decision trees combined).
    
    - Also uses 'gini' like the single tree.
    - class_weight='balanced' is important when the dataset is imbalanced.
    
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(criterion='gini', class_weight='balanced', min_samples_split=2)
    model.fit(X_train, y_train)
    return model
