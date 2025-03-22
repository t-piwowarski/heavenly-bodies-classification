from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, tags, title):
    """
    Plots a confusion matrix as a heatmap.
    
    Args:
        cm (ndarray): Confusion matrix.
        tags (list): Class labels (e.g. ['GALAXY', 'STAR', 'QSO']).
        title (str): Title of the plot.
    """
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Blues',
                xticklabels=tags, yticklabels=tags)
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_measures(name, TP, TN, FP, FN):
    """
    Calculates and prints detailed metrics for one class:
    - Sensitivity (Recall)
    - Accuracy (specific to the class)
    - Precision
    - Specificity

    These are calculated using:
    - TP: True Positives
    - TN: True Negatives
    - FP: False Positives
    - FN: False Negatives
    """
    sensitivity = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    specificity = TN / (FP + TN)
    print(f'\n{name}:\n  Sensitivity: {sensitivity:.3f}\n  Accuracy: {accuracy:.3f}\n  Precision: {precision:.3f}\n  Specificity: {specificity:.3f}')

def evaluate_model(y_true, y_pred, tags, model_name):
    """
    Full evaluation of a classification model:
    - Prints confusion matrix
    - Calculates TP, TN, FP, FN for each class
    - Uses calculate_measures() for detailed metrics per class

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        tags: List of class names
        model_name: Name of the model (for printing)
    """
    print(f"\n=== {model_name.upper()} ===")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, tags, f"Confusion Matrix - {model_name}")

    # Calculate TP, FN, FP, TN for each class (multi-class logic)
    TP = [cm[i, i] for i in range(3)]
    FN = [sum(cm[i, j] for j in range(3) if j != i) for i in range(3)]
    FP = [sum(cm[j, i] for j in range(3) if j != i) for i in range(3)]
    TN = [
        sum(cm[j, k] for j in range(3) for k in range(3)
            if j != i and k != i)
        for i in range(3)
    ]
    
    # Print measures for each class
    for i, label in enumerate(tags):
        calculate_measures(label, TP[i], TN[i], FP[i], FN[i])
