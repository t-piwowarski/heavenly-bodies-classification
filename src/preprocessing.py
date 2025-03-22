import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    """
    Loads the dataset and does some basic cleaning:
    - Converts class labels (Galaxy, Star, QSO) into numbers (0, 1, 2)
    - Drops columns that don’t help with classification (like IDs and metadata)
    
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)
    df['class'] = pd.factorize(df['class'])[0]
    df.drop(['obj_ID','run_ID','rerun_ID','field_ID','fiber_ID',
             'spec_obj_ID', 'plate', 'MJD', 'cam_col'], axis=1, inplace=True)
    return df

def scale_features(X_train, X_test):
    """
    Scales the features using StandardScaler (mean = 0, std = 1).
    This is important for models like KNN that depend on distance.
    
    Returns:
        Scaled versions of X_train and X_test
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def balance_classes(X, y, target_count=25000):
    """
    Balances the dataset so that each class has the same number of samples.
    - Downsamples the majority class (Galaxy) to target_count
    - Upsamples the minority classes (Star and QSO) to target_count using SMOTE
    
    Returns:
        Resampled feature set and labels
    """
    # First, reduce the number of Galaxy samples
    under = RandomUnderSampler(sampling_strategy={0: target_count}, random_state=42)
    
    # Then, generate more samples for QSO and Star
    over = SMOTE(sampling_strategy={1: target_count, 2: target_count}, random_state=42)
    
    # Create a pipeline to apply both steps
    pipeline = Pipeline(steps=[('under', under), ('over', over)])
    
    return pipeline.fit_resample(X, y)
