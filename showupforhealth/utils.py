# Import Librraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from datetime import datetime

# Import ShowUp params
from showupforhealth.params import *


def define_X_y(df, target):
    target = target

    X = df.drop(columns=target)
    y = df[target]

    print(f'X - independant variable shape: {X.shape}')
    print(f'y - dependant variable - {target}: {y.shape}')

    return X, y

def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=42):
    # Calculate intermediate size based on test_size
    intermediate_size = 1 - test_size

    # Calculate train_size from intermediate size and validation size
    train_size = 1 - val_size / intermediate_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=train_size, random_state=random_state)

    print(f"✅ OUTPUT: X_train, X_val, X_test, y_train, y_val, y_test")
    print(f"Train Set:  X_train, y_train - {X_train.shape}, {y_train.shape}")
    print(f"  Val Set:  X_val, y_val - - - {X_val.shape}, {y_val.shape}")
    print(f" Test Set:  X_test, y_test - - {X_test.shape}, {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def automl_tpot(X, y):
    # Select features and target
    features = X
    target = y

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # Create a tpot object with a few generations and population size.
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

    # Fit the tpot model on the training data
    tpot.fit(X_train, y_train)

    # Show the final model
    print(tpot.fitted_pipeline_)

    # Use the fitted model to make predictions on the test dataset
    test_predictions = tpot.predict(X_test)

    # Evaluate the model
    print(tpot.score(X_test, y_test))

    # Export the pipeline as a python script file
    time = datetime().now()
    tpot.export(f'{METRICS_OUT}tpot_pipeline_{time}.csv')
                

def feature_importance(model, X, y):
    """
    Displays the feature importances of a model using permutation importance.

    Parameters:
    - model: estimator instance. The model to evaluate.
    - X: DataFrame. The feature matrix.
    - y: Series. The target vector.

    Returns:
    - Permutation importance plot
    """
    # Train the model
    model.fit(X, y)

    # Calculate permutation importance
    result = permutation_importance(model, X, y, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()

    # Permutation importance plot
    plt.figure(figsize=(10, 5))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importances")
    plt.show()