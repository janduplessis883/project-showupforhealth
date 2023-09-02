# Import Libraries 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
from IPython.display import display, HTML, Javascript

# SKlearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.inspection import permutation_importance
from tpot import TPOTClassifier

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
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f'pipelines/tpot_pipeline_{time}.csv'
    output_path = os.path.join(root_dir, output_dir)
    tpot.export(output_path)


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

def evaluate_classification_model(model, X, y, cv=5):
    """
    Evaluates the performance of a model using cross-validation, a learning curve, and a ROC curve.

    Parameters:
    - model: estimator instance. The model to evaluate.
    - X: DataFrame. The feature matrix.
    - y: Series. The target vector.
    - cv: int, default=5. The number of cross-validation folds.

    Returns:
    - None
    """
    print(model)
    # Cross validation
    scoring = {'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1_score': make_scorer(f1_score, average='macro')}

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    # Compute means and standard deviations for each metric, and collect in a dictionary
    mean_std_scores = {metric: (np.mean(score_array), np.std(score_array)) for metric, score_array in scores.items()}

    # Create a DataFrame from the mean and std dictionary and display as HTML
    scores_df = pd.DataFrame(mean_std_scores, index=['Mean', 'Standard Deviation']).T
    display(HTML(scores_df.to_html()))

    # Learning curve
    train_sizes=np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Define the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(train_sizes, train_scores_mean, 'o-', color="#a10606", label="Training score")
    axs[0].plot(train_sizes, test_scores_mean, 'o-', color="#6b8550", label="Cross-validation score")
    axs[0].set_xlabel("Training examples")
    axs[0].set_ylabel("Score")
    axs[0].legend(loc="best")
    axs[0].set_title("Learning curve")

    # ROC curve
    cv = StratifiedKFold(n_splits=cv)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[train], y.iloc[train])
        viz = plot_roc_curve(model, X.iloc[test], y.iloc[test],
                            name='ROC fold {}'.format(i),
                            alpha=0.3, lw=1, ax=axs[1])
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    axs[1].plot(mean_fpr, mean_tpr, color='#023e8a',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.6)

    axs[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='#a10606',
            label='Chance', alpha=.6)
    axs[1].legend(loc="lower right")
    axs[1].set_title("ROC curve")

    # Show plots
    plt.tight_layout()
    plt.show()


# Permutation feature importance
def feature_importance(self, model, X, y):
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