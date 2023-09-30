from showupforhealth.params import *
from showupforhealth.interface.model_predict import *
import pytest


def test_upload_folder_excist():
    assert os.path.exists(f"{UPLOAD_FOLDER}/ECS_predict40.csv")


def test_imd_data_excist():
    assert os.path.exists(f"{IMD_DATA}")

def test_display_outcome_df_duplicates():
    class_labels = np.array([1, 0, 1, 0, 0])
    pt_id_df = pd.DataFrame({"Patient ID": [1, 2, 1, 4, 5]})
    
    result_df = display_outcome_df(class_labels, pt_id_df)
    
    assert result_df.duplicated().sum() == 1  # There is one duplicate entry after dropping duplicates

# Test case 3: Verify the returned DataFrame when there are no predicted no-shows
def test_display_outcome_df_no_show():
    class_labels = np.array([1, 1, 1, 1, 1])
    pt_id_df = pd.DataFrame({"Patient ID": [1, 2, 3, 4, 5]})
    
    result_df = display_outcome_df(class_labels, pt_id_df)
    
    assert len(result_df) == 0  # The DataFrame should be empty
    
def test_f1_score():
    y_true = K.variable(np.array([0, 1, 1, 0, 1]))
    y_pred = K.variable(np.array([0, 1, 1, 1, 0]))
    
    result = f1_score(y_true, y_pred)

    assert K.eval(result) == pytest.approx(0.6666666667, abs=1e-7)

# Test case 2: Verify the F1 score calculation when true positives and possible positives are zero
def test_f1_score_zero_true_positives():
    y_true = K.variable(np.array([0, 0, 0, 0, 0]))
    y_pred = K.variable(np.array([1, 1, 1, 1, 1]))
    
    result = f1_score(y_true, y_pred)

    assert K.eval(result) == pytest.approx(0.0, abs=1e-7)

# Test case 3: Verify the F1 score calculation when predicted positives are zero
def test_f1_score_zero_predicted_positives():
    y_true = K.variable(np.array([1, 1, 1, 1, 1]))
    y_pred = K.variable(np.array([0, 0, 0, 0, 0]))
    
    result = f1_score(y_true, y_pred)

    assert K.eval(result) == pytest.approx(0.0, abs=1e-7)