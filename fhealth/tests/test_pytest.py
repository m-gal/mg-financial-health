# Makes tests w/ the pytest package.

#     Open Terminal in the [./tests] folder and run
#     >pytest
#       or
#     >pytest -p no:warnings


import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])

from fhealth.config import model1_path
from fhealth.utils import model1_load_bins
from fhealth.utils import model1_get_rating
from fhealth.utils import model1_load

from fhealth.predict.model1_check_prediction import (
    main as model1_check_prediction,
)
from fhealth.predict.model1_logged_predict import main as model1_logged_predict
from fhealth.data.process_data_model2 import main as process_data_model2


@pytest.fixture
def model1_path(model1_path=model1_path):
    model1_path = Path(model1_path).resolve().__str__()
    return model1_path


def test_process_data_model2():
    cols = [
        "cik",
        "ddate",
        "seq",
        "seq_rank",
        "country",
        "sic",
        "_01_solvency_debt_ratio",
        "_01_liquid_current_ratio",
        "_01_liquid_quick_ratio",
        "_01_liquid_cash_ratio",
        "_01_profit_net_margin",
        "_01_profit_roa",
        "_01_active_acp",
        "_00_solvency_debt_ratio",
        "_00_liquid_current_ratio",
        "_00_liquid_quick_ratio",
        "_00_liquid_cash_ratio",
        "_00_profit_net_margin",
        "_00_profit_roa",
        "_00_active_acp",
        "yy_solvency_debt_ratio",
        "yy_liquid_current_ratio",
        "yy_liquid_quick_ratio",
        "yy_liquid_cash_ratio",
        "yy_profit_net_margin",
        "yy_profit_roa",
        "yy_active_acp",
        "y_solvency_debt_ratio",
        "y_liquid_current_ratio",
        "y_liquid_quick_ratio",
        "y_liquid_cash_ratio",
        "y_profit_net_margin",
        "y_profit_roa",
        "y_active_acp",
    ]
    df = process_data_model2(make_report=False)
    assert df.shape[1] == 34
    assert df.columns.to_list() == cols


def test_model1_load_bins(model1_path):
    artifacts_path = Path(model1_path).parent
    bins = model1_load_bins(artifacts_path)
    assert len(bins) == 9


def test_model1_load(model1_path):
    model = model1_load(model1_path)
    assert type(model).__name__ == "XGBClassifier"


def test_model1_check_prediction(model1_path):
    prediction = model1_check_prediction(model1_path)
    assert isinstance(prediction, list) == True
    assert len(prediction) == 2
    assert np.mean(prediction) <= 1


def test_model1_logged_prediction():
    prediction = model1_logged_predict()
    assert isinstance(prediction, np.ndarray) == True


def test_model1_get_rating(model1_path):
    artifacts_path = Path(model1_path).parent
    bins = model1_load_bins(artifacts_path)
    prediction = model1_check_prediction(model1_path)
    y_proba1 = prediction[0]
    group, rating, rating_desc = model1_get_rating(y_proba1, bins)
    assert group in ["A", "B", "C", "D"]
    assert rating in ["++A", "+A", "A", "++B", "+B", "B", "++C", "+C", "C", "D"]
    assert rating_desc in [
        "Excellent",
        "Very good",
        "Good",
        "Positive",
        "Normal",
        "Satisfactory",
        "Unsatisfactory",
        "Adverse",
        "Bad",
        "Critical",
    ]
