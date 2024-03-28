# %% Load libraries -------------------------------------------------------------
import sys

# import warnings
# import shap
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.utils import model1_load
from fhealth.utils import calc_df_scores
from fhealth.utils import calc_df_percentile
from fhealth.plots import plot_scores_dist


# %%
def main(model1_path):

    # Load model w/ XGBoost python API:
    model = model1_load(model1_path)
    model_name = type(model).__name__
    artifacts_path = Path(model1_path).parent

    # Get development data
    df = pd.read_csv(
        data_processed_dir / "model1_dev.csv",
        header=0,
        usecols=[
            "sic",
            "solvency_debt_ratio",
            "liquid_current_ratio",
            "liquid_quick_ratio",
            "liquid_cash_ratio",
            "profit_net_margin",
            "profit_roa",
            "active_acp",
            "y",
        ],
    )
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    y_proba1 = model.predict_proba(X)[:, 1]
    my_scores, bins = calc_df_scores(model, X, y)
    plot_scores_dist(model_name, my_scores, bins=20)
    pt = calc_df_percentile(my_scores)
    display(pt)
    pt.to_html(artifacts_path / "percentile_table.html")


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        model1_path="../mlflow/mlruns/2/e0c5bab2a1b24a99a89b2f9e3ca9b471/artifacts/model",
    )

# %%
