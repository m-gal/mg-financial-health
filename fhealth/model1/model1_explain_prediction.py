"""
    Use SHAP values to explain individual XGBoost classifier predictions.

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import warnings
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.config import model1_path
from fhealth.utils import model1_load
from fhealth.utils import model1_load_bins
from fhealth.utils import model1_get_rating
from fhealth.model1.model1_explain_entire import transform_shap_values


# %% Main -----------------------------------------------------------------------
def main(
    model1_path=model1_path,
    save_plots=False,
):
    """Use the SHAP package to explain the XGB prediction.

    Args:
        * model1_path (str): Folder's name for separatly saved model.\
            Defaults to None.
    """
    #! Input data need to be explained
    # name = "AAPL"
    # input_data = pd.DataFrame.from_dict(
    #     {
    #         "sic": [7372],
    #         "solvency_debt_ratio": [4.1],
    #         "liquid_current_ratio": [1],
    #         "liquid_quick_ratio": [1],
    #         "liquid_cash_ratio": [0.2],
    #         "profit_net_margin": [0.25],
    #         "profit_roa": [0.25],
    #         "active_acp": [15],
    #     }
    # )

    # name = "TESLA"
    # input_data = pd.DataFrame.from_dict(
    #     {
    #         "sic": [3711],
    #         "solvency_debt_ratio": [0.25],
    #         "liquid_current_ratio": [1.8],
    #         "liquid_quick_ratio": [1.2],
    #         "liquid_cash_ratio": [1.4],
    #         "profit_net_margin": [0.1],
    #         "profit_roa": [0.05],
    #         "active_acp": [17],
    #     }
    # )

    # * JO-ANN STORES INC: August 1, 2020 https://www.sec.gov/edgar/browse/?CIK=34151
    name = "JO-ANN STORES"
    input_data = pd.DataFrame.from_dict(
        {
            "sic": [5940],
            "solvency_debt_ratio": [0.943030605],
            "liquid_current_ratio": [1.252651386],
            "liquid_quick_ratio": [0.171570304],
            "liquid_cash_ratio": [0.037803626],
            "profit_net_margin": [0.018948941],
            "profit_roa": [0.00844145],
            "active_acp": [19.7591711],
        }
    )

    print(f"-------------- START: Explain the model prediction ---------------")
    print(f"Model path: {model1_path}")
    warnings.filterwarnings("ignore")
    artifacts_path = Path(model1_path).parent
    plots_path = Path(model1_path).parent / "plots/shap"

    try:
        plots_path.mkdir(parents=True)
    except:
        pass

    # Load model w/ XGBoost python API:
    model = model1_load(model1_path)
    # Load bins for rating groups
    bins = model1_load_bins(artifacts_path)

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

    # Include new instance into data set
    X_new = pd.concat([input_data, X], ignore_index=True)
    y_proba1_new = model.predict_proba(X_new)[:, 1]
    # Score for instance
    group, rating, rating_desc = model1_get_rating(y_proba1_new[0], bins)
    inst_score = round(y_proba1_new[0] * 1000)
    inst_rate = f"Rating: {rating} - Quality: {rating_desc}"
    title = f"{name}: Score predicted: {inst_score}"

    print(title)
    print(inst_rate)

    # ^ Explain predictions for others plots -----------------------------------
    # ^ Explain predictions
    explainer1 = shap.Explainer(model)
    shap_values1 = explainer1(X_new)
    # Get transfomed shap values to predicted values
    shap_values_transformed1 = transform_shap_values(shap_values1, y_proba1_new)

    """SHAP Waterfall Plot for an individual case"""
    shap.plots.waterfall(
        shap_values_transformed1[0],
        max_display=10,
        show=False,
    )
    fig = plt.gcf()
    fig.set_figwidth(11)
    fig.set_figheight(7)
    plt.title(f"{title} - {inst_rate}")
    display(fig)
    if save_plots:
        fig.savefig(plots_path / f"waterfall_plot_{name}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    # """SHAP Model Decisions Plot"""
    # expected_value1 = explainer1.expected_value
    # feature_names1 = X_new.columns.tolist()
    # shap.decision_plot(
    #     base_value=expected_value1,
    #     shap_values=explainer1.shap_values(X_new)[0],
    #     features=X_new.iloc[0],
    #     feature_names=feature_names1,
    #     feature_order="importance",
    #     title=f"Model's decisions plot. {title} - {inst_rate}",
    #     link="logit",
    #     show=False,
    # )
    # fig = plt.gcf()
    # fig.set_figwidth(10)
    # display(fig)
    # # fig.savefig(plots_path / f"decisions_plot_{name}.png", bbox_inches="tight")
    # plt.clf()
    # plt.close()


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %%
