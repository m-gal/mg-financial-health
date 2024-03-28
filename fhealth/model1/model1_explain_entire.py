"""
    Use SHAP values to understand XGBoost classifier predictions.

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import warnings
import random
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from IPython.display import display

# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.config import model1_path
from fhealth.utils import model1_load


def transform_shap_values(shap_values, y_proba1):
    """Compute the transformed base value,
    which consists in applying the logit function to the base value
    """
    import numpy as np

    # Importing the logit function for the base value transformation
    # untransformed_base_value = shap_values.base_values[-1]
    untransformed_base_value = sum(shap_values.base_values) / len(shap_values.base_values)
    base_value = 1 / (1 + np.exp(-untransformed_base_value))

    # Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(shap_values.values, axis=1)

    # Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = y_proba1 - base_value

    # The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain

    # Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / np.expand_dims(distance_coefficient, -1)

    # Finally resetting the base_value as it does not need to be transformed
    # shap_values_transformed.base_values = base_value
    shap_values_transformed.base_values = np.array(
        [base_value] * len(shap_values.base_values), dtype=np.float32
    )
    shap_values_transformed.data = shap_values.data

    # Now returning the transformed array
    return shap_values_transformed


# %% Main -----------------------------------------------------------------------
def main(
    model1_path=model1_path,
    save_plots=False,
):
    """Use the SHAP package to explain the XGB classifier.

    Args:
        * model1_path (str): Folder's name for separatly saved model.\
            Defaults to None.
    """
    print(f"--------------- START: Explain the model prediction ---------------")
    warnings.filterwarnings("ignore")
    # random.seed(42)

    plots_path = Path(model1_path).parent / "plots/shap"
    try:
        plots_path.mkdir(parents=True)
    except:
        pass

    # Load model w/ XGBoost python API:
    model = model1_load(model1_path)
    print(f"Model path: {model1_path}")

    # Get data
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

    # Get probability prediction
    y_proba1 = model.predict_proba(X)[:, 1]

    # Get three randon instansec from differ ranges
    which = random.randint(0, len(y))
    which_proba = y_proba1[which]
    which_title = f"case #{which} w/ score {round(which_proba*1000)}"

    idx_low = list(np.where(y_proba1 < 0.45)[0])
    lower = idx_low[random.randint(0, len(idx_low))]

    idx_up = list(np.where(y_proba1 > 0.55)[0])
    upper = idx_up[random.randint(0, len(idx_up))]

    # ^ Explain predictions for force plots ------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # """ Visualize a single prediction
    # """
    # plt.style.use(["default"])
    # shap.initjs()
    # shap.force_plot(
    #     base_value=explainer.expected_value,
    #     shap_values=shap_values[which, :],
    #     features=X.iloc[which, :],
    #     matplotlib=True,
    #     figsize=(20, 9),
    #     show=False,  # for gettin non-empty figure purpose
    #     text_rotation=90,
    #     link="logit",  # to get predicted score but not shap-value
    # )
    # fig = plt.gcf()
    # plt.title(f"Score prediction interpretability for individual {which_title}")
    # if save_plots:
    #     fig.savefig(plots_path / "force_plot_single.png", bbox_inches="tight")
    # display(fig)
    # plt.clf()  # clear the current figure
    # plt.close()

    # """ Visualize many predictions
    # """
    # shap.initjs()
    # fp = shap.force_plot(
    #     base_value=explainer.expected_value,
    #     shap_values=shap_values[:which, :],
    #     features=X.iloc[:which, :],
    #     link="logit",
    # )
    # display(fp)
    # if save_plots:
    #     shap.save_html(str(plots_path) + "/force_plot_multiple.html", fp)

    # ^ Explain predictions ----------------------------------------------------
    """ SHAP Dependence Plots:
    SHAP dependence plots show the effect of a single feature across the whole dataset.
    They plot a feature’s value vs. the SHAP value of that feature across many samples.
    SHAP dependence plots are similar to partial dependence plots, but account for the
    interaction effects present in the features, and are only defined in regions of the
    input space supported by data. The vertical dispersion of SHAP values at
    a single feature value is driven by interaction effects, and another feature is chosen
    for coloring to highlight possible interactions."""
    for feature in X.columns:
        shap.dependence_plot(
            feature,
            shap_values,
            X,
            display_features=X,
            show=False,
        )
        fig = plt.gcf()
        fig.set_figwidth(10)
        fig.set_figheight(6)
        plt.title(f"Dependence plot for {feature}")
        if save_plots:
            fig.savefig(
                plots_path / f"dependence_plot_{feature}.png",
                bbox_inches="tight",
            )
        display(fig)
        plt.clf()
        plt.close()

    # ^ Explain predictions for others plots -----------------------------------
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # Get transfomed shap values to predicted values
    shap_values_transformed = transform_shap_values(shap_values, y_proba1)

    """SHAP Summary Plot:
    Rather than use a typical feature importance bar chart,
    we use a density scatter plot of SHAP values for each feature to identify
    how much impact each feature has on the model output for individuals
    in the validation dataset. Features are sorted by the sum of the SHAP value
    magnitudes across all samples. It is interesting to note that the relationship
    feature has more total model impact than the captial gain feature,
    but for those samples where capital gain matters it has more impact than age.
    In other words, capital gain effects a few predictions by a large amount,
    while age effects all predictions by a smaller amount.
    Note that when the scatter points don’t fit on a line they pile up to show density,
    and the color of each point represents the feature value of that individual.
    """
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_size=(12, 8),
    )
    fig = plt.gcf()
    if save_plots:
        fig.savefig(plots_path / "summary_plot.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Bar Chart of Absolute Mean Importance:
    This takes the abs average of the SHAP value magnitudes across the dataset
    and plots it as a simple bar chart.
    """
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False,  # for gettin non-empty figure purpose
        plot_size=(12, 8),
    )
    fig = plt.gcf()
    if save_plots:
        fig.savefig(plots_path / "summary_plot_bar_abs.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Bar Chart of Mean Importance w/ direction:
    This takes the average of the SHAP value magnitudes across the dataset
    and plots it as a simple bar chart.
    """
    shap.plots.bar(shap_values.mean(0), show=False)
    fig = plt.gcf()
    fig.set_figwidth(11)
    fig.set_figheight(7)
    plt.tight_layout()
    if save_plots:
        fig.savefig(plots_path / "summary_plot_bar_avg.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Waterfall Plot for an individual case
    """
    shap.plots.waterfall(
        shap_values_transformed[which],
        max_display=10,
        show=False,
    )
    fig = plt.gcf()
    fig.set_figwidth(11)
    fig.set_figheight(7)
    plt.title(f"Score prediction making for individual {which_title}")
    if save_plots:
        fig.savefig(plots_path / f"waterfall_plot_{which}.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Heatmap Plot
    A heatmap visualizes the magnitude of data in two dimensions.
    The variation in color can give visual cues about how the data are clustered.
    You can use the hot-to-cold color scheme to show the relationships.
    The variable importance is shown in descending order in the Y-axis,
    with the bars resembling the bars in the bar chart.
    The f(x) curve on the top is the model predictions for the instances.
    The SHAP first runs a hierarchical clustering on the instances to cluster them,
    then orders the instances (using shap.order.hclust) on the X-axis.
    The center of the 2D heatmap is the base_value (using .base_value),
    which is the mean prediction for all instances.
    """
    shap.plots.heatmap(
        shap_values[:1000],
        show=False,
    )
    fig = plt.gcf()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    if save_plots:
        fig.savefig(plots_path / f"heatmap_plot.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Model Decisions Plot using cumulative SHAP values for an individual case
    When there are many predictors, a force plot may become busy and not present them well.
    A decision plot will be a good choice.
    The numbers in parenthesis are the predictor(variable) values in model.
    The bottom of a decision plot is the base value. The line heads to the left or right
    given the forces of the predictors.
    """
    expected_value = explainer.expected_value
    feature_names = X.columns.tolist()
    shap.decision_plot(
        base_value=expected_value,
        shap_values=explainer.shap_values(X)[which],
        features=X.iloc[which],
        feature_names=feature_names,
        feature_order="importance",
        title=f"Model decisions plot for individual {which_title}",
        link="logit",
        show=False,
    )
    fig = plt.gcf()
    fig.set_figwidth(10)
    # fig.set_figheight(6)
    if save_plots:
        fig.savefig(plots_path / f"decisions_plot_{which}.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """SHAP Model Decisions Plot for the three observation
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    # Plot decision for lower
    ax1 = fig.add_subplot(gs[0, 0])
    shap.decision_plot(
        base_value=expected_value,
        shap_values=explainer.shap_values(X)[lower],
        features=X.iloc[lower],
        feature_names=feature_names,
        feature_order=None,
        link="logit",
        show=False,
    )
    ax1.title.set_text(f"Observation w/ score {round(y_proba1[lower]*1000)}")

    # Plot decision for which
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    shap.decision_plot(
        base_value=expected_value,
        shap_values=explainer.shap_values(X)[which],
        features=X.iloc[which],
        feature_names=feature_names,
        feature_order=None,
        link="logit",
        show=False,
    )
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.title.set_text(f"{which_title}")

    # Plot decision for upper
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    shap.decision_plot(
        base_value=expected_value,
        shap_values=explainer.shap_values(X)[upper],
        features=X.iloc[upper],
        feature_names=feature_names,
        feature_order=None,
        link="logit",
        show=False,
    )
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.title.set_text(f"Observation w/ score {round(y_proba1[upper]*1000)}")

    fig = plt.gcf()
    fig.set_figwidth(14)
    plt.tight_layout()
    if save_plots:
        fig.savefig(plots_path / f"decisions_plot_3cases.png", bbox_inches="tight")
    display(fig)
    plt.clf()
    plt.close()

    """Identify outliers
    """

    N = 2000
    rows_rnd = random.sample(range(0, len(df)), N)
    X_rnd = df.iloc[rows_rnd, :-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values_rnd = explainer.shap_values(X)[rows_rnd]
        shap.decision_plot(
            base_value=expected_value,
            shap_values=shap_values_rnd,
            features=X_rnd,
            feature_names=feature_names,
            feature_order="hclust",
            return_objects=True,
            link="logit",
            show=False,
        )
        fig = plt.gcf()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.title(f"Decision plot for random 2000 instances")
        if save_plots:
            fig.savefig(plots_path / f"decisions_plot_2000.png", bbox_inches="tight")
        display(fig)
        plt.clf()
        plt.close()

    print(f"!!! DONE: Model explaination !!!")


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %%
