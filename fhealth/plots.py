""" Contains the functions for plotting some results used across the project.

    @author: mikhail.galkin
"""

# %% Import needed python libraryies and project config info
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from IPython.display import display

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


# ------------------------------------------------------------------------------
# ----------------------------- P L O T S --------------------------------------
# ------------------------------------------------------------------------------


def plot_cm(model_name, y_true, y_pred, set_name=None, normalize=True):
    """Plot Confusion Matrix from predictions and true labels
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"Confusion Matrix for {model_name}"
    else:
        title = f"Confusion Matrix for {model_name} on {set_name} set"

    skplt.metrics.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize,
        title=title,
        title_fontsize="small",
        text_fontsize="medium",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_roc_curve(model_name, y_true, y_prob, set_name=None):
    """Plot the ROC curves from labels and predicted scores/probabilities
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"Roc Curve for {model_name}"
    else:
        title = f"Roc Curve for {model_name} on {set_name} set"

    skplt.metrics.plot_roc(
        y_true=y_true,
        y_probas=y_prob,
        title=title,
        title_fontsize="small",
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_prec_recall(model_name, y_true, y_prob, set_name=None):
    """Plot Precision Recall Curve from labels and probabilities
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"Precision Recall Curve for {model_name}"
    else:
        title = f"Precision Recall Curve for {model_name} on {set_name} set"

    skplt.metrics.plot_precision_recall(
        y_true=y_true,
        y_probas=y_prob,
        title=title,
        title_fontsize="small",
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_ks_stat(model_name, y_true, y_prob, set_name=None):
    """Plot the KS Statistic plot from labels and scores/probabilities
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"KS Statistic Plot for {model_name}"
    else:
        title = f"KS Statistic Plot for {model_name} on {set_name} set"

    skplt.metrics.plot_ks_statistic(
        y_true=y_true,
        y_probas=y_prob,
        title=title,
        title_fontsize="small",
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_cum_gain(model_name, y_true, y_prob, set_name=None):
    """Plot the Cumulative Gains Plot from labels and scores/probabilities
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"Cumulative Gains Curve for {model_name}"
    else:
        title = f"Cumulative Gains Curve for {model_name} on {set_name} set"

    skplt.metrics.plot_cumulative_gain(
        y_true=y_true,
        y_probas=y_prob,
        title=title,
        title_fontsize="small",
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_lift(model_name, y_true, y_prob, set_name=None):
    """Plot the Lift Curve from labels and scores/probabilities
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = f"Lift Curve for {model_name}"
    else:
        title = f"Lift Curve for {model_name} on {set_name} set"

    skplt.metrics.plot_lift_curve(
        y_true=y_true,
        y_probas=y_prob,
        title=title,
        title_fontsize="small",
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_calibration(models_tuple, X_val, y_val, set_name=None):
    """Plots calibration curves for a set of classifier probability estimates.
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    if set_name is None:
        title = "Calibration plots (Reliability Curves)"
    else:
        title = f"Calibration plots (Reliability Curves) on {set_name} set"

    clf_names = []
    probas_list = []

    for model in models_tuple:
        if model is None:
            pass
        else:
            model_name = type(model).__name__
            y_probas = model.predict_proba(X_val)
            clf_names.append(model_name)
            probas_list.append(y_probas)

    skplt.metrics.plot_calibration_curve(
        y_true=y_val,
        probas_list=probas_list,
        clf_names=clf_names,
        title=title,
        title_fontsize="small",
        n_bins=10,
        text_fontsize=14,
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_learning(model, X, y, set_name=None):
    """Plots calibration curves for a set of classifier probability estimates.
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    model_name = type(model).__name__
    if set_name is None:
        title = f"Learning Curve for {model_name}"
    else:
        title = f"Learning Curve for {model_name} on {set_name} set"

    skplt.estimators.plot_learning_curve(
        clf=model,
        X=X,
        y=y,
        title=title,
        title_fontsize="small",
        cv=5,
        shuffle=True,
        scoring="roc_auc",
        random_state=42,
        text_fontsize="small",
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_feat_imp(model, feature_names=None, set_name=None, max_num_features=20):
    """Plots calibration curves for a set of classifier probability estimates.
    https://scikit-plot.readthedocs.io/en/stable/metrics.html
    """
    model_name = type(model).__name__
    if set_name is None:
        title = f"Feature Importance of {model_name}"
    else:
        title = f"Feature Importance of {model_name} on {set_name} set"

    skplt.estimators.plot_feature_importances(
        clf=model,
        feature_names=feature_names,
        max_num_features=max_num_features,
        title=title,
        title_fontsize="small",
        x_tick_rotation=45,
        figsize=(10, 8),
    )
    fig = plt.gcf()
    plt.close()
    return fig


def plot_learning_xgb(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)

    # plot classification error
    fig_err, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_axis, results["validation_0"]["error"], label="Train")
    ax.plot(x_axis, results["validation_1"]["error"], label="Test")
    ax.legend()
    plt.ylabel("Classification Error")
    plt.xlabel("n_estimators")
    plt.title("XGBoost Classification Error")
    display(fig_err)
    fig_err = plt.gcf()
    plt.close()

    # plot auc
    fig_auc, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_axis, results["validation_0"]["auc"], label="Train")
    ax.plot(x_axis, results["validation_1"]["auc"], label="Test")
    ax.legend()
    plt.ylabel("Classification AUC")
    plt.xlabel("n_estimators")
    plt.title("XGBoost Classification AUC")
    display(fig_auc)
    fig_auc = plt.gcf()
    plt.close()

    # plot log loss
    fig_log, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
    ax.legend()
    plt.ylabel("Log Loss")
    plt.xlabel("n_estimators")
    plt.title("XGBoost Log Loss")
    display(fig_log)
    fig_log = plt.gcf()
    plt.close()
    return (fig_err, fig_auc, fig_log)


def plot_scores_dist(model_name, scores, bins=10, is_scores=True):
    # plot scores disctibution for data sets
    title = f"{model_name}: Data Scores' distributions"
    # fig = plt.figure()
    # sns.set(font_scale=1.4)
    sns.displot(
        scores,
        bins=bins,
        x="score",
        hue="data",
        element="step",
        stat="probability",
        common_norm=False,
        height=10,
        aspect=1.2,
        legend=True,
    )
    plt.title(title)
    if is_scores:
        plt.xticks(range(0, 1001, round(1000 / bins)))
    fig_data = plt.gcf()
    display(fig_data)
    plt.close()

    # plot scores disctibution for class_labels
    sns.displot(
        scores,
        col="data",
        bins=bins,
        x="score",
        hue="true_label",
        element="step",
        stat="probability",
        common_norm=False,
        height=10,
        aspect=1.2,
        legend=True,
        rug=True,
    )
    if is_scores:
        plt.xticks(range(0, 1001, round(1000 / bins)))
    fig_label = plt.gcf()
    display(fig_label)
    plt.close()

    return (fig_data, fig_label)
