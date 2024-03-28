""" Contains the functions used across the project.

    @author: mikhail.galkin
"""

# %% Import needed python libraryies and project config info
# import os
# import sys
# import joblib
# import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow

from sklearn import metrics
from mlflow.tracking import MlflowClient
from IPython.display import display
from pprint import pprint


# ------------------------------------------------------------------------------
# ----------------------------- P A R A M E T E R S ----------------------------
# ------------------------------------------------------------------------------
# %% Set up: Pandas options
def pd_set_options():
    """Set parameters for PANDAS to InteractiveWindow"""

    display_settings = {
        "max_columns": 40,
        "max_rows": 400,  # default: 60
        "width": 500,
        "max_info_columns": 500,
        "expand_frame_repr": True,  # Wrap to multiple pages
        "float_format": lambda x: "%.5f" % x,
        "pprint_nest_depth": 4,
        "precision": 4,
        "show_dimensions": True,
    }
    print("\nPandas options established are:")
    for op, value in display_settings.items():
        pd.set_option(f"display.{op}", value)
        option = pd.get_option(f"display.{op}")
        print(f"\tdisplay.{op}: {option}")


# %% Set up: Reset Pandas options
def pd_reset_options():
    """Set parameters for PANDAS to InteractiveWindow"""
    pd.reset_option("all")
    print("Pandas all options re-established.")


# %% Set up: Matplotlib params
def matlotlib_set_params():
    """Set parameters for MATPLOTLIB to InteractiveWindow"""
    from matplotlib import rcParams

    rcParams["figure.figsize"] = 10, 8
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12


# ------------------------------------------------------------------------------
# ---------------------- L O A D I N G    S T U F F ----------------------------
# ------------------------------------------------------------------------------
from fhealth.config import mlflow_tracking_uri
from fhealth.config import model1_path


def model1_load(model1_path):
    # Load model w/ XGBoost python API:
    print(f"\nLoad early trained and saved model...")
    model = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model1_path / "model.xgb")
    model._Booster = booster
    return model


def model1_make_example_for(data_dir, data_file):
    """Create examples for checking model prediction

    Returns:
        examples : list of tupels (X, y) in Pandas DataFrames format
    """
    print(f"\nCreate example df for checking model #1 prediction...")
    cols = [
        "sic",
        # "countryba",
        # "ddate",
        "solvency_debt_ratio",
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",
        "profit_net_margin",
        "profit_roa",
        "active_acp",
        "y",
    ]
    # Get instances
    rows = [0, 999, 4444]
    df = pd.read_csv(
        data_dir / data_file,
        header=0,
        usecols=cols,
        nrows=4500,
    ).iloc[rows, :]

    # With 1 data example
    example_df_1 = df.iloc[[0], :-1]
    y_true_1 = df.iloc[[0], -1].to_list()
    # With 2 data examples
    example_df_2 = df.iloc[[1, 2], :-1]
    y_true_2 = df.iloc[[1, 2], -1].to_list()

    examples = [(example_df_1, y_true_1), (example_df_2, y_true_2)]
    return examples


def model1_load_bins(artifacts_path):
    # Load bins for rating groups
    with open(artifacts_path / "bins.txt") as f:
        bins = f.read()
    bins = bins.replace("[", "").replace("]", "").split()
    bins = [float(b) for b in bins]
    return bins


# ------------------------------------------------------------------------------
# ----------------------------- M E T R I C S ----------------------------------
# ------------------------------------------------------------------------------
def calc_mape(y_true, y_pred):
    """Mean absolute percentage error regression loss.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    0.3273809523809524...

    >>> y_true = np.array([1.0, -1.0, 2, -2])
    >>> y_pred = np.array([0.7, -0.7, 1.4, -1.4])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.30000000000000004...
    """
    # Epsilon: is an arbitrary small yet strictly positive numbe
    # to avoid undefined results when y is zero
    epsilon = np.finfo(np.float64).eps
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(ape)


def calc_binary_class_metrics(model, X, y, set_name=None):
    """https://neptune.ai/blog/evaluation-metrics-binary-classification"""
    print(f"\nCalculate model metrics...")
    model_name = type(model).__name__
    target = y.name
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    if set_name is None:
        set_name = ""
    else:
        set_name = f"-{set_name}"
    # * Metrics
    cm = metrics.confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    auc = metrics.roc_auc_score(y, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)  # True Positive Rate, Sensitivity
    precision = tp / (tp + fp)  # Positive Predictive Value
    f1 = (2 * recall * precision) / (recall + precision)
    avg_prec = metrics.average_precision_score(y, y_pred)
    log_loss = metrics.log_loss(y, y_pred)
    """Brier score is a measure of how far your predictions lie from the true values"""
    brier = metrics.brier_score_loss(y, y_pred_proba)
    """Cohen Kappa Metric tells you how much better is your model over the
    random classifier that predicts based on class frequencies."""
    kappa = metrics.cohen_kappa_score(y, y_pred)
    """Matthews Correlation Coefficient MCC It’s a correlation between
    predicted classes and ground truth"""
    mcc = metrics.matthews_corrcoef(y, y_pred)
    fp_rate = fp / (fp + tn)  # False Positive Rate, Type I error
    fn_rate = fn / (tp + fn)  # False Negative Rate, Type II error
    tn_rate = tn / (tn + fp)  # True Negative Rate, Specificity
    neg_pv = tn / (tn + fn)  # Negative Predictive Value
    fal_dr = fp / (tp + fp)  # False Discovery Rate

    print(f"On {set_name} set the {model_name} for [{target}] gives following metrics:")
    print(f"\tAUC:: {auc:.4f}")
    print(f"\tAccuracy:: {acc:.4f}")
    print(f"\tRecall:: {recall:.4f}")
    print(f"\tPrecision:: {precision:.4f}")
    print(f"\tF1:: {f1:.4f}")
    print(f"\tAverage precision:: {avg_prec:.4f}")
    print(f"\tLog loss:: {log_loss:.4f}")
    print(f"\tBrier score:: {brier:.4f}")
    print(f"\tCohen Kappa metric:: {kappa:.4f}")
    print(f"\tMatthews Correlation Coefficient:: {mcc:.4f}")

    eval_scores = {
        f"auc{set_name}": auc,
        f"acc{set_name}": acc,
        f"recall{set_name}": recall,
        f"precision{set_name}": precision,
        f"f1{set_name}": f1,
        f"avg_precision{set_name}": avg_prec,
        f"log_loss{set_name}": log_loss,
        f"brier{set_name}": brier,
        f"kappa{set_name}": kappa,
        f"mcc{set_name}": mcc,
        f"cm_tn{set_name}": tn,
        f"cm_fp{set_name}": fp,
        f"cm_fn{set_name}": fn,
        f"cm_tp{set_name}": tp,
        f"cm_fp_rate{set_name}": fp_rate,
        f"cm_fn_rate{set_name}": fn_rate,
        f"cm_tn_rate{set_name}": tn_rate,
        f"neg_pv{set_name}": neg_pv,
        f"false_dr{set_name}": fal_dr,
    }
    return eval_scores


def model1_get_rating(y_proba1, bins):
    if y_proba1 < bins[0]:
        rating_desc = "Critical"
        rating = "D"
        group = "D"
    elif y_proba1 < bins[1]:
        rating_desc = "Bad"
        rating = "C"
        group = "C"
    elif y_proba1 < bins[2]:
        rating_desc = "Adverse"
        rating = "+C"
        group = "C"
    elif y_proba1 < bins[3]:
        rating_desc = "Unsatisfactory"
        rating = "++C"
        group = "C"
    elif y_proba1 < bins[4]:
        rating_desc = "Satisfactory"
        rating = "B"
        group = "B"
    elif y_proba1 < bins[5]:
        rating_desc = "Normal"
        rating = "+B"
        group = "B"
    elif y_proba1 < bins[6]:
        rating_desc = "Positive"
        rating = "++B"
        group = "B"
    elif y_proba1 < bins[7]:
        rating_desc = "Good"
        rating = "A"
        group = "A"
    elif y_proba1 < bins[8]:
        rating_desc = "Very good"
        rating = "+A"
        group = "A"
    else:
        rating_desc = "Excellent"
        rating = "++A"
        group = "A"
    return group, rating, rating_desc


def calc_df_scores(model, X_train, y_train, X_test=None, y_test=None):
    if X_test is not None:
        y_proba1_test = model.predict_proba(X_test)[:, 1]
        test_score = pd.DataFrame(
            {
                "data": "Test",
                "y_proba1": y_proba1_test,
                "score": np.rint(y_proba1_test * 1000),
                "true_label": y_test,
                "sic": X_test["sic"],
            }
        )
        ds = "Train"
    else:
        test_score = None
        ds = "Dev"

    y_proba1_train = model.predict_proba(X_train)[:, 1]
    train_score = pd.DataFrame(
        {
            "data": ds,
            "y_proba1": y_proba1_train,
            "score": np.rint(y_proba1_train * 1000),
            "true_label": y_train,
            "sic": X_train["sic"],
        }
    )
    # Make only df
    scores = pd.concat([train_score, test_score], ignore_index=True)
    # Get bins boundaries
    _, bins = pd.qcut(y_proba1_train, q=10, retbins=True, precision=10)
    bins = bins[1:-1]
    # Get ratings
    scores[["group", "rating", "rating_desc"]] = scores["y_proba1"].apply(
        lambda x: pd.Series(model1_get_rating(x, bins))
    )
    return scores, bins


def calc_df_percentile(scores):
    # Calculate grouped DF
    pt = scores.groupby(["data", "group", "rating", "rating_desc"]).agg(
        {"score": ["max", "mean", "min"], "true_label": ["count", "sum"]}
    )
    # Rename columns: levels apriory are sorted
    pt.columns.set_levels(
        [["score", "true_label"], ["count", "max", "mean", "min", "1-good"]],
        inplace=True,
    )
    # Calculate percentage of Rating in grouped Data
    _count = pt.iloc[:, pt.columns.get_level_values(1) == "count"]  # aux df
    pt["score", "percent"] = _count.groupby(level=0).apply(lambda x: 100 * x / x.sum())

    # Calculate count of "0" true labels (bad)
    _good = pt.iloc[:, pt.columns.get_level_values(1).isin(["count", "1-good"])]  # aux df
    pt["true_label", "0-bad"] = _good.apply(lambda x: x[0] - x[1], axis=1)
    pt = pt.sort_index(axis=1)
    return pt


# ------------------------------------------------------------------------------
# -----------------C O L U M N S   M A N I P U L A T I O N S -------------------
# ------------------------------------------------------------------------------
def cols_get_mixed_dtypes(df):
    print(f"Get info about columns w/ mixed types...")
    mixed_dtypes = {
        c: dtype
        for c in df.columns
        if (dtype := pd.api.types.infer_dtype(df[c])).startswith("mixed")
    }
    pprint(mixed_dtypes)
    mixed_dtypes = list(mixed_dtypes.keys())
    return mixed_dtypes


def raws_drop_duplicats(df):
    """Drop fully duplicated rows"""
    n = df.duplicated().sum()  # Number of duplicated rows
    df.drop_duplicates(keep="first", inplace=True)
    df = df.reset_index(drop=True)
    print(f"Droped {n} duplicated rows")
    return df


def cols_reorder(df):
    cols = df.columns.to_list()
    cols.sort()
    df = df[cols]
    return df


def cols_get_na(df):
    """Get: Info about NA's in df"""
    print(f"\n#NA = {df.isna().sum().sum()}\n%NA = {df.isna().sum().sum()/df.size*100}")
    # View NA's through variables
    df_na = pd.concat(
        [df.isna().sum(), df.isna().sum() / len(df) * 100, df.notna().sum()],
        axis=1,
        keys=["# NA", "% NA", "# ~NA"],
    ).sort_values("% NA")
    return df_na


def df_get_glimpse(df, n_rows=4):
    """Returns a glimpse for related DF

    Args:
        df ([type]): original Pandas DataFrame
        rnd_n_rows (int, optional): # randomly picked up rows to show.
            Defaults to 4.
    """
    # %% Get first info about data set
    print(f"\nGET GLIMPSE: ---------------------------------------------------")
    print(f"Count of duplicated rows : {df.duplicated().sum()}")
    print(f"\n----DF: Information about:")
    display(df.info(verbose=True, show_counts=True, memory_usage=True))
    print(f"\n----DF: Descriptive statistics:")
    display(df.describe(include=None).round(3).T)
    print(f"\n----DF: %Missing data:")
    display(cols_get_na(df))
    if n_rows is not None:
        print(f"\n----DF: Random {n_rows} rows:")
        display(df.sample(n=n_rows).T)


def cols_coerce_to_num(df, cols_to_num):
    if cols_to_num is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to numeric...")
        df[cols_to_num] = df[cols_to_num].apply(pd.to_numeric, errors="coerce")
    return df


def cols_coerce_to_str(df, cols_to_str):
    if cols_to_str is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to string...")
        df[cols_to_str] = df[cols_to_str].astype(str)
        # Replace 'nan' with NaN
        df[cols_to_str] = df[cols_to_str].replace({"nan": np.nan})
    return df


def cols_coerce_to_datetime(df, cols_to_datetime):
    if cols_to_datetime is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to datetime...")
        for col in cols_to_datetime:
            print(f"Coerse to_datetime: {col}")
            df[col] = pd.to_datetime(df[col], errors="raise")
    return df


def cols_cat_to_dummies(df, cols_to_dummies):
    print(f"\nConvert categorical to pd.dummies (OHE)...")
    for col in cols_to_dummies:
        print(f"\tConvert column: {col}")
        dummy = pd.get_dummies(df[[col]])
        df = df.drop(columns=[col])
        df = df.merge(dummy, left_index=True, right_index=True)
    return df


# ------------------------------------------------------------------------------
# -------------------------- U T I L I T I E S ---------------------------------
# ------------------------------------------------------------------------------
def outliers_get_zscores(df, cols_to_check=None, sigma=3):
    print(f"\nGet columns w/ outliers w/ {sigma}-sigma...")
    cols_to_drop = ["sic", "cik", "countryba", "name", "ddate"]
    if cols_to_check is None:
        cols_to_check = df.columns.drop(cols_to_drop).tolist()
    else:
        cols_to_check = cols_to_check
    cols = []
    nums = []
    df_out = None
    for col in cols_to_check:
        mean = df[col].mean()
        std = df[col].std()
        z = np.abs(df[col] - mean) > (sigma * std)
        num_outliers = z.sum()

        if num_outliers > 0:
            print(f"\t{col}: {num_outliers} ouliers.")
            display(df.loc[z, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, z], axis=1)
    display(df_out.sum())
    return df_out


def outliers_get_quantiles(df, cols_to_check=None, treshold=0.999):
    print(f"\nGet columns w/ outliers w/ {treshold} quantile treshold...")
    if cols_to_check is None:
        cols_to_check = df.columns.drop("imo").tolist()
    else:
        cols_to_check = cols_to_check
    cols = []
    nums = []
    df_out = None
    for col in cols_to_check:
        cutoff = df[col].quantile(treshold)
        q = df[col] > cutoff
        num_outliers = q.sum()

        if num_outliers > 0:
            print(f"\t{col}: cutoff = {cutoff}: {num_outliers} ouliers.")
            display(df.loc[q, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, q], axis=1)
    display(df_out.sum())
    return df_out


def outliers_ridoff(df, df_out):
    print(f"\nRid off outliers via z-score..")
    idx = df_out.sum(axis=1) > 0
    df = df[~idx]
    print(f"Totally deleted {sum(idx)} outliers...")
    print(f"Data w/o outliers has: {len(df)} rows X {len(df.columns)} cols")
    return df


# ------------------------------------------------------------------------------
# ----------------------------- M L   F L O W ----------------------------------
# ------------------------------------------------------------------------------
def mlflow_get_run_data(run_id):
    import time

    """Fetch params, metrics, tags, and artifacts
    in the specified run for MLflow Tracking
    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    start_time = client.get_run(run_id).info.start_time
    start_time = time.localtime(start_time / 1000.0)
    run_data = (data.params, data.metrics, tags, artifacts, start_time)
    return run_data


def mlflow_del_default_experiment():
    print(f"\nDelete 'Default' experiment from MLflow loggs...")
    try:
        default_id = mlflow.get_experiment_by_name("Default").experiment_id
        default_loc = mlflow.get_experiment_by_name("Default").artifact_location
        mlflow.delete_experiment(default_id)
        print(f"'Default' experiment located:'{default_loc}' was deleted.\n")
    except Exception as e:
        print(f"'Default' experiment doesnt exist: {e}\n")
    return True
