"""
    LOAD preprocessed development dataset for Resulting financial health scoring:
        "model1_dev.csv"
    and SPLIT it onto TRAIN & TEST sets
    with BAYESIAN Cross-Validation Searching algorithm upon model
    FIND the best hyperparams for a model on TRAIN set
    TRAIN ML model w/ hyperparams found out on DEV
    LOG results with MLflow Tracking

    @author: mikhail.galkin
"""

#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit
#                            _
#    _._ _..._ .-',     _.._(`))
#   '-. `     '  /-._.-'    ',/
#      )         \            '.
#     / _    _    |             \
#    |  a    a    /              |
#    \   .-.                     ;
#     '-('' ).-'       ,'       ;
#        '-;           |      .'
#           \           \    /
#           | 7  .__  _.-\   \
#           | |  |  ``/  /`  /
#          /,_|  |   /,_/   /
#            /,_/      '`-'

# %% Load libraries
import os
import sys
import time
import winsound
import warnings

import pandas as pd
import sklearn
import mlflow
import xgboost as xgb

from sklearn import model_selection
from sklearn import metrics
from skopt import BayesSearchCV

from pprint import pprint
from IPython.display import display

# %% Load project"s stuff
os.environ["NUMEXPR_MAX_THREADS"] = "48"
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
os.environ["NUMEXPR_MAX_THREADS"] = "48"

# Load custom classes and utils
from fhealth.config import temp_dir
from fhealth.config import data_processed_dir
from fhealth.config import mlflow_tracking_uri

from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options
from fhealth.utils import calc_binary_class_metrics
from fhealth.utils import calc_df_scores
from fhealth.utils import calc_df_percentile
from fhealth.model1._xgb_class import model as get_model_xgb
from fhealth.model1._xgb_class import param_search as get_param_search_xgb
from fhealth import plots


# %% Print out functions --------------------------------------------------------
def print_versions():
    print(f"Packages versions:")
    print(f"\tPandas: {pd.__version__}")
    print(f"\tScikit-learn: {sklearn.__version__}")
    print(f"\tXGBoost: {xgb.__version__}")
    print(f"\tMLflow: {mlflow.__version__}")


def print_model(model_name, model, model_params):
    print(f" ")
    print(f"model_name: {model_name}")
    pprint(model)
    print(f"model_params:")
    pprint(model_params)


# %% Custom funcs ---------------------------------------------------------------
def get_data(df, target):
    print(f"\nGet train & test data with the target [{target}]:")
    rnd_state = 42
    y_dev = df.pop(target)
    X_dev = df
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_dev,
        y_dev,
        train_size=0.8,
        random_state=rnd_state,
        shuffle=True,
    )
    print(f"X_train: {X_train.shape[1]} variables & {X_train.shape[0]} records.")
    print(f"X_test: {X_test.shape[1]} variables & {X_test.shape[0]} records.")
    return X_dev, y_dev, X_train, y_train, X_test, y_test


def log_model_plots(model, X, y, set_name=None):
    print(f"\nPlot and log model's resulting plots...")
    model_name = type(model).__name__
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    if set_name is None:
        preff = ""
    else:
        preff = f"{set_name}-"

    fig_cm = plots.plot_cm(model_name, y, y_pred, set_name, normalize=False)
    mlflow.log_figure(fig_cm, f"./plots/{preff}confusion_matrix.png")
    display(fig_cm)

    fig_cm_norm = plots.plot_cm(model_name, y, y_pred, set_name, normalize=True)
    mlflow.log_figure(fig_cm_norm, f"./plots/{preff}confusion_matrix_norm.png")
    display(fig_cm_norm)

    fig_roc = plots.plot_roc_curve(model_name, y, y_prob, set_name)
    mlflow.log_figure(fig_roc, f"./plots/{preff}roc_curve.png")
    display(fig_roc)

    fig_pr = plots.plot_prec_recall(model_name, y, y_prob, set_name)
    mlflow.log_figure(fig_pr, f"./plots/{preff}precision_recall.png")
    display(fig_pr)

    fig_ks = plots.plot_ks_stat(model_name, y, y_prob, set_name)
    mlflow.log_figure(fig_ks, f"./plots/{preff}ks_statistic.png")
    display(fig_ks)

    fig_cg = plots.plot_cum_gain(model_name, y, y_prob, set_name)
    mlflow.log_figure(fig_cg, f"./plots/{preff}cumulative_gains.png")
    display(fig_cg)

    fig_lf = plots.plot_lift(model_name, y, y_prob, set_name)
    mlflow.log_figure(fig_lf, f"./plots/{preff}lift_curve.png")
    display(fig_lf)

    fig_cl = plots.plot_calibration((model, None), X, y, set_name)
    mlflow.log_figure(fig_cl, f"./plots/{preff}calibration.png")
    display(fig_cl)


# %% ------------------- Bayesian Optimization Module----------------------------
def bayesian_search_cv(
    model_pack_initial,
    X_train,
    y_train,
    bayes_space,
    n_bayes_params_sets=50,
    n_splits=5,
    log_model=False,
    tracking_uri=mlflow_tracking_uri,
):
    rnd_state = 42
    print(f"\n---------- Bayesian optimization of hyper-params started....")
    print(f"Parameters' space:")
    pprint(bayes_space)
    model = model_pack_initial[0]
    model_name = model_pack_initial[1]
    target = y_train.name

    # Define metrics
    scoring = metrics.get_scorer("roc_auc")
    # Define num. of CV splits and K-repeats
    cv = model_selection.RepeatedKFold(
        n_splits=n_splits,
        n_repeats=1,  #  n times with different randomization in each repetition.
        random_state=rnd_state,
    )
    # Define bayesian space search
    bayes_search = BayesSearchCV(
        model,
        search_spaces=bayes_space,
        n_iter=n_bayes_params_sets,
        scoring=scoring,
        cv=cv,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=3,
        random_state=rnd_state,
    )

    # Callback handler
    def on_step(optim_result):
        """Print scores after each iteration while performing optimization"""
        score = bayes_search.best_score_
        print(f"\nCurrent best score: {score} ...")

    # * MLflow: Setup tracking
    tracking_uri = tracking_uri
    experiment_name = "Model#1:Search"
    run_name = f"{model_name}-{target}: bayes"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # * MLflow: Enable autologging
    if model_name[:3] == "XGB":
        mlflow.xgboost.autolog(log_input_examples=False, log_models=log_model)
    else:
        mlflow.sklearn.autolog(log_input_examples=False, log_models=log_model)

    # * MLflow: Fit model with MLflow logging
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        print(f"\nMLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
        print(f"MLFLOW_ARTIFACT_URI: {mlflow.get_artifact_uri()}")
        print(f"MLFLOW_EXPERIMENT_NAME: {experiment_name}")
        print(f"EXPERIMENT_ID: {experiment_id}")
        print(f"RUN_NAME: {run_name}")
        print(f"RUN_ID: {run_id}\n")

        mlflow.set_tag("dev.period", "2019-03-31")

        tic = time.time()
        model_bayes_search = bayes_search.fit(
            X_train,
            y_train,
            callback=on_step,
        )
        min, sec = divmod(time.time() - tic, 60)
        print(f"Bayesian search took: {int(min)}min {int(sec)}sec")

        # * Disable autologging
        if model_name[:3] == "XGB":
            mlflow.xgboost.autolog(disable=True)
        else:
            mlflow.sklearn.autolog(disable=True)

        # * Log results
        print(f"\nLog metrics and model's results...")
        mlflow.log_metric("best_CV_score", bayes_search.best_score_)
        my_metrics = calc_binary_class_metrics(model_bayes_search, X_train, y_train)
        mlflow.log_metrics(my_metrics)

        # * Log params
        search_params = {"n_iter": n_bayes_params_sets, "n_splits": n_splits}
        mlflow.log_params(model_bayes_search.best_params_)
        mlflow.log_params(search_params)

        # * log artifacts
        mlflow.log_dict(dict(model_bayes_search.best_params_), "best_params.json")

        # * Log plots
        log_model_plots(model_bayes_search, X_train, y_train, "train")

    print(f"\nBayesian search: Best score is:\n\t {model_bayes_search.best_score_}")
    print(f"Bayesian search: Best params are:\n\t")
    pprint(model_bayes_search.best_params_)

    model_pack_opted = (
        model_bayes_search.best_estimator_,
        dict(model_bayes_search.best_params_),
    )
    winsound.Beep(frequency=2000, duration=300)
    return model_pack_opted, (experiment_id, run_id)


def train_model(
    model_pack_opted,
    X_train,
    y_train,
    X_test,
    y_test,
    early_stop=False,
    log_model=False,
    log_input=False,
    set_name="train",
    tracking_uri=mlflow_tracking_uri,
):
    model = model_pack_opted[0]
    model_name = type(model_pack_opted[0]).__name__
    model_params = model_pack_opted[1]
    target = y_train.name
    input_example = X_train.iloc[[0, -1]]  # first and last rows
    # to prevent overfitting parameters
    fit_params = {
        "early_stopping_rounds": 100,
        "eval_metric": ["error", "auc", "logloss"],
        "eval_set": [(X_train, y_train), (X_test, y_test)],
        "verbose": False,
    }

    print(f"\nTrain the {model_name} model for [{target}]...")
    print(f"Train dataset: X={X_train.shape}, y={y_train.shape}")
    print(f"Model_params:")
    pprint(model_params)

    # * MLflow: Setup tracking
    tracking_uri = tracking_uri
    experiment_name = "Model#1:Train"
    run_name = f"{model_name}-{target}: {set_name}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # * MLflow: Enable autologging
    if model_name[:3] == "XGB":
        mlflow.xgboost.autolog(log_input_examples=log_input, log_models=log_model)
    else:
        mlflow.sklearn.autolog(log_input_examples=log_input, log_models=log_model)

    # * MLflow: Fit model with MLflow logging
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        print(f"\nMLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
        print(f"MLFLOW_ARTIFACT_URI: {mlflow.get_artifact_uri()}")
        print(f"MLFLOW_EXPERIMENT_NAME: {experiment_name}")
        print(f"EXPERIMENT_ID: {experiment_id}")
        print(f"RUN_NAME: {run_name}")
        print(f"RUN_ID: {run_id}\n")

        mlflow.set_tag("dev.period", "2019-03-31")

        tic = time.time()
        if early_stop:
            model = model.fit(X_train, y_train, **fit_params)
            model_params["n_estimators"] = model.best_iteration
            figs_lc = plots.plot_learning_xgb(model)
            mlflow.log_figure(figs_lc[0], "./plots/train-xgb_lc_error.png")
            mlflow.log_figure(figs_lc[1], "./plots/train-xgb_lc_auc.png")
            mlflow.log_figure(figs_lc[2], "./plots/train-xgb_lc_logloss.png")
        else:
            model = model.fit(X_train, y_train)
        min, sec = divmod(time.time() - tic, 60)
        print(f"Model training on {set_name} took: {int(min)}min {int(sec)}sec")

        # * Log a XGBoost model
        if model_name[:3] == "XGB":
            if log_model:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    input_example=input_example,
                )
            # * Disable autologging
            mlflow.xgboost.autolog(disable=True)
        else:
            mlflow.sklearn.autolog(disable=True)

        # * Log results
        mlflow.log_params(model_params)
        mlflow.log_dict(model_params, "model_params.json")
        # Feature importance
        fig_imp = plots.plot_feat_imp(model, X_train.columns, set_name=set_name)
        mlflow.log_figure(fig_imp, "./plots/feature_imp.png")
        display(fig_imp)

        my_metrics = calc_binary_class_metrics(model, X_train, y_train, set_name=set_name)
        mlflow.log_metrics(my_metrics)
        log_model_plots(model, X_train, y_train, set_name=set_name)

        if X_test is not None and y_test is not None:
            set_name = "test"
            my_metrics = calc_binary_class_metrics(model, X_test, y_test, set_name=set_name)
            mlflow.log_metrics(my_metrics)
            log_model_plots(model, X_test, y_test, set_name=set_name)

            # Scores' distributions
            my_scores, bins = calc_df_scores(model, X_train, y_train, X_test, y_test)
        else:
            my_scores, bins = calc_df_scores(model, X_train, y_train)

        # * Log bins
        binsstr = str(bins).replace("\n", "")
        mlflow.log_text(binsstr, "bins.txt")

        figs_score = plots.plot_scores_dist(model_name, my_scores)
        mlflow.log_figure(figs_score[0], "./plots/scores_data_dist.png")
        mlflow.log_figure(figs_score[1], "./plots/scores_label_dist.png")

        # * Logg percentile table
        pt = calc_df_percentile(my_scores)
        display(pt)
        pt.to_html(temp_dir / "percentile_table.html")
        mlflow.log_artifact(temp_dir / "percentile_table.html")

    return (
        (model, model_params),
        (experiment_id, run_id),
        (my_metrics, my_scores),
    )


# %% Main =======================================================================
def main(
    data_file,
    n_bayes_params_sets,
    n_splits,
):
    """Performs FITTING MODELS on DEVELOPMENT set with searching an
    optimal hyper-parameters with Baeysian search.

    Args:
        file_to_load (str): name of csv file w/ processed development data
        target (str): target to predict
        * n_bayes_params_sets (int, optional): Number of parameter settings that are sampled.\
            Defaults to 20.
        * n_splits (int, optional): Num. of splits in Cross-Validation strategy.\
            Defaults to 5.
    """
    warnings.filterwarnings("ignore")
    print(f"---- START: General Financial Health Score's Model Training ----")
    pd_set_options()
    print_versions()

    # * Data
    dev_periods = [
        "2019-03-31",
        "2019-06-30",
        "2019-09-30",
        "2019-12-31",
        "2020-03-31",
        "2020-06-30",
        "2020-09-30",
        "2020-12-31",
        "2021-03-31",
    ]
    target = "y"
    cols = [
        "sic",
        # "countryba",
        "solvency_debt_ratio",
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",
        "profit_net_margin",
        "profit_roa",
        # "active_dio",
        "active_acp",
    ]
    print(f"\nLoad data:\n\tDir: {data_processed_dir}\n\tFile: {data_file}")
    df = pd.read_csv(
        data_processed_dir / data_file,
        header=0,
        usecols=cols + [target] + ["ddate"],
    )
    # Trim time period
    df = df.loc[df["ddate"].isin(dev_periods)].reset_index(drop=True)
    df.drop(["ddate"], axis=1, inplace=True)
    # Save the dev data set
    df.to_csv(data_processed_dir / "model1_dev.csv", header=True, index=False)

    print(f"\tLoaded: {len(df)} rows X {len(df.columns)} cols...")
    X_dev, y_dev, X_train, y_train, X_test, y_test = get_data(df, target)

    # * Bayesian optimization
    print(f"\n---------- Bayesian optimization on TRAIN set -------------------")
    # Find opt for hypert-params on TRAIN set
    model_pack_initial = get_model_xgb()

    _, _, bayes_space = get_param_search_xgb()
    bayes_space.pop("booster")  # use for XGB
    bayes_space.pop("colsample_bytree")  # use for XGB
    # bayes_space.pop("criterion")  # use for RF

    model_pack_opted, _ = bayesian_search_cv(
        model_pack_initial,
        X_train,
        y_train,
        bayes_space,
        n_bayes_params_sets=n_bayes_params_sets,
        n_splits=n_splits,
        log_model=False,
    )

    # * Train & Evaluate model on TRAIN & unseen TEST sets
    print(f"\n---------- Train & Evaluate model on unseen TEST set ------------")
    model_pack, _, results = train_model(
        model_pack_opted,
        X_train,
        y_train,
        X_test,
        y_test,
        early_stop=True,
        log_input=False,
        log_model=False,
        set_name="train",
    )

    # * Train final model on whole DEV set
    print(f"\n------------- Train final model on DEV set-----------------------")
    _, _, _ = train_model(
        model_pack,
        X_dev,
        y_dev,
        None,
        None,
        early_stop=False,
        log_input=True,
        log_model=True,
        set_name="dev",
    )

    pd_reset_options()
    print(f"\n!!! DONE: Train the model !!!")
    winsound.Beep(frequency=3000, duration=300)


# %% RUN ========================================================================
if __name__ == "__main__":  #! Make sure that Tracking Server has been run.
    main(
        data_file="fhealth_dev_sic.csv",
        n_bayes_params_sets=50,
        n_splits=4,
    )

# %% Auxilary -------------------------------------------------------------------
# data_dir = data_processed_dir
# data_file = "fhealth_dev_sic.csv"
# n_bayes_params_sets = 2
# n_splits = 2
# log_model = False

# early_stop = False
# log_model = False
# log_input = False
# set_name = "train"
# tracking_uri = mlflow_tracking_uri
# %%
