# ------------------------------------------------------------------------------
# ----------------------------- X G B O O S T ----------------------------------
# --------------- ---------- C L A S S I F I E R -------------------------------
# ------------------------------------------------------------------------------
# %% Import models's libraries
import xgboost as xgb
from skopt.space import Real, Categorical, Integer


# %% Define parameters and model ------------------------------------------------
def model():
    """
    XGBoost Classifier for Scikit-Learn API with default params
    * booster: default= gbtree
        Which booster to use. Can be gbtree, gblinear or dart;
        gbtree and dart use tree based models while gblinear uses linear functions.
    * objective: default=binary:logistic
        Specify the learning task and the corresponding learning objective.
        The classification objective options are below:
            - binary:logistic: logistic regression for binary classification,
                output probability.
            - binary:logitraw: logistic regression for binary classification,
                output score before logistic transformation.
            - binary:hinge: hinge loss for binary classification.
                This makes predictions of 0 or 1, rather than producing probabilities.
            - multi:softmax: set XGBoost to do multiclass classification using the softmax objective,
                you also need to set num_class(number of classes).
            - multi:softprob: same as softmax, but output a vector of ndata * nclass,
                which can be further reshaped to ndata * nclass matrix.
                The result contains predicted probability of each data point belonging to each class.
    * n_estimators: int, default=100. Number of boosting rounds.
    * subsample: default=1. Lower ratios avoid over-fitting
    * colsample_bytree: default=1. Lower ratios avoid over-fitting.
    * max_depth: default=6. Lower ratios avoid over-fitting.
    * min_child_weight: default=1. Larger values avoid over-fitting.
    * eta (lr): default=0.1. Lower values avoid over-fitting.
    * reg_lambda: default=1. L2 regularization:. Larger values avoid over-fitting.
    * reg_alpha: default=0. L1 regularization. Larger values avoid over-fitting.
    * gamma: default=0. Larger values avoid over-fitting.
    * scale_pos_weight: default=1]. A value greater than 0 should be used
        * in case of high class imbalance as it helps in faster convergence.
    """
    rnd_state = 42
    model_params = {
        # "tree_method": "gpu_hist", #!  for GPU exploiting
        # "gpu_id": 0, #!  for GPU exploiting
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 1000,
        "subsample": 1,
        "colsample_bytree": 1,
        "max_depth": 6,
        "min_child_weight": 1,
        "learning_rate": 0.1,
        "reg_lambda": 1,
        "reg_alpha": 0,
        "gamma": 0,
        "scale_pos_weight": 1,
        "n_jobs": -1,
        "random_state": rnd_state,
    }
    # fit_params = {
    #     "early_stopping_rounds": 50,
    #     "eval_metric": "auc",
    #     "eval_set": [(X_val, y_val)],
    # }
    model = xgb.XGBClassifier(**model_params)
    model_name = type(model).__name__

    return (model, model_name, model_params)


# %% --------------------------- Search CV --------------------------------------
def param_search():
    # Parameters' distributions tune in case RANDOMIZED grid search
    ## Dictionary with parameters names (str) as keys and distributions
    ## or lists of parameters to try.
    ## If a list is given, it is sampled uniformly.
    params_dist = {
        "n_estimators": [x for x in range(50, 1001, 50)],
        "subsample": [x / 10 for x in range(5, 11, 1)],
        "colsample_bytree": [x / 10 for x in range(6, 11, 1)],
        "max_depth": [x for x in range(6, 51, 4)],
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "reg_lambda": [1],  # + [x for x in range(1, 11, 1)],
        "reg_alpha": [0],  # + [x for x in range(0, 11, 1)],
        "gamma": [0.0],  # + [x/10 for x in range(5, 60, 5)]
    }
    # Parameters what we wish to tune in case SIMPLE grid search
    ## Dictionary with parameters names (str) as keys
    ## and lists of parameter settings to try as values
    params_grid = {
        "n_estimators": [2400, 2500, 2600],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "max_depth": [6, 12, 24],
        "min_child_weight": [1] + [x for x in range(2, 11, 2)],
        "learning_rate": [0.01, 0.1, 0.3],
        "reg_lambda": [0.5, 1, 2],
        "reg_alpha": [0, 0.5, 1],
        "gamma": [0, 1, 2, 5],
    }
    # Their core idea of Bayesian Optimization is simple:
    # when a region of the space turns out to be good, it should be explored more.
    # Real: Continuous hyperparameter space.
    # Integer: Discrete hyperparameter space.
    # Categorical: Categorical hyperparameter space.
    bayes_space = {
        "booster": Categorical(["gbtree", "dart", "gblinear"]),
        "n_estimators": Integer(100, 3000),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.7, 1.0),
        "max_depth": Integer(3, 9),
        "min_child_weight": Integer(10, 20),
        "learning_rate": Real(0.001, 0.3),
        "reg_lambda": Real(15, 20),
        "reg_alpha": Real(15, 20),
        "gamma": Real(10, 15),
    }

    return params_dist, params_grid, bayes_space
