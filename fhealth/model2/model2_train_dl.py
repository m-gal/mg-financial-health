"""
    LOAD preprocessed development dataset for model #2:
        "model2_dev.csv"
    and SPLIT it onto TRAIN & TEST sets
    no FIND the best hyperparams for a model on TRAIN set
    TRAIN ML model w/ hyperparams found out on DEV
    no LOG results with MLflow Tracking

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

# %% Setup the Tensorflow
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Setup
import os
import sys
import time
import winsound
import warnings
import json
import random  # for reproducibility

import numpy as np
import pandas as pd
from sklearn import model_selection

import mlflow

from pprint import pprint
from IPython.display import display


# %% Load project"s stuff
os.environ["NUMEXPR_MAX_THREADS"] = "48"
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])


# Load custom classes and utils
from fhealth.config import project_dir
from fhealth.config import data_processed_dir
from fhealth.config import mlflow_tracking_uri
from fhealth.config import tensorboard_dir

from fhealth.utils import pd_set_options
from fhealth.utils import pd_reset_options

from fhealth.model2._nn import encode_numeric_float_feature
from fhealth.model2._nn import encode_categorical_feature_with_defined_vocab
from fhealth.model2._nn import vanilla_multiin_multiout
from fhealth.model2._nn import honey_multiin_multiout
from fhealth.model2._nn import eval_model
from fhealth.model2._nn import plot_loss
from fhealth import plots

# from fhealth.model2._nn import fit_model
# from fhealth.model2._nn import eval_model
# from fhealth.model2._nn import predict_model


# %% Print out functions --------------------------------------------------------
def print_versions():
    print(f"\nPackages versions:")
    print(f"\tTensorflow: {tf.__version__}")
    print(f"\tKeras: {keras.__version__}")
    print(f"\tMLflow: {mlflow.__version__}")
    print(f"\tPandas: {pd.__version__}")

    print(
        "Num CPUs Available:",
        len(tf.config.experimental.list_physical_devices("CPU")),
        ":",
        tf.config.experimental.list_physical_devices("CPU"),
    )
    print(
        "Num GPUs Available:",
        len(tf.config.experimental.list_physical_devices("GPU")),
        ":",
        tf.config.experimental.list_physical_devices("GPU"),
    )


# %% Custom funcs ---------------------------------------------------------------
def df_split_data(df, rnd_state):
    print(f"\nGet train & test data ...")
    df_train_val, df_test = model_selection.train_test_split(
        df,
        train_size=0.8,
        random_state=rnd_state,
        shuffle=True,
    )

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        train_size=0.85,
        random_state=rnd_state,
        shuffle=True,
    )

    print(f"Dev set: {df.shape[1]} vars & {df.shape[0]} rows.")
    print(f"df_train: {df_train.shape[1]} vars & {df_train.shape[0]} rows.")
    print(f"df_val: {df_val.shape[1]} vars & {df_val.shape[0]} rows.")
    print(f"df_test: {df_test.shape[1]} vars & {df_test.shape[0]} rows.")
    return df_train, df_val, df_test


def df_to_ds(df, sets_of_features, batch_size=None, shuffle=True):
    df_X = df[sets_of_features["x"]].copy()

    labels = []
    for y in sets_of_features["y"]:
        labels.append(df[y])

    # The given tensors are sliced along their first dimension.
    # This operation preserves the structure of the input tensors,
    # removing the first dimension of each tensor and using it as the dataset dims.
    # All input tensors must have the same size in their first dimensions.
    ds_X = tf.data.Dataset.from_tensor_slices(dict(df_X))
    ds_labels = tf.data.Dataset.from_tensor_slices(tuple(labels))
    ds = tf.data.Dataset.zip((ds_X, ds_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_X), seed=42)

    # Combines consecutive elements of this dataset into batches.
    if batch_size is not None:
        ds = ds.batch(batch_size)
    return ds


def create_model_inputs_and_features(
    set_of_features,
    categorical_features_with_defined_vocabulary,
    dataset,
):
    print(f"Create inputs & Encode features ...")
    inputs = {}
    encoded_features = {}
    for feature_name in set_of_features["x"]:
        print(f"Feature: <{feature_name}> :")
        if feature_name in set_of_features["types"]["numeric_float"]:
            input, encoded_feature = encode_numeric_float_feature(feature_name, dataset)
        elif feature_name in set_of_features["types"]["categorical_int_w_vocab"]:
            vocabulary = categorical_features_with_defined_vocabulary[feature_name]
            (
                input,
                encoded_feature,
            ) = encode_categorical_feature_with_defined_vocab(
                feature_name, vocabulary, dataset, is_string=False
            )
        elif feature_name in set_of_features["types"]["categorical_str_w_vocab"]:
            vocabulary = categorical_features_with_defined_vocabulary[feature_name]
            (
                input,
                encoded_feature,
            ) = encode_categorical_feature_with_defined_vocab(
                feature_name, vocabulary, dataset, is_string=True
            )

        inputs[feature_name] = input
        encoded_features[feature_name] = encoded_feature
    return inputs, encoded_features


def log_model_plots(model, df, df_name, set_of_features):
    print(f"\nPlot and log model's resulting plots...")
    model_name = model.name
    dataset = df_to_ds(df, set_of_features)
    probas = model.predict(dataset.batch(len(dataset)))

    for i, y in enumerate(set_of_features["y"]):
        print(f"{y}")
        full_name = f"{y}: {model_name}"

        y_true = df[y]
        y_pred = np.where(probas[i] >= 0.5, 1, 0)
        y_prob = np.concatenate((probas[i], 1 - probas[i]), axis=1)

        df_probas = pd.DataFrame({"score": probas[i].ravel(), "true_label": y_true}).reset_index(
            drop=True
        )
        df_probas["data"] = df_name

        fig_cm_norm = plots.plot_cm(full_name, y_true, y_pred, df_name, normalize=True)
        mlflow.log_figure(fig_cm_norm, f"./plots/{y}/{df_name}_confusion_matrix_norm.png")
        display(fig_cm_norm)

        fig_roc = plots.plot_roc_curve(full_name, y_true, y_prob, df_name)
        mlflow.log_figure(fig_roc, f"./plots/{y}/{df_name}_roc_curve.png")
        display(fig_roc)

        fig_ks = plots.plot_ks_stat(full_name, y_true, y_prob, df_name)
        mlflow.log_figure(fig_ks, f"./plots/{y}/{df_name}_ks_statistic.png")
        display(fig_ks)

        figs_score = plots.plot_scores_dist(model_name, df_probas, is_scores=False)
        mlflow.log_figure(figs_score[0], f"./plots/{y}/{df_name}_proba_data_dist.png")
        mlflow.log_figure(figs_score[1], f"./plots/{y}/{df_name}_proba_label_dist.png")


# %% Main =======================================================================
def main(data_file):
    # data_file = "model2_dev.csv"

    # For reproducubility
    rnd_state = 42
    np.random.seed(rnd_state)
    random.seed(rnd_state)
    tf.random.set_seed(rnd_state)

    print(f"\n---- START: Future Financial Health Prediction Model Training --")
    warnings.filterwarnings("ignore")
    pd_set_options()
    print_versions()

    print(f"\n---- Load data from development set ----------------------------")
    print(f"Load data:\n\tDir: {data_processed_dir}\n\tFile: {data_file}")
    cols_inf = ["country", "sic"]
    cols_xx = [
        "solvency_debt_ratio",
        "liquid_current_ratio",
        "liquid_quick_ratio",
        "liquid_cash_ratio",
        "profit_net_margin",
        "profit_roa",
        "active_acp",
    ]
    cols_x_00 = ["_00_" + col for col in cols_xx]
    cols_x_01 = ["_01_" + col for col in cols_xx]
    cols_x = cols_inf + cols_x_01 + cols_x_00
    cols_y = [
        "y_solvency_debt_ratio",
        "y_liquid_current_ratio",
        "y_liquid_quick_ratio",
        "y_liquid_cash_ratio",
        "y_profit_net_margin",
        "y_profit_roa",
        "y_active_acp",
    ]

    df = pd.read_csv(
        data_processed_dir / data_file,
        header=0,
        usecols=cols_x + cols_y,
        float_precision="round_trip",
    )
    # df.drop(["ddate", "seq", "seq_rank"], axis=1, inplace=True)
    display(df.info())

    print(f"\n---- Define dataset metadata of features -----------------------")
    # A lists of the numerical feature names.
    NUMERIC_FLOAT_FEATURE_NAMES = cols_x_01 + cols_x_00
    NUMERIC_INTEGER_FEATURE_NAMES = [42]  # ["age"]
    # A lists of the binary feature names. Feature like "is_animal"
    BINARY_FEATURE_NAMES = [42]  # ["is_animal"]

    # Categorical features we want be embedded
    # Get the dictionary of the categorical features and their vocabulary for NN.
    with open(data_processed_dir / "model2_cat_feats_vocab.json", "r") as f:
        categorical_features_with_defined_vocabulary = json.load(f)
    CATEGORICAL_INTEGER_FEATURE_NAMES_W_VOCAB = [
        list(categorical_features_with_defined_vocabulary.keys())[0]
    ]
    CATEGORICAL_STRING_FEATURE_NAMES_W_VOCAB = [
        list(categorical_features_with_defined_vocabulary.keys())[1]
    ]
    # Textual features we want be embedded
    TEXT_FEATURE_NAMES = [42]

    # A dictionary of all the input columns\features.
    SETS_OF_FEATURES = {
        "y": cols_y,
        "x": cols_x,
        "sets": {
            "set_in": ["sic", "country"],
            "set_00": cols_x_00,
            "set_01": cols_x_01,
        },
        "types": {
            "numeric_float": NUMERIC_FLOAT_FEATURE_NAMES,
            "numeric_int": NUMERIC_INTEGER_FEATURE_NAMES,
            "binary": BINARY_FEATURE_NAMES,
            "categorical_int_w_vocab": CATEGORICAL_INTEGER_FEATURE_NAMES_W_VOCAB,
            "categorical_str_w_vocab": CATEGORICAL_STRING_FEATURE_NAMES_W_VOCAB,
            "text": TEXT_FEATURE_NAMES,
        },
    }
    display(SETS_OF_FEATURES)

    print(f"\n---- Split the data and convert it into tf.Dataset -------------")
    # Split the dada
    df_train, df_val, df_test = df_split_data(df, rnd_state)

    # Convert dataframe to dataset
    dataset_train = df_to_ds(df_train, SETS_OF_FEATURES)
    dataset_val = df_to_ds(df_val, SETS_OF_FEATURES)

    print(f"\nInspect the dataset's elements ...")
    pprint(dataset_train.element_spec)

    print(f"\n---- Prepare the data ------------------------------------------")
    # Let's batch the datasets:
    BATCH_SIZE = len(dataset_train)
    NEURONS = 512
    EPOCHS = 50
    ACTIVATION_LAYERS = "relu"
    ACTIVATION_OUTPUT = "sigmoid"
    OPTIMIZER = keras.optimizers.RMSprop(learning_rate=0.001)  # "adam"
    LOSS = "binary_crossentropy"
    METRICS = [
        keras.metrics.BinaryAccuracy(name="acc"),
        keras.metrics.AUC(name="auc"),
        # keras.metrics.TruePositives(name="tp"),
        # keras.metrics.FalsePositives(name="fp"),
        # keras.metrics.TrueNegatives(name="tn"),
        # keras.metrics.FalseNegatives(name="fn"),
        # keras.metrics.Precision(name="precision"),
        # keras.metrics.Recall(name="recall"),
        keras.metrics.MeanAbsoluteError(name="mae"),
    ]

    #! Batch the datasets. It is nesessary for feature encoding
    ds_train = dataset_train.batch(BATCH_SIZE)
    ds_val = dataset_val.batch(BATCH_SIZE)

    # Create inputs and encode features
    inputs, encoded_features = create_model_inputs_and_features(
        SETS_OF_FEATURES,
        categorical_features_with_defined_vocabulary,
        ds_train,
    )

    print(f"\n---- Build & Train the model -----------------------------------")

    #! Build the model
    model = honey_multiin_multiout(
        inputs,
        encoded_features,
        sets_of_inputs=SETS_OF_FEATURES["sets"],
        neurons=NEURONS,
        activation_layers=ACTIVATION_LAYERS,
        activation_output=ACTIVATION_OUTPUT,
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS,
    )

    # Plot the model
    fig_model = keras.utils.plot_model(
        model,
        to_file=project_dir / f"pics/{model.name}.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=False,
        dpi=100,
        rankdir="LR",
    )
    display(fig_model)

    # * MLflow: Setup tracking
    experiment_name = "Model#2:Search"
    run_name = f"{model.name}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # * MLflow: Enable autologging
    mlflow.keras.autolog(log_models=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        print(f"\nMLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
        print(f"MLFLOW_ARTIFACT_URI: {mlflow.get_artifact_uri()}")
        print(f"MLFLOW_EXPERIMENT_NAME: {experiment_name}")
        print(f"EXPERIMENT_ID: {experiment_id}")
        print(f"RUN_NAME: {run_name}")
        print(f"RUN_ID: {run_id}\n")

        mlflow.set_tag("tag", "test")

        # Fit the model
        fit_time = time.strftime("%Y%m%d-%H%M")
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[
                # # folder [tensorboard] -> run Terminal:
                # # >>>tensorboard --logdir=<tensorboard_dir>
                # keras.callbacks.TensorBoard(
                #     log_dir=tensorboard_dir
                #     / fit_time
                # ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    verbose=1,
                    mode="max",
                ),
                # keras.callbacks.ModelCheckpoint(
                #     filepath=FILEPATH.as_posix(),
                #     monitor="val_auc",
                #     verbose=1,
                #     mode="max",
                #     save_best_only=True,
                # ),
                # keras.callbacks.LearningRateScheduler(
                #     scheduler_exp, verbose=1,
                # ),
            ],
        )
        # * Disable autologging
        mlflow.keras.autolog(disable=True)

        # * Log params
        mlflow.log_params(
            {
                "neurons": NEURONS,
                "hidden_activation": ACTIVATION_LAYERS,
            }
        )
        # * Log metrics averages
        _, eval_scores_avg_train = eval_model(model, dataset_train, BATCH_SIZE, "train")
        _, eval_scores_avg_val = eval_model(model, dataset_val, BATCH_SIZE, "val")
        my_metrics = {
            **eval_scores_avg_train,
            **eval_scores_avg_val,
        }
        mlflow.log_metrics(my_metrics)

        # * Log plots
        mlflow.log_artifact(project_dir / f"pics/{model.name}.png")
        # Plot the model's loss
        fig_loss = plot_loss(history, model.name)
        mlflow.log_figure(fig_loss, "./plots/loss.png")
        display(fig_loss)

        log_model_plots(model, df_train, "train", SETS_OF_FEATURES)
        log_model_plots(model, df_val, "val", SETS_OF_FEATURES)

        model.reset_states()

    print(f"\n---- Check for predictions -------------------------------------")
    sample = df_test.loc[[11783]]
    # sample = df_test.sample(n=1)
    sample_X = sample[SETS_OF_FEATURES["x"]].to_dict("r")[0]
    sample_Y = sample[SETS_OF_FEATURES["y"]].to_dict("r")[0]

    sample_input = {name: tf.convert_to_tensor([value]) for name, value in sample_X.items()}
    predictions = model.predict(sample_input)

    print(f"Inference on random sample from the test data:")
    print(f"\t1 - financial wellness will be better then current")
    print(f"\t0 - financial  wellness will be worser\n")
    for i, p in enumerate(predictions):
        ratio = list(sample_Y.keys())[i][2:]
        real = list(sample_Y.values())[i]
        print(f"{ratio} (true value: {real}):\thave a {p[0][0]:.1%} probability of will be better")

    pd_reset_options()
    print(f"\n!!! DONE: Train the model !!!")
    winsound.Beep(frequency=3000, duration=300)


# %% RUN ========================================================================
if __name__ == "__main__":  #! Make sure that Tracking Server has been run.
    main(data_file="model2_dev.csv")

# %%
