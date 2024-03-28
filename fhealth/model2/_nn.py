# ------------------------------------------------------------------------------
# ----------------------------- N E U R A L ------------------------------------
# -------------------------------- N E T ---------------------------------------
# ------------------------------------------------------------------------------
# %% Setup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import math
import matplotlib.pyplot as plt

from IPython.display import display

#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ U T I L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""#%% Function for demonstrate several types of feature column ----------------
# TensorFlow provides many types of feature columns.
# In this section, we will create several types of feature columns,
# and demonstrate how they transform a column from the dataframe.
## We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(ds_val))[0]
# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())
"""


def encode_numeric_float_feature(numeric_feature_name, dataset):
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[numeric_feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create model input for feature
    input = layers.Input(shape=(1,), name=numeric_feature_name)
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization(name=f"normalized_{numeric_feature_name}")
    print(f"\tLearn the statistics of the feature's data ...")
    normalizer.adapt(feature_ds)
    print(f"\tNormalize the input feature ...")
    encoded_feature = normalizer(input)
    #! It is possible to not use the Normalizing:
    # encoded_feature = input

    return input, encoded_feature


def encode_categorical_feature_with_defined_vocab(
    categorical_feature_name, vocab, dataset, is_string
):
    if is_string:
        lookup_class = layers.StringLookup
        dtype_class = "string"
    else:
        lookup_class = layers.IntegerLookup
        dtype_class = "int32"

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[categorical_feature_name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Create model input for feature
    input = layers.Input(
        shape=(1,),
        name=categorical_feature_name,
        dtype=dtype_class,
    )
    # Since we are not using a mask token we set mask_token to None,
    # and possible expecting any out of vocabulary (oov) token - set num_oov_indices to 1.
    lookup = lookup_class(vocabulary=vocab, name=f"lookup_{categorical_feature_name}")
    print(f"\tTurn the categorical input into integer indices ...")
    encoded_feature = lookup(input)

    print(f"\tCreate an embedding layer with the specified dimensions ...")
    #! It is possible to not use the embedding. Just comment the below.
    embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
    embedding = layers.Embedding(
        input_dim=lookup.vocabulary_size(),
        output_dim=embedding_dims,
        name=f"lookuped_{categorical_feature_name}",
    )
    embedded_feature = embedding(encoded_feature)
    flatten = layers.Flatten(name=f"embedded_{categorical_feature_name}")
    encoded_feature = flatten(embedded_feature)

    return input, encoded_feature


# %% Create an input pipeline using tf.data
def eval_model(model, dataset, batch_size, dataset_name):
    print(f"\nModel evaluation for: {dataset_name} ...")
    evals = model.evaluate(dataset.batch(batch_size), verbose=0)
    metrics_names = [name + f"_{dataset_name}" for name in model.metrics_names]
    eval_scores = dict(zip(metrics_names, evals))

    # Calculate average for some metrics
    metrics = ["_loss_", "_acc_", "_auc_", "_mae_"]
    eval_scores_avg = {}
    for metric in metrics:
        values = [value for key, value in eval_scores.items() if metric in key]
        avg_metric = sum(values) / len(values)
        eval_scores_avg[f"_avg{metric}{dataset_name}"] = avg_metric

    return eval_scores, eval_scores_avg


def predict_model(model, dataset, batch_size, dataset_name):
    print("\nModel prediction for: " + dataset_name)
    prediction = model.predict(dataset.batch(batch_size))
    print(dataset_name + " predicted")
    return prediction


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~ P L O T S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def plot_loss(history, nn_name):
    plt.figure(figsize=(12, 8))
    # * Var#1: Use a log scale to show the wide range of values.
    # plt.semilogy(
    #     history.epoch, history.history["loss"], label="Train: " + nn_name
    # )
    # plt.semilogy(
    #     history.epoch,
    #     history.history["val_loss"],
    #     label="Val: " + nn_name,
    #     linestyle="--",
    # )
    # * Var#2: W/o a log scaling.
    plt.plot(history.epoch, history.history["loss"], label="Train: " + nn_name)
    plt.plot(
        history.epoch,
        history.history["val_loss"],
        label="Val: " + nn_name,
        linestyle="--",
    )

    plt.title(nn_name + ": model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


def plot_metrics(history, metrics, nn_name):
    plt.figure(figsize=(12, 12))
    metrics_name = [metric.name for metric in metrics]
    for n, metric in enumerate(metrics_name):
        plt.subplot(2, 1, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train: " + nn_name)
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            linestyle="--",
            label="Val: " + nn_name,
        )
        plt.xlabel("epoch")
        plt.ylabel(metric)
        y_min = 0.99 * min(min(history.history[metric]), min(history.history["val_" + metric]))
        plt.ylim([y_min, 1])
        plt.legend()
    # get current figure
    fig = plt.gcf()
    plt.close()
    return fig


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~ M O D E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def vanilla_multiin(
    inputs,
    encoded_features,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    all = layers.concatenate(encoded_features_list, name="all")
    x = layers.Dense(units=neurons, activation=activation_layers, name="dense1")(all)
    outputs = layers.Dense(1, activation=activation_output, name="outputs")(x)

    model = keras.Model(
        inputs_list,
        outputs,
        name="vanilla_multiin",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model


def vanilla_multiin_multiout(
    inputs: dict,
    encoded_features: dict,
    sets_of_inputs: dict,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    # Define sets
    sets_of_inputs["set_in"]
    sets_of_inputs["set_00"]
    sets_of_inputs["set_01"]

    encoded_features_list_in = [encoded_features[x] for x in sets_of_inputs["set_in"]]
    encoded_features_list_00 = [encoded_features[x] for x in sets_of_inputs["set_00"]]
    encoded_features_list_01 = [encoded_features[x] for x in sets_of_inputs["set_01"]]

    # Constuct the model graph
    set_in = layers.concatenate(encoded_features_list_in, name="set_in")
    set_00 = layers.concatenate(encoded_features_list_00, name="set_00")
    set_01 = layers.concatenate(encoded_features_list_01, name="set_01")

    # * Multi Inputs
    x_in = layers.Dense(units=neurons, activation=activation_layers, name="dense_in")(set_in)
    x_00 = layers.Dense(units=neurons, activation=activation_layers, name="dense_00")(set_00)
    x_01 = layers.Dense(units=neurons, activation=activation_layers, name="dense_01")(set_01)

    # * United model's part
    all = layers.concatenate([x_in, x_00, x_01], name="all")
    x = layers.Dense(units=neurons, activation=activation_layers, name="dense_all")(all)

    # * Multioutputs
    out_solvency_debt_ratio = layers.Dense(
        1, activation=activation_output, name="solvency_debt_ratio"
    )(x)
    out_liquid_current_ratio = layers.Dense(
        1, activation=activation_output, name="liquid_current_ratio"
    )(x)
    out_liquid_quick_ratio = layers.Dense(
        1, activation=activation_output, name="liquid_quick_ratio"
    )(x)
    out_liquid_cash_ratio = layers.Dense(1, activation=activation_output, name="liquid_cash_ratio")(
        x
    )
    out_profit_net_margin = layers.Dense(1, activation=activation_output, name="profit_net_margin")(
        x
    )
    out_profit_roa = layers.Dense(1, activation=activation_output, name="profit_roa")(x)
    out_active_acp = layers.Dense(1, activation=activation_output, name="active_acp")(x)

    outputs = [
        out_solvency_debt_ratio,
        out_liquid_current_ratio,
        out_liquid_quick_ratio,
        out_liquid_cash_ratio,
        out_profit_net_margin,
        out_profit_roa,
        out_active_acp,
    ]
    model = keras.Model(
        inputs_list,
        outputs,
        name="vanilla_multiin_multiout",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model


def honey_multiin_multiout(
    inputs: dict,
    encoded_features: dict,
    sets_of_inputs: dict,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=True,
):
    # # You needs 'input' and 'encoded_features' as a lists
    inputs_list = list(inputs.values())
    encoded_features_list = list(encoded_features.values())

    # Define sets
    sets_of_inputs["set_in"]
    sets_of_inputs["set_00"]
    sets_of_inputs["set_01"]

    encoded_features_list_in = [encoded_features[x] for x in sets_of_inputs["set_in"]]
    encoded_features_list_00 = [encoded_features[x] for x in sets_of_inputs["set_00"]]
    encoded_features_list_01 = [encoded_features[x] for x in sets_of_inputs["set_01"]]

    # Constuct the model graph
    set_in = layers.concatenate(encoded_features_list_in, name="set_in")
    set_00 = layers.concatenate(encoded_features_list_00, name="set_00")
    set_01 = layers.concatenate(encoded_features_list_01, name="set_01")

    # * Multi Inputs
    x_in = layers.Dense(units=neurons, activation=activation_layers, name="dense_in")(set_in)
    x_00 = layers.Dense(units=neurons, activation=activation_layers, name="dense_00")(set_00)
    x_01 = layers.Dense(units=neurons, activation=activation_layers, name="dense_01")(set_01)

    # * United model's part
    all = layers.concatenate([x_in, x_00, x_01], name="all")
    x = layers.Dense(units=neurons, activation=activation_layers, name="dense_1")(all)
    x = layers.Dense(units=neurons / 2, activation=activation_layers, name="dense_2")(x)

    # * Multi Outputs
    out_solvency_debt_ratio = layers.Dense(
        1, activation=activation_output, name="solvency_debt_ratio"
    )(x)
    out_liquid_current_ratio = layers.Dense(
        1, activation=activation_output, name="liquid_current_ratio"
    )(x)
    out_liquid_quick_ratio = layers.Dense(
        1, activation=activation_output, name="liquid_quick_ratio"
    )(x)
    out_liquid_cash_ratio = layers.Dense(1, activation=activation_output, name="liquid_cash_ratio")(
        x
    )
    out_profit_net_margin = layers.Dense(1, activation=activation_output, name="profit_net_margin")(
        x
    )
    out_profit_roa = layers.Dense(1, activation=activation_output, name="profit_roa")(x)
    out_active_acp = layers.Dense(1, activation=activation_output, name="active_acp")(x)

    outputs = [
        out_solvency_debt_ratio,
        out_liquid_current_ratio,
        out_liquid_quick_ratio,
        out_liquid_cash_ratio,
        out_profit_net_margin,
        out_profit_roa,
        out_active_acp,
    ]
    model = keras.Model(
        inputs_list,
        outputs,
        name="honey_multiin_multiout",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model
