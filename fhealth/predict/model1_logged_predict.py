"""
    Helps to get MLflow's runs results from
    MLflow LocalArtifactRepository and FileStore
    fetch best model
    make predicition on a Pandas DataFrame with examples from sample data

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import mlflow
from sklearn.metrics import brier_score_loss

# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.config import mlflow_tracking_uri
from fhealth.utils import model1_make_example_for


# %% Some funcs -----------------------------------------------------------------
def get_best_model(tracking_uri, experiment_name, metrics):
    # * Setup MLflow tracking server
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Get runs
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)
    df_runs.sort_values([metrics], ascending=True, inplace=True)

    # Get the best one
    # The best model has a specific runid that can be used to execute the deployment.
    best_runid = df_runs["run_id"].head(1).values[0]
    best_metrics = df_runs[[metrics]].head(1).values[0]

    logged_model = f"{tracking_uri}/{experiment_id}/{best_runid}/artifacts/model"

    print(f"\nExperiment's name: {experiment_name}")
    print(f"Amount of runs: {len(df_runs)}")
    print(f"TOP 3 of runs:")
    print(df_runs[["run_id", metrics, "tags.mlflow.runName"]].head(3))
    print(f"\nThe best run_id: {best_runid} w/ best '{metrics}' = {best_metrics}")

    return logged_model


# %% Main -----------------------------------------------------------------------
def main(
    experiment_name="Model#1:Train",
    metrics="metrics.log_loss-dev",
):
    data_dir = data_processed_dir
    data_file = "model1_dev.csv"
    tracking_uri = mlflow_tracking_uri
    logged_model = get_best_model(tracking_uri, experiment_name, metrics)
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)
    # Get examples
    examples = model1_make_example_for(data_dir, data_file)
    # Predict
    prediction = model.predict(examples[0][0])
    err = brier_score_loss(examples[0][1], prediction)
    # Print out
    print(f"\nFor sample:\n{examples[0][0]}")
    print(f"Real Value: {examples[0][1]}")
    print(f"Predicted Value: {prediction}")
    print(f"Brier Score: {err}\n")

    print("--------------------------------------------------------------------")
    print("To serve (deploy) this model locally")
    print("open a new Terminal in the ../mlflow and run the command:")
    print(f">  mlflow models serve -m {logged_model} -p 8080")
    print("--------------------------------------------------------------------")

    return prediction


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
