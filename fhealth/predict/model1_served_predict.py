"""
    Script to make predictions on MLflow locally deployed model.

    To serve (deploy) the model locally, open Terminal in the ./mlflow and run:
    $ mlflow models serve -m <Full_Path_to_your_model> -p 8080

    where <Full_Path_to_your_model> can be drawn from the model Artifacts
    in MLflow UI or using Python API,
    or it may be a your own custom folder on local machine.

    $ mlflow models serve -m ../mlflow/mlruns/2/e3064f999afb43f1bdcbb33941f3be25/artifacts/model -p 8080
    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import requests

sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.utils import model1_make_example_for


# %% Main -----------------------------------------------------------------------
def main(
    port="8080",  # 8080: serve as MLflow Model. 1234: serve from MLflow Regisry
    host="127.0.0.1",
    headers={"Content-Type": "application/json; format=pandas-split"},
):
    request_uri = f"http://{host}:{port}/invocations"

    data_dir = data_processed_dir
    data_file = "model1_dev.csv"
    examples = model1_make_example_for(data_dir, data_file)
    example_df = examples[0][0]
    example_json = example_df.to_json(orient="split")
    print(f"{request_uri}")
    print(example_df, "\n")
    print(example_json)

    try:
        response = requests.post(url=request_uri, data=example_json, headers=headers)
        print(f"\nModel response is:")
        print(response.content)
        print("!!! DONE !!!")
    except Exception as ex:
        raise (ex)


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
