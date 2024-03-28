"""
    Stub to do one-off prediction using the trained model.

    @author: mikhail.galkin
"""

# %% Load libraries -------------------------------------------------------------
import sys
import warnings
from sklearn.metrics import brier_score_loss

# %% Load project's stuff -------------------------------------------------------
sys.path.extend(["..", "../..", "../../..", ".", "./.", "././."])
from fhealth.config import data_processed_dir
from fhealth.config import model1_path
from fhealth.utils import model1_load
from fhealth.utils import model1_make_example_for


# %% Some funcs -----------------------------------------------------------------
def predict_example_df(model, examples):
    # examples = model1_make_example_for()
    for e in examples:
        print(e[0])
        # Make prediction
        prediction = model.predict_proba(e[0])[:, 1]
        err = brier_score_loss(e[1], prediction)
        # Print out
        print(f"\nReal Value: {e[1]}")
        print(f"Predicted Value: {list(prediction)}")
        print(f"Brier Score: {err}")
        print(f"* Brier score is a measure of how far your predictions lie from the true values\n")

    return list(prediction)


# %% Main -----------------------------------------------------------------------
def main(model1_path=model1_path):
    """Checks a one-off prediction using the trained model.

    Args:
        * model1_path (str): Folder's name for separatly saved model.\
            Defaults to None.
    """
    print(f"--------------- START: Check an one-off prediction ----------------")
    print(f"Model path: {model1_path}")
    warnings.filterwarnings("ignore")
    data_dir = data_processed_dir
    data_file = "model1_dev.csv"

    if model1_path is None:
        print(f"No model's name...")
        pass
    else:
        # # For model have been saved aside:
        model = model1_load(model1_path)
        examples = model1_make_example_for(data_dir, data_file)
        prediction = predict_example_df(model=model, examples=examples)
        print(f"!!! DONE: Check an one-off prediction !!!")

    return prediction


# %% Run ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
