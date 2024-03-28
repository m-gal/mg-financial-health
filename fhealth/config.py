"""
    # notebook.ipynb
    |   from projectname.config import data_path
    |   import pandas as pd
    |   df = pd.read_csv(data_path)  # clean!
By using these config.py files,
we get clean code in exchange for an investment of time naming variables logically.
"""

# %% Import libraries
from pathlib import Path

# %% Define project's paths
p = Path(__file__).parent.resolve()
project_dir = p.parent.resolve()

data_raw_dir = project_dir / "data/raw"
data_processed_dir = project_dir / "data/processed"
data_precleaned_dir = project_dir / "data/precleaned"

docs_dir = project_dir / "docs"
mlflow_dir = project_dir / "mlflow"
models_dir = project_dir / "models"
pipelines_dir = project_dir / "pipelines"
reports_dir = project_dir / "reports"
temp_dir = project_dir / "temp"
tensorboard_dir = project_dir / "tensorboard"

# %% Some project stuff
MODEL_DIR = "mlruns/2/bb368abba9e042ab957c53b7a8b19ebc/artifacts/model"

model1_path = mlflow_dir / MODEL_DIR
mlflow_tracking_uri = "../mlflow/mlruns"
