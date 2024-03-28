"""
    RESTful API with FastAPI package

    You can run the application server using
        >>> uvicorn app.main:app --reload
    Here app.main indicates you use main.py file inside the 'app' directory
    and :app indicates our FastAPI instance name.

    command: uvicorn {folder}.{module}:app --reload

    open Terminal in ./fhealth and run:
        >>> uvicorn api.fast_api:app --reload
        ! need to change model1_path from "../path" to "./path"
    or better open Terminal in ./fhealth/api and run:
        >>> uvicorn fast_api:app --reload

    !Example Value for testing:
    {
        "sic": 3420,
        "solvency_debt_ratio": 0.522054,
        "liquid_current_ratio": 6.336683,
        "liquid_quick_ratio": 2.558935,
        "liquid_cash_ratio": 0.334832,
        "profit_net_margin": 0.025725,
        "profit_roa": 0.029588,
        "active_acp": 72.378706
    }

    !Response body must be:
    {
        "general score": 561
    }

    @author: mikhail.galkin
"""

# %% Load libraries
import uvicorn
import mlflow
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initializing a FastAPI App Instance
app = FastAPI()


# Define request body
class input_data(BaseModel):
    sic: int
    solvency_debt_ratio: float
    liquid_current_ratio: float
    liquid_quick_ratio: float
    liquid_cash_ratio: float
    profit_net_margin: float
    profit_roa: float
    active_acp: float


# Load model
## if Terminal run from ./fhealth => 1 dot in "./mlflow/mlruns/path_to_model"
model1_path = "../../mlflow/mlruns/2/072dfc9af5b84ae098a5981e91b620b5/artifacts/model"
model = mlflow.xgboost.load_model(model1_path)
print(f"\nLoad model from...")
print(f"{model1_path}")


# Defining a Simple GET Request
@app.get("/finscore/")
def get_root():
    return {"Welcome": "The General Financial Scoring API"}


# Creating an Endpoint to recieve the data to make prediction on.
@app.post("/finscore/predict")
async def predict(data: input_data):

    data_list = [
        {
            "sic": data.sic,
            "solvency_debt_ratio": data.solvency_debt_ratio,
            "liquid_current_ratio": data.liquid_current_ratio,
            "liquid_quick_ratio": data.liquid_quick_ratio,
            "liquid_cash_ratio": data.liquid_cash_ratio,
            "profit_net_margin": data.profit_net_margin,
            "profit_roa": data.profit_roa,
            "active_acp": data.active_acp,
        }
    ]
    df = pd.DataFrame.from_records(data_list)
    dmx = xgb.DMatrix(df)
    score = round(model.predict(dmx)[0] * 1000)

    return {"score": score}


if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)
